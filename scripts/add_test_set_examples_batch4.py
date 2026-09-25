"""
Fourth batch: one more pair for Integer Overflow, the last category still at
the original count of 4.

Source: solana-labs/solana-program-library/governance/program/src/processor/
process_revoke_governing_tokens.rs (never used anywhere in training or the
existing test set -- verified by grep). SAFE is the real function
(checked_sub().ok_or(...) on the deposit amount); VULNERABLE replaces it with
a raw `-`, the same synthesis technique used for every other Integer
Overflow pair in this dataset.
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

CODE_IO_SAFE = """pub fn process_revoke_governing_tokens(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governing_token_holding_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governing_token_mint_info = next_account_info(account_info_iter)?;
    let revoke_authority_info = next_account_info(account_info_iter)?;
    let realm_config_info = next_account_info(account_info_iter)?;
    let spl_token_info = next_account_info(account_info_iter)?;

    let realm_data = get_realm_data(program_id, realm_info)?;
    realm_data.assert_is_valid_governing_token_mint_and_holding(
        program_id,
        realm_info.key,
        governing_token_mint_info.key,
        governing_token_holding_info.key,
    )?;

    let realm_config_data =
        get_realm_config_data_for_realm(program_id, realm_config_info, realm_info.key)?;
    realm_config_data
        .assert_can_revoke_governing_token(&realm_data, governing_token_mint_info.key)?;

    let mut token_owner_record_data = get_token_owner_record_data_for_realm_and_governing_mint(
        program_id,
        token_owner_record_info,
        realm_info.key,
        governing_token_mint_info.key,
    )?;

    if *revoke_authority_info.key == token_owner_record_data.governing_token_owner {
        if !revoke_authority_info.is_signer {
            return Err(GovernanceError::GoverningTokenOwnerMustSign.into());
        }
    } else {
        assert_spl_token_mint_authority_is_signer(governing_token_mint_info, revoke_authority_info)?;
    }

    // checked_sub rejects the instruction outright if amount exceeds the
    // recorded deposit, instead of silently wrapping around to a huge
    // unsigned value.
    token_owner_record_data.governing_token_deposit_amount = token_owner_record_data
        .governing_token_deposit_amount
        .checked_sub(amount)
        .ok_or(GovernanceError::InvalidRevokeAmount)?;

    token_owner_record_data.serialize(&mut token_owner_record_info.data.borrow_mut()[..])?;

    burn_spl_tokens_signed(
        governing_token_holding_info,
        governing_token_mint_info,
        realm_info,
        &get_realm_address_seeds(&realm_data.name),
        program_id,
        amount,
        spl_token_info,
    )?;

    Ok(())
}"""

CODE_IO_VULN = """pub fn process_revoke_governing_tokens(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governing_token_holding_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governing_token_mint_info = next_account_info(account_info_iter)?;
    let revoke_authority_info = next_account_info(account_info_iter)?;
    let realm_config_info = next_account_info(account_info_iter)?;
    let spl_token_info = next_account_info(account_info_iter)?;

    let realm_data = get_realm_data(program_id, realm_info)?;
    realm_data.assert_is_valid_governing_token_mint_and_holding(
        program_id,
        realm_info.key,
        governing_token_mint_info.key,
        governing_token_holding_info.key,
    )?;

    let realm_config_data =
        get_realm_config_data_for_realm(program_id, realm_config_info, realm_info.key)?;
    realm_config_data
        .assert_can_revoke_governing_token(&realm_data, governing_token_mint_info.key)?;

    let mut token_owner_record_data = get_token_owner_record_data_for_realm_and_governing_mint(
        program_id,
        token_owner_record_info,
        realm_info.key,
        governing_token_mint_info.key,
    )?;

    if *revoke_authority_info.key == token_owner_record_data.governing_token_owner {
        if !revoke_authority_info.is_signer {
            return Err(GovernanceError::GoverningTokenOwnerMustSign.into());
        }
    } else {
        assert_spl_token_mint_authority_is_signer(governing_token_mint_info, revoke_authority_info)?;
    }

    // Raw subtraction instead of checked_sub: if amount is larger than the
    // recorded deposit, this wraps around to a huge u64 instead of erroring,
    // corrupting the owner's recorded deposit balance.
    token_owner_record_data.governing_token_deposit_amount =
        token_owner_record_data.governing_token_deposit_amount - amount;

    token_owner_record_data.serialize(&mut token_owner_record_info.data.borrow_mut()[..])?;

    burn_spl_tokens_signed(
        governing_token_holding_info,
        governing_token_mint_info,
        realm_info,
        &get_realm_address_seeds(&realm_data.name),
        program_id,
        amount,
        spl_token_info,
    )?;

    Ok(())
}"""


def write_entry(idx, vulnerability, owasp_code, label, code):
    fname = f"solana_{idx:02d}.json"
    path = os.path.join(TEST_DIR, fname)
    if os.path.exists(path):
        raise FileExistsError(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"vulnerability": vulnerability, "owasp_code": owasp_code, "label": label, "smart_contract": code},
            f,
            indent=4,
        )
    return fname


def main():
    existing = [f for f in os.listdir(TEST_DIR) if f.startswith("solana_") and f.endswith(".json")]
    next_idx = len(existing) + 1

    written = [
        write_entry(next_idx, "integer_overflow", "V8", "VULNERABLE", CODE_IO_VULN),
        write_entry(next_idx + 1, "not_vulnerable", "V8", "SAFE", CODE_IO_SAFE),
    ]

    print(f"Wrote {len(written)} new test files:")
    for w in written:
        print(" ", w)
    print(f"Test set size: {len(existing)} -> {len(existing) + len(written)}")


if __name__ == "__main__":
    main()

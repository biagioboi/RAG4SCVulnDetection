"""
Third batch of test-set expansion, finally landing the two categories that
were skipped in batch 1 for lack of an unambiguous real match: Unchecked
External Calls and Type Confusion (both weakest so far).

Sources (neither used anywhere in training or the existing test set --
verified by grep):
  - solana-program/stake-pool/program/src/processor.rs, fn
    process_withdraw_sol (trimmed of unrelated fee/authority checks to keep
    the example focused): 3 real CPI calls (token_burn, token_transfer,
    stake_withdraw), all correctly propagated with `?`. VULNERABLE variant
    discards the stake_withdraw CPI result with `let _ =`, so a failed
    on-chain SOL withdrawal goes unnoticed while the pool's bookkeeping
    (pool_token_supply, total_lamports) is updated as if it had succeeded.
  - solana-labs/solana-program-library/governance/program/src/processor/
    process_add_signatory.rs (Type Confusion): SAFE is the real function
    (loads governance/proposal state through typed, discriminator-checking
    getters); VULNERABLE is a synthesized variant using raw try_from_slice
    for both, same technique as the other governance-derived pairs added
    this session.
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

# ============================================================
# Unchecked External Calls -- stake-pool/process_withdraw_sol (trimmed)
# ============================================================

CODE_UC_SAFE = """fn process_withdraw_sol(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    pool_tokens: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let stake_pool_info = next_account_info(account_info_iter)?;
    let withdraw_authority_info = next_account_info(account_info_iter)?;
    let user_transfer_authority_info = next_account_info(account_info_iter)?;
    let burn_from_pool_info = next_account_info(account_info_iter)?;
    let reserve_stake_info = next_account_info(account_info_iter)?;
    let destination_lamports_info = next_account_info(account_info_iter)?;
    let pool_mint_info = next_account_info(account_info_iter)?;
    let clock_info = next_account_info(account_info_iter)?;
    let stake_history_info = next_account_info(account_info_iter)?;
    let token_program_info = next_account_info(account_info_iter)?;

    check_account_owner(stake_pool_info, program_id)?;
    let mut stake_pool = try_from_slice_unchecked::<StakePool>(&stake_pool_info.data.borrow())?;

    stake_pool.check_authority_withdraw(withdraw_authority_info.key, program_id, stake_pool_info.key)?;
    let decimals = stake_pool.check_mint(pool_mint_info)?;
    stake_pool.check_reserve_stake(reserve_stake_info)?;

    let withdraw_lamports = stake_pool
        .calc_lamports_withdraw_amount(pool_tokens)
        .ok_or(StakePoolError::CalculationFailure)?;
    if withdraw_lamports == 0 {
        return Err(StakePoolError::WithdrawalTooSmall.into());
    }

    Self::token_burn(
        token_program_info.clone(),
        burn_from_pool_info.clone(),
        pool_mint_info.clone(),
        user_transfer_authority_info.clone(),
        pool_tokens,
    )?;

    // Every CPI result is propagated with `?`, including the stake
    // withdrawal below -- if it fails, the instruction aborts before any
    // pool bookkeeping is updated.
    Self::stake_withdraw(
        stake_pool_info.key,
        reserve_stake_info.clone(),
        withdraw_authority_info.clone(),
        AUTHORITY_WITHDRAW,
        stake_pool.stake_withdraw_bump_seed,
        destination_lamports_info.clone(),
        clock_info.clone(),
        stake_history_info.clone(),
        withdraw_lamports,
    )?;

    stake_pool.pool_token_supply = stake_pool
        .pool_token_supply
        .checked_sub(pool_tokens)
        .ok_or(StakePoolError::CalculationFailure)?;
    stake_pool.total_lamports = stake_pool
        .total_lamports
        .checked_sub(withdraw_lamports)
        .ok_or(StakePoolError::CalculationFailure)?;
    borsh::to_writer(&mut stake_pool_info.data.borrow_mut()[..], &stake_pool)?;

    Ok(())
}"""

CODE_UC_VULN = """fn process_withdraw_sol(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    pool_tokens: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let stake_pool_info = next_account_info(account_info_iter)?;
    let withdraw_authority_info = next_account_info(account_info_iter)?;
    let user_transfer_authority_info = next_account_info(account_info_iter)?;
    let burn_from_pool_info = next_account_info(account_info_iter)?;
    let reserve_stake_info = next_account_info(account_info_iter)?;
    let destination_lamports_info = next_account_info(account_info_iter)?;
    let pool_mint_info = next_account_info(account_info_iter)?;
    let clock_info = next_account_info(account_info_iter)?;
    let stake_history_info = next_account_info(account_info_iter)?;
    let token_program_info = next_account_info(account_info_iter)?;

    check_account_owner(stake_pool_info, program_id)?;
    let mut stake_pool = try_from_slice_unchecked::<StakePool>(&stake_pool_info.data.borrow())?;

    stake_pool.check_authority_withdraw(withdraw_authority_info.key, program_id, stake_pool_info.key)?;
    let decimals = stake_pool.check_mint(pool_mint_info)?;
    stake_pool.check_reserve_stake(reserve_stake_info)?;

    let withdraw_lamports = stake_pool
        .calc_lamports_withdraw_amount(pool_tokens)
        .ok_or(StakePoolError::CalculationFailure)?;
    if withdraw_lamports == 0 {
        return Err(StakePoolError::WithdrawalTooSmall.into());
    }

    Self::token_burn(
        token_program_info.clone(),
        burn_from_pool_info.clone(),
        pool_mint_info.clone(),
        user_transfer_authority_info.clone(),
        pool_tokens,
    )?;

    // The result of the stake withdrawal CPI is discarded instead of
    // propagated. If the stake program fails to move the lamports (e.g. the
    // reserve stake account is still locked or has insufficient balance),
    // the error is silently swallowed and execution continues as if the
    // withdrawal succeeded.
    let _ = Self::stake_withdraw(
        stake_pool_info.key,
        reserve_stake_info.clone(),
        withdraw_authority_info.clone(),
        AUTHORITY_WITHDRAW,
        stake_pool.stake_withdraw_bump_seed,
        destination_lamports_info.clone(),
        clock_info.clone(),
        stake_history_info.clone(),
        withdraw_lamports,
    );

    stake_pool.pool_token_supply = stake_pool
        .pool_token_supply
        .checked_sub(pool_tokens)
        .ok_or(StakePoolError::CalculationFailure)?;
    stake_pool.total_lamports = stake_pool
        .total_lamports
        .checked_sub(withdraw_lamports)
        .ok_or(StakePoolError::CalculationFailure)?;
    borsh::to_writer(&mut stake_pool_info.data.borrow_mut()[..], &stake_pool)?;

    Ok(())
}"""

# ============================================================
# Type Confusion -- governance/process_add_signatory.rs
# ============================================================

CODE_TC_SAFE = """pub fn process_add_signatory(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    signatory: Pubkey,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let signatory_record_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;

    let governance_data = get_governance_data(program_id, governance_info)?;

    let mut proposal_data =
        get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_edit_signatories()?;

    if !signatory_record_info.data_is_empty() {
        return Err(GovernanceError::SignatoryRecordAlreadyExists.into());
    }

    if proposal_data.signatories_count < governance_data.required_signatories_count {
        let required_signatory_info = next_account_info(account_info_iter)?;
        let required_signatory_data = get_required_signatory_data_for_governance(
            program_id,
            required_signatory_info,
            governance_info.key,
        )?;
        if required_signatory_data.signatory != signatory {
            return Err(GovernanceError::InvalidSignatoryAddress.into());
        }
    } else {
        let token_owner_record_info = next_account_info(account_info_iter)?;
        let governance_authority_info = next_account_info(account_info_iter)?;
        let token_owner_record_data = get_token_owner_record_data_for_proposal_owner(
            program_id,
            token_owner_record_info,
            &proposal_data.token_owner_record,
        )?;
        token_owner_record_data
            .assert_token_owner_or_delegate_is_signer(governance_authority_info)?;
    }

    let rent = Rent::get()?;
    let signatory_record_data = SignatoryRecordV2 {
        account_type: GovernanceAccountType::SignatoryRecordV2,
        proposal: *proposal_info.key,
        signatory,
        signed_off: false,
        reserved_v2: [0; 8],
    };

    create_and_serialize_account_signed::<SignatoryRecordV2>(
        payer_info,
        signatory_record_info,
        &signatory_record_data,
        &get_signatory_record_address_seeds(proposal_info.key, &signatory),
        program_id,
        system_info,
        &rent,
        0,
    )?;

    proposal_data.signatories_count = proposal_data.signatories_count.checked_add(1).unwrap();
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    Ok(())
}"""

CODE_TC_VULN = """pub fn process_add_signatory(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    signatory: Pubkey,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let signatory_record_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;

    // Governance and proposal state are deserialized directly from raw
    // bytes, skipping get_governance_data and
    // get_proposal_data_for_governance, so neither the account_type
    // discriminator nor the binding between this proposal and this
    // governance account is checked.
    let governance_data = Governance::try_from_slice(&governance_info.data.borrow())?;
    let mut proposal_data = ProposalV2::try_from_slice(&proposal_info.data.borrow())?;
    proposal_data.assert_can_edit_signatories()?;

    if !signatory_record_info.data_is_empty() {
        return Err(GovernanceError::SignatoryRecordAlreadyExists.into());
    }

    if proposal_data.signatories_count < governance_data.required_signatories_count {
        let required_signatory_info = next_account_info(account_info_iter)?;
        let required_signatory_data = get_required_signatory_data_for_governance(
            program_id,
            required_signatory_info,
            governance_info.key,
        )?;
        if required_signatory_data.signatory != signatory {
            return Err(GovernanceError::InvalidSignatoryAddress.into());
        }
    } else {
        let token_owner_record_info = next_account_info(account_info_iter)?;
        let governance_authority_info = next_account_info(account_info_iter)?;
        let token_owner_record_data = get_token_owner_record_data_for_proposal_owner(
            program_id,
            token_owner_record_info,
            &proposal_data.token_owner_record,
        )?;
        token_owner_record_data
            .assert_token_owner_or_delegate_is_signer(governance_authority_info)?;
    }

    let rent = Rent::get()?;
    let signatory_record_data = SignatoryRecordV2 {
        account_type: GovernanceAccountType::SignatoryRecordV2,
        proposal: *proposal_info.key,
        signatory,
        signed_off: false,
        reserved_v2: [0; 8],
    };

    create_and_serialize_account_signed::<SignatoryRecordV2>(
        payer_info,
        signatory_record_info,
        &signatory_record_data,
        &get_signatory_record_address_seeds(proposal_info.key, &signatory),
        program_id,
        system_info,
        &rent,
        0,
    )?;

    proposal_data.signatories_count = proposal_data.signatories_count.checked_add(1).unwrap();
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    Ok(())
}"""


def write_entry(idx, vulnerability, owasp_code, label, code):
    fname = f"solana_{idx:02d}.json"
    path = os.path.join(TEST_DIR, fname)
    if os.path.exists(path):
        raise FileExistsError(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vulnerability": vulnerability,
                "owasp_code": owasp_code,
                "label": label,
                "smart_contract": code,
            },
            f,
            indent=4,
        )
    return fname


def main():
    existing = [f for f in os.listdir(TEST_DIR) if f.startswith("solana_") and f.endswith(".json")]
    next_idx = len(existing) + 1

    pairs = [
        ("unchecked_calls", "V6", CODE_UC_VULN, CODE_UC_SAFE),
        ("type_confusion", "V4", CODE_TC_VULN, CODE_TC_SAFE),
    ]

    written = []
    for vuln_key, owasp, code_vuln, code_safe in pairs:
        written.append(write_entry(next_idx, vuln_key, owasp, "VULNERABLE", code_vuln))
        next_idx += 1
        written.append(write_entry(next_idx, "not_vulnerable", owasp, "SAFE", code_safe))
        next_idx += 1

    print(f"Wrote {len(written)} new test files ({len(pairs)} pairs):")
    for w in written:
        print(" ", w)
    print(f"Test set size: {len(existing)} -> {len(existing) + len(written)}")


if __name__ == "__main__":
    main()

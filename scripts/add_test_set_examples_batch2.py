"""
Second batch of test-set expansion, targeting the two categories that are
still thin after batch 1: Bump Seed Canonicalization and CPI Reentrancy
(both at 4/71 before this batch).

Source: solana-labs/solana-program-library/token-lending/program/src/
processor.rs, fn process_withdraw_obligation_collateral (never used
anywhere in training or the existing test set -- verified by grep). This one
real function legitimately demonstrates BOTH categories correctly at once:
  - Bump Seed: derives the lending market authority PDA with
    Pubkey::create_program_address(seeds, program_id) using
    `lending_market.bump_seed`, a bump stored on-chain at market-init time
    and re-verified on every call -- the "store and verify" canonical
    pattern (security_check bullet 2 in vulnerability_info.json), not the
    find_program_address-at-call-time pattern our other examples use, so it
    adds real pattern diversity rather than a near-duplicate.
  - CPI Reentrancy: all obligation state mutations (withdraw(), mark_stale(),
    Obligation::pack()) happen BEFORE the spl_token_transfer CPI call.

Two independent synthesized VULNERABLE variants are derived from it (one per
category), each changing only the one relevant thing and leaving the rest of
the real function untouched -- same methodology as dataset_construction
batch1 sample 12 and every other "modified from source" pair in this
project.
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

# ============================================================
# Shared real base: process_withdraw_obligation_collateral
# (token-lending). Trimmed of doc comments only.
# ============================================================

CODE_BASE_SAFE = """fn process_withdraw_obligation_collateral(
    program_id: &Pubkey,
    collateral_amount: u64,
    accounts: &[AccountInfo],
) -> ProgramResult {
    if collateral_amount == 0 {
        return Err(LendingError::InvalidAmount.into());
    }

    let account_info_iter = &mut accounts.iter();
    let source_collateral_info = next_account_info(account_info_iter)?;
    let destination_collateral_info = next_account_info(account_info_iter)?;
    let withdraw_reserve_info = next_account_info(account_info_iter)?;
    let obligation_info = next_account_info(account_info_iter)?;
    let lending_market_info = next_account_info(account_info_iter)?;
    let lending_market_authority_info = next_account_info(account_info_iter)?;
    let obligation_owner_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(next_account_info(account_info_iter)?)?;
    let token_program_id = next_account_info(account_info_iter)?;

    let lending_market = LendingMarket::unpack(&lending_market_info.data.borrow())?;
    let withdraw_reserve = Reserve::unpack(&withdraw_reserve_info.data.borrow())?;
    let mut obligation = Obligation::unpack(&obligation_info.data.borrow())?;

    if !obligation_owner_info.is_signer {
        return Err(LendingError::InvalidSigner.into());
    }
    if &obligation.owner != obligation_owner_info.key {
        return Err(LendingError::InvalidObligationOwner.into());
    }

    let (collateral, collateral_index) =
        obligation.find_collateral_in_deposits(*withdraw_reserve_info.key)?;
    if collateral.deposited_amount == 0 {
        return Err(LendingError::ObligationCollateralEmpty.into());
    }

    // The authority PDA's bump was computed once via find_program_address at
    // market-init time and stored on-chain as lending_market.bump_seed. Every
    // subsequent call re-derives the address from that STORED canonical bump
    // and rejects the instruction if it doesn't match the provided account,
    // rather than trusting any bump passed in fresh instruction data.
    let authority_signer_seeds = &[
        lending_market_info.key.as_ref(),
        &[lending_market.bump_seed],
    ];
    let lending_market_authority_pubkey =
        Pubkey::create_program_address(authority_signer_seeds, program_id)?;
    if &lending_market_authority_pubkey != lending_market_authority_info.key {
        return Err(LendingError::InvalidMarketAuthority.into());
    }

    let withdraw_amount = collateral.deposited_amount.min(collateral_amount);

    // State is mutated and persisted BEFORE the CPI below, so if the token
    // program were ever to call back into this program, it would observe
    // the already-updated (post-withdrawal) obligation state.
    obligation.withdraw(withdraw_amount, collateral_index)?;
    obligation.last_update.mark_stale();
    Obligation::pack(obligation, &mut obligation_info.data.borrow_mut())?;

    spl_token_transfer(TokenTransferParams {
        source: source_collateral_info.clone(),
        destination: destination_collateral_info.clone(),
        amount: withdraw_amount,
        authority: lending_market_authority_info.clone(),
        authority_signer_seeds,
        token_program: token_program_id.clone(),
    })?;

    Ok(())
}"""

CODE_BUMP_VULN = """fn process_withdraw_obligation_collateral(
    program_id: &Pubkey,
    collateral_amount: u64,
    bump_seed: u8,
    accounts: &[AccountInfo],
) -> ProgramResult {
    if collateral_amount == 0 {
        return Err(LendingError::InvalidAmount.into());
    }

    let account_info_iter = &mut accounts.iter();
    let source_collateral_info = next_account_info(account_info_iter)?;
    let destination_collateral_info = next_account_info(account_info_iter)?;
    let withdraw_reserve_info = next_account_info(account_info_iter)?;
    let obligation_info = next_account_info(account_info_iter)?;
    let lending_market_info = next_account_info(account_info_iter)?;
    let lending_market_authority_info = next_account_info(account_info_iter)?;
    let obligation_owner_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(next_account_info(account_info_iter)?)?;
    let token_program_id = next_account_info(account_info_iter)?;

    let withdraw_reserve = Reserve::unpack(&withdraw_reserve_info.data.borrow())?;
    let mut obligation = Obligation::unpack(&obligation_info.data.borrow())?;

    if !obligation_owner_info.is_signer {
        return Err(LendingError::InvalidSigner.into());
    }
    if &obligation.owner != obligation_owner_info.key {
        return Err(LendingError::InvalidObligationOwner.into());
    }

    let (collateral, collateral_index) =
        obligation.find_collateral_in_deposits(*withdraw_reserve_info.key)?;
    if collateral.deposited_amount == 0 {
        return Err(LendingError::ObligationCollateralEmpty.into());
    }

    // The bump seed is taken directly from instruction data instead of the
    // canonical value stored on-chain at market-init time. create_program_address
    // only checks that this bump/seed combination produces the given address --
    // it does not verify the bump is the unique canonical one for these seeds.
    let authority_signer_seeds = &[
        lending_market_info.key.as_ref(),
        &[bump_seed],
    ];
    let lending_market_authority_pubkey =
        Pubkey::create_program_address(authority_signer_seeds, program_id)?;
    if &lending_market_authority_pubkey != lending_market_authority_info.key {
        return Err(LendingError::InvalidMarketAuthority.into());
    }

    let withdraw_amount = collateral.deposited_amount.min(collateral_amount);

    obligation.withdraw(withdraw_amount, collateral_index)?;
    obligation.last_update.mark_stale();
    Obligation::pack(obligation, &mut obligation_info.data.borrow_mut())?;

    spl_token_transfer(TokenTransferParams {
        source: source_collateral_info.clone(),
        destination: destination_collateral_info.clone(),
        amount: withdraw_amount,
        authority: lending_market_authority_info.clone(),
        authority_signer_seeds,
        token_program: token_program_id.clone(),
    })?;

    Ok(())
}"""

CODE_CPI_VULN = """fn process_withdraw_obligation_collateral(
    program_id: &Pubkey,
    collateral_amount: u64,
    accounts: &[AccountInfo],
) -> ProgramResult {
    if collateral_amount == 0 {
        return Err(LendingError::InvalidAmount.into());
    }

    let account_info_iter = &mut accounts.iter();
    let source_collateral_info = next_account_info(account_info_iter)?;
    let destination_collateral_info = next_account_info(account_info_iter)?;
    let withdraw_reserve_info = next_account_info(account_info_iter)?;
    let obligation_info = next_account_info(account_info_iter)?;
    let lending_market_info = next_account_info(account_info_iter)?;
    let lending_market_authority_info = next_account_info(account_info_iter)?;
    let obligation_owner_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(next_account_info(account_info_iter)?)?;
    let token_program_id = next_account_info(account_info_iter)?;

    let lending_market = LendingMarket::unpack(&lending_market_info.data.borrow())?;
    let withdraw_reserve = Reserve::unpack(&withdraw_reserve_info.data.borrow())?;
    let mut obligation = Obligation::unpack(&obligation_info.data.borrow())?;

    if !obligation_owner_info.is_signer {
        return Err(LendingError::InvalidSigner.into());
    }
    if &obligation.owner != obligation_owner_info.key {
        return Err(LendingError::InvalidObligationOwner.into());
    }

    let (collateral, collateral_index) =
        obligation.find_collateral_in_deposits(*withdraw_reserve_info.key)?;
    if collateral.deposited_amount == 0 {
        return Err(LendingError::ObligationCollateralEmpty.into());
    }

    let authority_signer_seeds = &[
        lending_market_info.key.as_ref(),
        &[lending_market.bump_seed],
    ];
    let lending_market_authority_pubkey =
        Pubkey::create_program_address(authority_signer_seeds, program_id)?;
    if &lending_market_authority_pubkey != lending_market_authority_info.key {
        return Err(LendingError::InvalidMarketAuthority.into());
    }

    let withdraw_amount = collateral.deposited_amount.min(collateral_amount);

    // The token transfer CPI is issued FIRST, before the obligation's
    // collateral bookkeeping is updated. If the token program (or an
    // extension hook it invokes) calls back into this program before
    // returning, it would observe the obligation still showing the
    // collateral as not yet withdrawn.
    spl_token_transfer(TokenTransferParams {
        source: source_collateral_info.clone(),
        destination: destination_collateral_info.clone(),
        amount: withdraw_amount,
        authority: lending_market_authority_info.clone(),
        authority_signer_seeds,
        token_program: token_program_id.clone(),
    })?;

    obligation.withdraw(withdraw_amount, collateral_index)?;
    obligation.last_update.mark_stale();
    Obligation::pack(obligation, &mut obligation_info.data.borrow_mut())?;

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
        ("bump_seed", "V9", CODE_BUMP_VULN, CODE_BASE_SAFE),
        ("cpi", "V5", CODE_CPI_VULN, CODE_BASE_SAFE),
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

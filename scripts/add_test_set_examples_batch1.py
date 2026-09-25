"""
First batch of test-set expansion: the 59-contract Solana test set gives only
4-5 examples per vulnerability category, so per-category recall is quantized
to 25pp steps and a single flipped case swings the metric wildly (confirmed
via bootstrap: F1 95% CI on the current best config is [0.48, 0.81] --
too wide to trust point estimates). This adds new REAL, manually-verified
contracts as held-out test examples, sourced from repos/functions never used
in training or in the existing 59-file test set, to avoid leakage.

Sources:
  - coral-xyz/sealevel-attacks (0-signer-authorization, 1-account-data-matching,
    2-owner-checks, 4-initialization, 10-sysvar-address-checking): insecure/
    recommended pairs used directly. Only categories with an unambiguous
    match to our 7-category taxonomy were used; arbitrary-cpi and
    closing-accounts were deliberately skipped -- they describe attack
    patterns ("untrusted CPI target", "account revival") that don't map
    cleanly onto CPI Reentrancy/Unchecked External Calls/DoS without
    stretching the label, and a mislabeled test example is worse than one
    fewer example.
  - solana-labs/solana-program-library/governance/program/src/processor/
    process_cancel_proposal.rs (Type Confusion): not used anywhere in
    training; SAFE is the real function (typed getters), VULNERABLE is a
    synthesized variant (raw try_from_slice, same technique as the other
    governance-derived training pairs added earlier this session).

Mapping used (per data/knowledge_base/vulnerability_info.json definitions):
  - signer-authorization, account-data-matching, sysvar-address-checking
    -> Missing Key Check (V1): all three are "caller/account identity not
       verified against expected signer/key" per the V1 security_check.
  - owner-checks, process_cancel_proposal -> Type Confusion (V4): both are
    "account deserialized without verifying owner program / discriminator"
    per the V4 security_check.
  - initialization (reinitialization) -> Denial of Service (V10): DoS's own
    definition explicitly names "re-initialization of already-initialized
    accounts" as the primary example.

This is a first, conservative batch (6 pairs = 12 files): 59 -> 71 contracts.
Brings missing_key_check 4->7, type_confusion 4->6, dos 4->5. Further batches
should keep growing the thinner categories (type_confusion, unchecked_calls,
bump_seed, cpi) toward ~10-15 each.
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

# ============================================================
# 1. Missing Key Check -- sealevel-attacks/0-signer-authorization
# ============================================================

CODE_MKC1_VULN = """use anchor_lang::prelude::*;

#[program]
pub mod signer_authorization_insecure {
    use super::*;

    pub fn log_message(ctx: Context<LogMessage>) -> ProgramResult {
        msg!("GM {}", ctx.accounts.authority.key().to_string());
        Ok(())
    }
}

#[derive(Accounts)]
pub struct LogMessage<'info> {
    authority: AccountInfo<'info>,
}"""

CODE_MKC1_SAFE = """use anchor_lang::prelude::*;

#[program]
pub mod signer_authorization_recommended {
    use super::*;

    pub fn log_message(ctx: Context<LogMessage>) -> ProgramResult {
        msg!("GM {}", ctx.accounts.authority.key().to_string());
        Ok(())
    }
}

#[derive(Accounts)]
pub struct LogMessage<'info> {
    authority: Signer<'info>,
}"""

# ============================================================
# 2. Missing Key Check -- sealevel-attacks/1-account-data-matching
# ============================================================

CODE_MKC2_VULN = """use anchor_lang::prelude::*;
use anchor_lang::solana_program::program_pack::Pack;
use spl_token::state::Account as SplTokenAccount;

#[program]
pub mod account_data_matching_insecure {
    use super::*;

    pub fn log_message(ctx: Context<LogMessage>) -> ProgramResult {
        let token = SplTokenAccount::unpack(&ctx.accounts.token.data.borrow())?;
        msg!("Your account balance is: {}", token.amount);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct LogMessage<'info> {
    token: AccountInfo<'info>,
    authority: Signer<'info>,
}"""

CODE_MKC2_SAFE = """use anchor_lang::prelude::*;
use anchor_spl::token::TokenAccount;

#[program]
pub mod account_data_matching_recommended {
    use super::*;

    pub fn log_message(ctx: Context<LogMessage>) -> ProgramResult {
        msg!("Your account balance is: {}", ctx.accounts.token.amount);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct LogMessage<'info> {
    #[account(constraint = authority.key == &token.owner)]
    token: Account<'info, TokenAccount>,
    authority: Signer<'info>,
}"""

# ============================================================
# 3. Missing Key Check -- sealevel-attacks/10-sysvar-address-checking
# ============================================================

CODE_MKC3_VULN = """use anchor_lang::prelude::*;

#[program]
pub mod sysvar_address_checking_insecure {
    use super::*;

    pub fn check_sysvar_address(ctx: Context<CheckSysvarAddress>) -> Result<()> {
        msg!("Rent Key -> {}", ctx.accounts.rent.key().to_string());
        Ok(())
    }
}

#[derive(Accounts)]
pub struct CheckSysvarAddress<'info> {
    rent: AccountInfo<'info>,
}"""

CODE_MKC3_SAFE = """use anchor_lang::prelude::*;

#[program]
pub mod sysvar_address_checking_recommended {
    use super::*;

    pub fn check_sysvar_address(ctx: Context<CheckSysvarAddress>) -> Result<()> {
        msg!("Rent Key -> {}", ctx.accounts.rent.key().to_string());
        Ok(())
    }
}

#[derive(Accounts)]
pub struct CheckSysvarAddress<'info> {
    rent: Sysvar<'info, Rent>,
}"""

# ============================================================
# 4. Type Confusion -- sealevel-attacks/2-owner-checks
# ============================================================

CODE_TC1_VULN = """use anchor_lang::prelude::*;
use anchor_lang::solana_program::program_error::ProgramError;
use anchor_lang::solana_program::program_pack::Pack;
use spl_token::state::Account as SplTokenAccount;

#[program]
pub mod owner_checks_insecure {
    use super::*;

    pub fn log_message(ctx: Context<LogMessage>) -> ProgramResult {
        let token = SplTokenAccount::unpack(&ctx.accounts.token.data.borrow())?;
        if ctx.accounts.authority.key != &token.owner {
            return Err(ProgramError::InvalidAccountData);
        }
        msg!("Your account balance is: {}", token.amount);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct LogMessage<'info> {
    token: AccountInfo<'info>,
    authority: Signer<'info>,
}"""

CODE_TC1_SAFE = """use anchor_lang::prelude::*;
use anchor_spl::token::TokenAccount;

#[program]
pub mod owner_checks_recommended {
    use super::*;

    pub fn log_message(ctx: Context<LogMessage>) -> ProgramResult {
        msg!("Your account balance is: {}", ctx.accounts.token.amount);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct LogMessage<'info> {
    #[account(constraint = authority.key == &token.owner)]
    token: Account<'info, TokenAccount>,
    authority: Signer<'info>,
}"""

# ============================================================
# 5. Type Confusion -- governance/process_cancel_proposal.rs
# (SAFE is the real function; VULNERABLE is a synthesized variant)
# ============================================================

CODE_TC2_SAFE = """pub fn process_cancel_proposal(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let proposal_owner_record_info = next_account_info(account_info_iter)?;
    let governance_authority_info = next_account_info(account_info_iter)?;

    let clock = Clock::get()?;

    assert_is_valid_realm(program_id, realm_info)?;

    let mut governance_data =
        get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;

    let mut proposal_data =
        get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_cancel(&governance_data.config, clock.unix_timestamp)?;

    let mut proposal_owner_record_data = get_token_owner_record_data_for_proposal_owner(
        program_id,
        proposal_owner_record_info,
        &proposal_data.token_owner_record,
    )?;

    proposal_owner_record_data
        .assert_token_owner_or_delegate_is_signer(governance_authority_info)?;

    proposal_owner_record_data.decrease_outstanding_proposal_count();
    proposal_owner_record_data.serialize(&mut proposal_owner_record_info.data.borrow_mut()[..])?;

    proposal_data.state = ProposalState::Cancelled;
    proposal_data.closed_at = Some(clock.unix_timestamp);

    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    governance_data.active_proposal_count = governance_data.active_proposal_count.saturating_sub(1);
    governance_data.serialize(&mut governance_info.data.borrow_mut()[..])?;

    Ok(())
}"""

CODE_TC2_VULN = """pub fn process_cancel_proposal(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let proposal_owner_record_info = next_account_info(account_info_iter)?;
    let governance_authority_info = next_account_info(account_info_iter)?;

    let clock = Clock::get()?;

    assert_is_valid_realm(program_id, realm_info)?;

    let mut governance_data =
        get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;

    // Proposal and owner-record state are deserialized directly from raw
    // bytes, skipping get_proposal_data_for_governance and
    // get_token_owner_record_data_for_proposal_owner, so neither the
    // account_type discriminator nor the binding to this governance/proposal
    // pair is checked.
    let mut proposal_data = ProposalV2::try_from_slice(&proposal_info.data.borrow())?;
    proposal_data.assert_can_cancel(&governance_data.config, clock.unix_timestamp)?;

    let mut proposal_owner_record_data =
        TokenOwnerRecordV2::try_from_slice(&proposal_owner_record_info.data.borrow())?;

    proposal_owner_record_data
        .assert_token_owner_or_delegate_is_signer(governance_authority_info)?;

    proposal_owner_record_data.decrease_outstanding_proposal_count();
    proposal_owner_record_data.serialize(&mut proposal_owner_record_info.data.borrow_mut()[..])?;

    proposal_data.state = ProposalState::Cancelled;
    proposal_data.closed_at = Some(clock.unix_timestamp);

    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    governance_data.active_proposal_count = governance_data.active_proposal_count.saturating_sub(1);
    governance_data.serialize(&mut governance_info.data.borrow_mut()[..])?;

    Ok(())
}"""

# ============================================================
# 6. Denial of Service -- sealevel-attacks/4-initialization
# ============================================================

CODE_DOS1_VULN = """use anchor_lang::prelude::*;
use borsh::{BorshDeserialize, BorshSerialize};
use std::ops::DerefMut;

#[program]
pub mod initialization_insecure {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> ProgramResult {
        let mut user = User::try_from_slice(&ctx.accounts.user.data.borrow()).unwrap();

        user.authority = ctx.accounts.authority.key();

        let mut storage = ctx.accounts.user.try_borrow_mut_data()?;
        user.serialize(storage.deref_mut()).unwrap();
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    user: AccountInfo<'info>,
    authority: Signer<'info>,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct User {
    authority: Pubkey,
}"""

CODE_DOS1_SAFE = """use anchor_lang::prelude::*;

#[program]
pub mod reinitialization_recommended {
    use super::*;

    pub fn init(_ctx: Context<Init>) -> ProgramResult {
        msg!("GM");
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Init<'info> {
    #[account(init, payer = authority, space = 8+32)]
    user: Account<'info, User>,
    #[account(mut)]
    authority: Signer<'info>,
    system_program: Program<'info, System>,
}

#[account]
pub struct User {
    authority: Pubkey,
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
    next_idx = len(existing) + 1  # 59 existing -> start at 60

    pairs = [
        ("missing_key_check", "V1", CODE_MKC1_VULN, CODE_MKC1_SAFE),
        ("missing_key_check", "V1", CODE_MKC2_VULN, CODE_MKC2_SAFE),
        ("missing_key_check", "V1", CODE_MKC3_VULN, CODE_MKC3_SAFE),
        ("type_confusion", "V4", CODE_TC1_VULN, CODE_TC1_SAFE),
        ("type_confusion", "V4", CODE_TC2_VULN, CODE_TC2_SAFE),
        ("dos", "V10", CODE_DOS1_VULN, CODE_DOS1_SAFE),
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

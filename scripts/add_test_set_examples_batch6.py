"""
Sixth batch: targets the two categories still stuck at 5 (Unchecked External
Calls, CPI Reentrancy), from two more official Anchor test programs never
used anywhere in training or the existing test set (verified by grep).

  - Unchecked External Calls: coral-xyz/anchor tests/auction-house,
    fn withdraw_from_fee. Real code propagates the single invoke_signed
    (system_instruction::transfer) with `?`. VULNERABLE discards it.
  - CPI Reentrancy: coral-xyz/anchor tests/lockup/programs/registry,
    fn end_unstake. The REAL shipped code already performs the CPI
    (token::transfer) before marking pending_withdrawal.burned = true --
    it is the vulnerable side (same "real code is vulnerable, SAFE is the
    synthesized reordered fix" situation as lockup::withdraw, used in
    training -- this is a different function, no duplication).
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

# ============================================================
# 1. Unchecked External Calls -- anchor/tests/auction-house, withdraw_from_fee
# ============================================================

CODE_UC_SAFE = """pub fn withdraw_from_fee(ctx: Context<WithdrawFromFee>, amount: u64) -> Result<()> {
    let auction_house_fee_account = &ctx.accounts.auction_house_fee_account;
    let fee_withdrawal_destination = &ctx.accounts.fee_withdrawal_destination;
    let auction_house = &ctx.accounts.auction_house;
    let system_program = &ctx.accounts.system_program;

    let auction_house_key = auction_house.key();
    let seeds = [
        PREFIX.as_bytes(),
        auction_house_key.as_ref(),
        FEE_PAYER.as_bytes(),
        &[auction_house.fee_payer_bump],
    ];

    invoke_signed(
        &system_instruction::transfer(
            &auction_house_fee_account.key(),
            &fee_withdrawal_destination.key(),
            amount,
        ),
        &[
            auction_house_fee_account.to_account_info(),
            fee_withdrawal_destination.to_account_info(),
            system_program.to_account_info(),
        ],
        &[&seeds],
    )?;

    Ok(())
}"""

CODE_UC_VULN = """pub fn withdraw_from_fee(ctx: Context<WithdrawFromFee>, amount: u64) -> Result<()> {
    let auction_house_fee_account = &ctx.accounts.auction_house_fee_account;
    let fee_withdrawal_destination = &ctx.accounts.fee_withdrawal_destination;
    let auction_house = &ctx.accounts.auction_house;
    let system_program = &ctx.accounts.system_program;

    let auction_house_key = auction_house.key();
    let seeds = [
        PREFIX.as_bytes(),
        auction_house_key.as_ref(),
        FEE_PAYER.as_bytes(),
        &[auction_house.fee_payer_bump],
    ];

    // The transfer's result is discarded instead of propagated.
    let _ = invoke_signed(
        &system_instruction::transfer(
            &auction_house_fee_account.key(),
            &fee_withdrawal_destination.key(),
            amount,
        ),
        &[
            auction_house_fee_account.to_account_info(),
            fee_withdrawal_destination.to_account_info(),
            system_program.to_account_info(),
        ],
        &[&seeds],
    );

    Ok(())
}"""

# ============================================================
# 2. CPI Reentrancy -- anchor/tests/lockup/programs/registry, end_unstake
# ============================================================

CODE_CPI_VULN = """pub fn end_unstake(ctx: Context<EndUnstake>) -> Result<()> {
    if ctx.accounts.pending_withdrawal.end_ts > ctx.accounts.clock.unix_timestamp {
        return err!(ErrorCode::UnstakeTimelock);
    }

    let balances = {
        if ctx.accounts.pending_withdrawal.locked {
            &ctx.accounts.member.balances_locked
        } else {
            &ctx.accounts.member.balances
        }
    };
    if &balances.vault != ctx.accounts.vault.key {
        return err!(ErrorCode::InvalidVault);
    }
    if &balances.vault_pw != ctx.accounts.vault_pw.key {
        return err!(ErrorCode::InvalidVault);
    }

    // Transfer tokens between vaults.
    {
        let seeds = &[
            ctx.accounts.registrar.to_account_info().key.as_ref(),
            ctx.accounts.member.to_account_info().key.as_ref(),
            &[ctx.accounts.member.nonce],
        ];
        let signer = &[&seeds[..]];
        let cpi_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.clone(),
            Transfer {
                from: ctx.accounts.vault_pw.to_account_info(),
                to: ctx.accounts.vault.to_account_info(),
                authority: ctx.accounts.member_signer.clone(),
            },
            signer,
        );
        token::transfer(cpi_ctx, ctx.accounts.pending_withdrawal.amount)?;
    }

    // Burn the pending withdrawal receipt AFTER the CPI.
    let pending_withdrawal = &mut ctx.accounts.pending_withdrawal;
    pending_withdrawal.burned = true;

    Ok(())
}"""

CODE_CPI_SAFE = """pub fn end_unstake(ctx: Context<EndUnstake>) -> Result<()> {
    if ctx.accounts.pending_withdrawal.end_ts > ctx.accounts.clock.unix_timestamp {
        return err!(ErrorCode::UnstakeTimelock);
    }

    let balances = {
        if ctx.accounts.pending_withdrawal.locked {
            &ctx.accounts.member.balances_locked
        } else {
            &ctx.accounts.member.balances
        }
    };
    if &balances.vault != ctx.accounts.vault.key {
        return err!(ErrorCode::InvalidVault);
    }
    if &balances.vault_pw != ctx.accounts.vault_pw.key {
        return err!(ErrorCode::InvalidVault);
    }

    // Burn the pending withdrawal receipt BEFORE the CPI, so a callback
    // during the transfer would see it already consumed.
    let pending_withdrawal = &mut ctx.accounts.pending_withdrawal;
    pending_withdrawal.burned = true;
    let withdrawal_amount = pending_withdrawal.amount;

    let seeds = &[
        ctx.accounts.registrar.to_account_info().key.as_ref(),
        ctx.accounts.member.to_account_info().key.as_ref(),
        &[ctx.accounts.member.nonce],
    ];
    let signer = &[&seeds[..]];
    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.token_program.clone(),
        Transfer {
            from: ctx.accounts.vault_pw.to_account_info(),
            to: ctx.accounts.vault.to_account_info(),
            authority: ctx.accounts.member_signer.clone(),
        },
        signer,
    );
    token::transfer(cpi_ctx, withdrawal_amount)?;

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

    pairs = [
        ("unchecked_calls", "V6", CODE_UC_VULN, CODE_UC_SAFE),
        ("cpi", "V5", CODE_CPI_VULN, CODE_CPI_SAFE),
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

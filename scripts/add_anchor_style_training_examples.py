"""
Extend training with Anchor-style (#[program]/Context/#[account(...)]) real
examples for the 5 categories that only had native/raw-processor-style
examples until now: Missing Key Check, Denial of Service, CPI Reentrancy,
Unchecked External Calls, Integer Overflow. (Type Confusion and Bump Seed
already got Anchor-style pairs earlier this session, from sealevel-attacks.)

All 5 come from coral-xyz/anchor's own official test/example programs
(tests/multisig, tests/lockup, tests/escrow, tests/ido-pool) -- canonical,
maintained-by-the-framework-authors code, never used anywhere in training or
the test set (verified by grep). This closes a real coverage gap: every
example the model had seen before this addition used native
next_account_info/AccountInfo style; real-world Solana programs increasingly
use Anchor, and the newly-expanded test set already exposed 3/3 missed
detections on Anchor-style Missing Key Check variants specifically because
of this gap.

Per-pair sourcing (methodology: "direct extraction" for the real side,
"synthesized vulnerable/safe variant" for the modified side, same technique
used throughout this project):

  1. Missing Key Check -- tests/multisig, fn `approve`. Real code requires
     the caller to be a Signer AND to be listed in multisig.owners
     (position().ok_or(InvalidOwner)). VULNERABLE variant drops the
     owner-list check, so any signer can approve.

  2. Denial of Service -- tests/lockup, fn `whitelist_add`. Real code checks
     `whitelist.len() == WHITELIST_SIZE` before pushing. VULNERABLE variant
     drops the bound, allowing unbounded growth.

  3. CPI Reentrancy -- tests/lockup, fn `withdraw`. Here the REAL shipped
     code already performs the CPI (token::transfer) before the local state
     update (vesting.outstanding -= amount) -- it is the vulnerable side.
     The SAFE variant is the synthesized fix: reorder state update before
     the CPI.

  4. Unchecked External Calls -- tests/escrow, fn `exchange`. Real code
     propagates the return value of all three CPI calls with `?`.
     VULNERABLE variant discards the middle transfer_checked result with
     `let _ =`.

  5. Integer Overflow -- tests/ido-pool, fn
     `exchange_redeemable_for_watermelon`. Real code computes the payout
     with checked_mul(...).unwrap().checked_div(...).unwrap() on u128.
     VULNERABLE variant replaces it with raw `*`/`/`.
"""

import json
import os

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
TRAIN_PATH = os.path.join(REPO_DIR, "data", "training", "dataset_think_format.jsonl")

DEVELOPER_PROMPT = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
Your task is to analyze Rust code precisely and systematically to identify security vulnerabilities.

### Required behavior:
- Always perform a structured, explicit reasoning phase first and put it inside the <think> block.
  - In <think>...</think> you must:
    - Summarize the contract's purpose and high-level architecture.
    - Inspect the logic block-by-block (or line-by-line for short snippets).
    - Note any suspicious patterns, missing authorization checks, unsafe operations, or other issues with evidence (point to the code lines/constructs).
    - Use concise, technical language and show the chain of reasoning (why you suspect an issue).

- After </think>, provide the final judgment in the <final>...</final> block using this exact structure:
  - A short summary sentence (one or two lines).
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it can be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

### Constraints:
- Do NOT include any extra commentary, greetings, or meta-text.
- If no vulnerability is found, explicitly write "### Vulnerability: Not Vulnerable" and "### Risk: No significant risk identified"."""

USER_TEMPLATE = """Please perform a detailed security analysis of the following Solana smart contract.
Carefully examine its logic, identify any potential vulnerabilities:

Contract Code:

{code}"""


def sample(id_, code, thinking, content):
    return {
        "id": id_,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(code=code)},
            {"role": "assistant", "thinking": thinking, "content": content},
        ],
    }


# ============================================================
# 1. Missing Key Check -- anchor/tests/multisig, fn approve
# ============================================================

CODE_MKC_SAFE = """#[derive(Accounts)]
pub struct Approve<'info> {
    multisig: Account<'info, Multisig>,
    #[account(mut, has_one = multisig)]
    transaction: Account<'info, Transaction>,
    // One of the multisig owners. Checked in the handler.
    owner: Signer<'info>,
}

pub fn approve(ctx: Context<Approve>) -> Result<()> {
    let owner_index = ctx
        .accounts
        .multisig
        .owners
        .iter()
        .position(|a| a == ctx.accounts.owner.key)
        .ok_or(error!(ErrorCode::InvalidOwner))?;

    ctx.accounts.transaction.signers[owner_index] = true;

    Ok(())
}"""

CODE_MKC_VULN = """#[derive(Accounts)]
pub struct Approve<'info> {
    multisig: Account<'info, Multisig>,
    #[account(mut, has_one = multisig)]
    transaction: Account<'info, Transaction>,
    owner: Signer<'info>,
}

pub fn approve(ctx: Context<Approve>) -> Result<()> {
    // The owner is required to sign, but nothing checks that this signer is
    // actually listed in multisig.owners -- any keypair can sign and mark an
    // arbitrary signer slot as approved.
    let signer_index = ctx.accounts.transaction.signers.len() - 1;
    ctx.accounts.transaction.signers[signer_index] = true;

    Ok(())
}"""

THINKING_MKC_VULN = """Analyzing the Solana smart contract function `approve`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for authorization and access control.

2. **Expected security checks for V1 (Missing Key Check)**:
   - Check 1: caller is an authorized signer (is_signer == true)
   - Check 2: the provided authority account matches an expected key stored on-chain
   - Check 3: account ownership validated

3. **Code inspection**: Examining the implementation of `approve`:
   - `owner` is typed as `Signer<'info>`, so Anchor guarantees `is_signer == true`. However, nothing checks that `owner.key` is actually one of the addresses listed in `multisig.owners`. The handler marks a signer slot as approved without ever looking up which owner (if any) this signer corresponds to.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing verification that the signer is an authorized multisig owner. Any keypair, not just a listed owner, can sign this instruction and have it counted toward the approval threshold, allowing an outsider to help push a malicious transaction past `execute_transaction`'s signature count.

5. **Severity assessment**: CRITICAL - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_MKC_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V1 - Missing Key Check (Access Control)
### Explanation: The approve handler only checks that owner is a signer, never that owner.key is listed in multisig.owners. Any signer can approve a transaction, bypassing the multisig's authorization model.
### Risk: CRITICAL - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""

THINKING_MKC_SAFE = """Analyzing the Solana smart contract function `approve`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for authorization and access control.

2. **Expected security checks for V1 (Missing Key Check)**:
   - Check 1: caller is an authorized signer (is_signer == true)
   - Check 2: the provided authority account matches an expected key stored on-chain
   - Check 3: account ownership validated

3. **Code inspection**: Examining the implementation of `approve`:
   - `owner` is typed as `Signer<'info>` (enforces is_signer), and the handler additionally looks up `owner.key` in `multisig.owners` via `.position(...).ok_or(InvalidOwner)`, rejecting the instruction if the signer is not a listed owner. Only then is the corresponding signer slot marked approved.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V1 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles authorization and access control appropriately."""

CONTENT_MKC_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The signer's key is looked up in multisig.owners before any signer slot is marked approved, so only listed owners can contribute to the approval threshold.
### Risk: No significant risk identified."""


# ============================================================
# 2. Denial of Service -- anchor/tests/lockup, fn whitelist_add
# ============================================================

CODE_DOS_SAFE = """pub fn whitelist_add(&mut self, ctx: Context<Auth>, entry: WhitelistEntry) -> Result<()> {
    if self.whitelist.len() == Self::WHITELIST_SIZE {
        return err!(ErrorCode::WhitelistFull);
    }
    if self.whitelist.contains(&entry) {
        return err!(ErrorCode::WhitelistEntryAlreadyExists);
    }
    self.whitelist.push(entry);
    Ok(())
}"""

CODE_DOS_VULN = """pub fn whitelist_add(&mut self, ctx: Context<Auth>, entry: WhitelistEntry) -> Result<()> {
    if self.whitelist.contains(&entry) {
        return err!(ErrorCode::WhitelistEntryAlreadyExists);
    }
    self.whitelist.push(entry);
    Ok(())
}"""

THINKING_DOS_VULN = """Analyzing the Solana smart contract function `whitelist_add`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for unbounded growth and resource-exhaustion guards.

2. **Expected security checks for V10 (Denial of Service)**:
   - Check 1: initialization functions reject re-initialization
   - Check 2: accounts verify rent exemption status
   - Check 3: oracle/price data includes staleness checks
   - Check 4: loops/collections are bounded with explicit limits

3. **Code inspection**: Examining the implementation of `whitelist_add`:
   - The duplicate check (`whitelist.contains`) is present, but the size-bound check (`whitelist.len() == WHITELIST_SIZE`) that exists in the reference implementation has been removed. Entries can be pushed onto `self.whitelist` indefinitely.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from an unbounded collection: since the account's on-chain storage is allocated to a fixed size, repeatedly calling this instruction will eventually overflow the account's data buffer (or, if the Vec instead silently reallocates conceptually, blow the compute/serialization budget on every subsequent read of the whitelist), rendering the whitelist -- and any instruction that iterates over it -- unusable.

5. **Severity assessment**: MEDIUM - This vulnerability can be directly exploited by a malicious actor with repeated calls."""

CONTENT_DOS_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V10 - Denial of Service (DoS)
### Explanation: whitelist_add no longer checks the whitelist against its fixed maximum size (WHITELIST_SIZE) before pushing. Repeated calls grow the collection past its allocated on-chain space, eventually breaking every instruction that reads the whitelist.
### Risk: MEDIUM - This vulnerability can be exploited to disrupt normal contract operation."""

THINKING_DOS_SAFE = """Analyzing the Solana smart contract function `whitelist_add`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for unbounded growth and resource-exhaustion guards.

2. **Expected security checks for V10 (Denial of Service)**:
   - Check 1: initialization functions reject re-initialization
   - Check 2: accounts verify rent exemption status
   - Check 3: oracle/price data includes staleness checks
   - Check 4: loops/collections are bounded with explicit limits

3. **Code inspection**: Examining the implementation of `whitelist_add`:
   - Before pushing, the code checks `self.whitelist.len() == Self::WHITELIST_SIZE` and rejects the call once the fixed maximum is reached. A separate check rejects duplicate entries.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V10 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract bounds the whitelist's growth appropriately."""

CONTENT_DOS_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: whitelist_add rejects new entries once the collection reaches its fixed WHITELIST_SIZE, preventing unbounded growth of the on-chain whitelist.
### Risk: No significant risk identified."""


# ============================================================
# 3. CPI Reentrancy -- anchor/tests/lockup, fn withdraw
# (real shipped code is the VULNERABLE side; SAFE is the reordered fix)
# ============================================================

CODE_CPI_VULN = """pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
    if amount
        > calculator::available_for_withdrawal(
            &ctx.accounts.vesting,
            ctx.accounts.clock.unix_timestamp,
        )
    {
        return err!(ErrorCode::InsufficientWithdrawalBalance);
    }

    // Transfer funds out.
    let seeds = &[
        ctx.accounts.vesting.to_account_info().key.as_ref(),
        &[ctx.accounts.vesting.nonce],
    ];
    let signer = &[&seeds[..]];
    let cpi_ctx = CpiContext::from(&*ctx.accounts).with_signer(signer);
    token::transfer(cpi_ctx, amount)?;

    // Bookkeeping.
    let vesting = &mut ctx.accounts.vesting;
    vesting.outstanding -= amount;

    Ok(())
}"""

CODE_CPI_SAFE = """pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
    if amount
        > calculator::available_for_withdrawal(
            &ctx.accounts.vesting,
            ctx.accounts.clock.unix_timestamp,
        )
    {
        return err!(ErrorCode::InsufficientWithdrawalBalance);
    }

    // Bookkeeping is updated BEFORE the CPI, so if the token program (or a
    // hook it invokes) called back into this program, it would observe the
    // already-decremented outstanding balance.
    let vesting = &mut ctx.accounts.vesting;
    vesting.outstanding = vesting
        .outstanding
        .checked_sub(amount)
        .ok_or(ErrorCode::InsufficientWithdrawalBalance)?;

    let seeds = &[
        ctx.accounts.vesting.to_account_info().key.as_ref(),
        &[ctx.accounts.vesting.nonce],
    ];
    let signer = &[&seeds[..]];
    let cpi_ctx = CpiContext::from(&*ctx.accounts).with_signer(signer);
    token::transfer(cpi_ctx, amount)?;

    Ok(())
}"""

THINKING_CPI_VULN = """Analyzing the Solana smart contract function `withdraw`.

1. **Function overview**: This function is part of a Solana program. I need to examine the ordering of CPI calls relative to local state mutations.

2. **Expected security checks for V5 (CPI Reentrancy)**:
   - Check 1: validate inputs
   - Check 2: update local state
   - Check 3: perform CPI only after state is updated

3. **Code inspection**: Examining the implementation of `withdraw`:
   - `token::transfer(cpi_ctx, amount)?` is called first, and only afterward does `vesting.outstanding -= amount` update the local bookkeeping. Between the CPI and the bookkeeping update, `vesting.outstanding` still reflects the pre-withdrawal balance.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from performing the CPI before updating local state. If the invoked token program (or an extension hook it triggers) were to call back into this program's `withdraw` (or another instruction reading `vesting.outstanding`) before returning, it would see the stale, not-yet-decremented outstanding balance and could authorize a second withdrawal against the same funds.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_CPI_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V5 - CPI Reentrancy
### Explanation: token::transfer is invoked before vesting.outstanding is decremented. A callback during the CPI would observe the stale (pre-withdrawal) outstanding balance, enabling a double withdrawal.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""

THINKING_CPI_SAFE = """Analyzing the Solana smart contract function `withdraw`.

1. **Function overview**: This function is part of a Solana program. I need to examine the ordering of CPI calls relative to local state mutations.

2. **Expected security checks for V5 (CPI Reentrancy)**:
   - Check 1: validate inputs
   - Check 2: update local state
   - Check 3: perform CPI only after state is updated

3. **Code inspection**: Examining the implementation of `withdraw`:
   - `vesting.outstanding` is decremented (via `checked_sub`) and persisted before `token::transfer` is invoked. By the time the CPI executes, the local state already reflects the withdrawal.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V5 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract updates its state before performing the cross-program invocation."""

CONTENT_CPI_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: vesting.outstanding is decremented before the token::transfer CPI is issued, so no callback during the CPI can observe stale pre-withdrawal state.
### Risk: No significant risk identified."""


# ============================================================
# 4. Unchecked External Calls -- anchor/tests/escrow, fn exchange
# ============================================================

CODE_UC_SAFE = """pub fn exchange(ctx: Context<Exchange>) -> Result<()> {
    let (_pda, bump_seed) = Pubkey::find_program_address(&[ESCROW_PDA_SEED], ctx.program_id);
    let seeds = &[&ESCROW_PDA_SEED[..], &[bump_seed]];

    token_interface::transfer_checked(
        ctx.accounts.into_transfer_to_taker_context().with_signer(&[&seeds[..]]),
        ctx.accounts.escrow_account.initializer_amount,
        ctx.accounts.receive_mint.decimals,
    )?;

    token_interface::transfer_checked(
        ctx.accounts.into_transfer_to_initializer_context(),
        ctx.accounts.escrow_account.taker_amount,
        ctx.accounts.deposit_mint.decimals,
    )?;

    token_interface::set_authority(
        ctx.accounts.into_set_authority_context().with_signer(&[&seeds[..]]),
        AuthorityType::AccountOwner,
        Some(ctx.accounts.escrow_account.initializer_key),
    )?;

    Ok(())
}"""

CODE_UC_VULN = """pub fn exchange(ctx: Context<Exchange>) -> Result<()> {
    let (_pda, bump_seed) = Pubkey::find_program_address(&[ESCROW_PDA_SEED], ctx.program_id);
    let seeds = &[&ESCROW_PDA_SEED[..], &[bump_seed]];

    token_interface::transfer_checked(
        ctx.accounts.into_transfer_to_taker_context().with_signer(&[&seeds[..]]),
        ctx.accounts.escrow_account.initializer_amount,
        ctx.accounts.receive_mint.decimals,
    )?;

    // The result of the second transfer (taker paying the initializer) is
    // discarded instead of propagated.
    let _ = token_interface::transfer_checked(
        ctx.accounts.into_transfer_to_initializer_context(),
        ctx.accounts.escrow_account.taker_amount,
        ctx.accounts.deposit_mint.decimals,
    );

    token_interface::set_authority(
        ctx.accounts.into_set_authority_context().with_signer(&[&seeds[..]]),
        AuthorityType::AccountOwner,
        Some(ctx.accounts.escrow_account.initializer_key),
    )?;

    Ok(())
}"""

THINKING_UC_VULN = """Analyzing the Solana smart contract function `exchange`.

1. **Function overview**: This function is part of a Solana program. I need to examine how CPI results are handled.

2. **Expected security checks for V6 (Unchecked External Calls)**:
   - Check 1: all CPI calls (invoke/invoke_signed, or Anchor CPI helpers) propagate their Result with `?`
   - Check 2: no CPI return value discarded with `let _ =` or `.ok()`
   - Check 3: a failed CPI halts the instruction

3. **Code inspection**: Examining the implementation of `exchange`:
   - The first `transfer_checked` (taker receives the initializer's deposit) and the `set_authority` call both propagate their result with `?`. The middle `transfer_checked` (taker pays the initializer) is called as `let _ = token_interface::transfer_checked(...)`, discarding its result.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from an unchecked CPI return value: if the taker's payment transfer fails (insufficient balance, frozen account, etc.), execution continues anyway -- the escrow's PDA-held deposit is still handed to the taker via the first transfer, and the authority handover still completes, so the initializer receives nothing while the taker walks away with the funds.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_UC_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V6 - Unchecked External Calls
### Explanation: The transfer_checked call that pays the initializer discards its Result with `let _ =` instead of propagating it. If that transfer fails, the exchange still completes as if it had succeeded, letting the taker receive funds without paying.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""

THINKING_UC_SAFE = """Analyzing the Solana smart contract function `exchange`.

1. **Function overview**: This function is part of a Solana program. I need to examine how CPI results are handled.

2. **Expected security checks for V6 (Unchecked External Calls)**:
   - Check 1: all CPI calls propagate their Result with `?`
   - Check 2: no CPI return value discarded with `let _ =` or `.ok()`
   - Check 3: a failed CPI halts the instruction

3. **Code inspection**: Examining the implementation of `exchange`:
   - All three CPI calls (`transfer_checked` twice, `set_authority`) propagate their result with `?`. A failure in any of them aborts the instruction before the later steps run.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V6 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract properly checks all cross-program invocation results."""

CONTENT_UC_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: All three CPI calls in exchange propagate their Result with `?`, so a failure in any transfer or authority change aborts the instruction instead of silently continuing.
### Risk: No significant risk identified."""


# ============================================================
# 5. Integer Overflow -- anchor/tests/ido-pool,
#    fn exchange_redeemable_for_watermelon
# ============================================================

CODE_IO_SAFE = """pub fn exchange_redeemable_for_watermelon(
    ctx: Context<ExchangeRedeemableForWatermelon>,
    amount: u64,
) -> Result<()> {
    if ctx.accounts.user_redeemable.amount < amount {
        return err!(ErrorCode::LowRedeemable);
    }

    // Calculate watermelon tokens due.
    let watermelon_amount = (amount as u128)
        .checked_mul(ctx.accounts.pool_watermelon.amount as u128)
        .unwrap()
        .checked_div(ctx.accounts.redeemable_mint.supply as u128)
        .unwrap();

    token::burn(ctx.accounts.into_burn_context(), amount)?;
    token::transfer(ctx.accounts.into_transfer_context(), watermelon_amount as u64)?;

    Ok(())
}"""

CODE_IO_VULN = """pub fn exchange_redeemable_for_watermelon(
    ctx: Context<ExchangeRedeemableForWatermelon>,
    amount: u64,
) -> Result<()> {
    if ctx.accounts.user_redeemable.amount < amount {
        return err!(ErrorCode::LowRedeemable);
    }

    // Calculate watermelon tokens due using raw arithmetic instead of the
    // checked variants: the multiplication can silently wrap on large pool
    // balances before the division ever runs.
    let watermelon_amount = (amount as u128 * ctx.accounts.pool_watermelon.amount as u128)
        / ctx.accounts.redeemable_mint.supply as u128;

    token::burn(ctx.accounts.into_burn_context(), amount)?;
    token::transfer(ctx.accounts.into_transfer_context(), watermelon_amount as u64)?;

    Ok(())
}"""

THINKING_IO_VULN = """Analyzing the Solana smart contract function `exchange_redeemable_for_watermelon`.

1. **Function overview**: This function is part of a Solana program. I need to examine its arithmetic for overflow/underflow protection.

2. **Expected security checks for V8 (Integer Overflow)**:
   - Check 1: arithmetic uses checked_add/checked_sub/checked_mul/checked_div
   - Check 2: type casts between u128 and u64 are range-checked
   - Check 3: no raw +, -, * on financial calculations

3. **Code inspection**: Examining the implementation of `exchange_redeemable_for_watermelon`:
   - `watermelon_amount` is computed as `(amount as u128 * pool_watermelon.amount as u128) / redeemable_mint.supply as u128` using raw `*` and `/` instead of `checked_mul`/`checked_div`.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from unchecked arithmetic: even in u128, a large enough `amount` and `pool_watermelon.amount` can overflow the multiplication and wrap around silently (release-mode Rust does not panic on this by default), producing a corrupted, potentially tiny or attacker-favorable `watermelon_amount` that is then paid out via `token::transfer`.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_IO_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V8 - Integer Overflow
### Explanation: The watermelon payout is computed with raw * and / instead of checked_mul/checked_div. A sufficiently large multiplication can wrap around silently, corrupting the amount paid out.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""

THINKING_IO_SAFE = """Analyzing the Solana smart contract function `exchange_redeemable_for_watermelon`.

1. **Function overview**: This function is part of a Solana program. I need to examine its arithmetic for overflow/underflow protection.

2. **Expected security checks for V8 (Integer Overflow)**:
   - Check 1: arithmetic uses checked_add/checked_sub/checked_mul/checked_div
   - Check 2: type casts between u128 and u64 are range-checked
   - Check 3: no raw +, -, * on financial calculations

3. **Code inspection**: Examining the implementation of `exchange_redeemable_for_watermelon`:
   - `watermelon_amount` is computed with `checked_mul(...).unwrap()` followed by `checked_div(...).unwrap()` on `u128` values. `unwrap()` on `None` panics and aborts the transaction rather than silently wrapping, satisfying the intent of the check even though a hand-rolled error would be more descriptive.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V8 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract uses checked arithmetic for its payout calculation."""

CONTENT_IO_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The watermelon payout is computed with checked_mul/checked_div on u128, which abort the transaction via unwrap() on overflow instead of silently wrapping.
### Risk: No significant risk identified."""


def main():
    train_samples = [
        sample("missing_key_check_vulnerable_214_anchor_multisig_approve", CODE_MKC_VULN, THINKING_MKC_VULN, CONTENT_MKC_VULN),
        sample("missing_key_check_safe_214_anchor_multisig_approve", CODE_MKC_SAFE, THINKING_MKC_SAFE, CONTENT_MKC_SAFE),
        sample("dos_vulnerable_214_anchor_lockup_whitelist_add", CODE_DOS_VULN, THINKING_DOS_VULN, CONTENT_DOS_VULN),
        sample("dos_safe_214_anchor_lockup_whitelist_add", CODE_DOS_SAFE, THINKING_DOS_SAFE, CONTENT_DOS_SAFE),
        sample("cpi_vulnerable_214_anchor_lockup_withdraw", CODE_CPI_VULN, THINKING_CPI_VULN, CONTENT_CPI_VULN),
        sample("cpi_safe_214_anchor_lockup_withdraw", CODE_CPI_SAFE, THINKING_CPI_SAFE, CONTENT_CPI_SAFE),
        sample("unchecked_calls_vulnerable_214_anchor_escrow_exchange", CODE_UC_VULN, THINKING_UC_VULN, CONTENT_UC_VULN),
        sample("unchecked_calls_safe_214_anchor_escrow_exchange", CODE_UC_SAFE, THINKING_UC_SAFE, CONTENT_UC_SAFE),
        sample("integer_overflow_vulnerable_214_anchor_idopool_exchange", CODE_IO_VULN, THINKING_IO_VULN, CONTENT_IO_VULN),
        sample("integer_overflow_safe_214_anchor_idopool_exchange", CODE_IO_SAFE, THINKING_IO_SAFE, CONTENT_IO_SAFE),
    ]

    with open(TRAIN_PATH, "a", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Appended {len(train_samples)} samples to {TRAIN_PATH}")


if __name__ == "__main__":
    main()

"""
Second batch of "generic safe" training examples for LLM-Contract-Analyzer's
Solana dataset (see /home/biasi/.claude/plans/tingly-purring-pinwheel.md).

New sources (never used in the 285-sample corpus nor in batch 1), manually
audited against data/knowledge_base/vulnerability_info.json:
  - solana-program/stake-pool (process_remove_validator_from_pool,
    process_cleanup_removed_validator_entries)
  - solana-labs/solana-program-library token-lending (process_init_obligation,
    process_deposit_obligation_collateral)
  - solana-program/associated-token-account (process_recover_nested)
"""

import json
import os

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TRAIN_PATH = os.path.join(REPO_DIR, "data", "training", "dataset_think_format.jsonl")
VAL_PATH = os.path.join(REPO_DIR, "data", "training", "validation_dataset.jsonl")

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


def make_example(example_id, code, thinking, content):
    return {
        "id": example_id,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(code=code)},
            {"role": "assistant", "thinking": thinking, "content": content},
        ],
    }


CODE_6 = '''fn process_remove_validator_from_pool(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let stake_pool_info = next_account_info(account_info_iter)?;
    let staker_info = next_account_info(account_info_iter)?;
    let withdraw_authority_info = next_account_info(account_info_iter)?;
    let validator_list_info = next_account_info(account_info_iter)?;
    let stake_account_info = next_account_info(account_info_iter)?;
    let transient_stake_account_info = next_account_info(account_info_iter)?;
    let clock_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(clock_info)?;
    let stake_program_info = next_account_info(account_info_iter)?;

    check_stake_program(stake_program_info.key)?;
    check_account_owner(stake_pool_info, program_id)?;

    let mut stake_pool = try_from_slice_unchecked::<StakePool>(&stake_pool_info.data.borrow())?;
    if !stake_pool.is_valid() {
        return Err(StakePoolError::InvalidState.into());
    }

    stake_pool.check_authority_withdraw(withdraw_authority_info.key, program_id, stake_pool_info.key)?;
    stake_pool.check_staker(staker_info)?;

    if stake_pool.last_update_epoch < clock.epoch {
        return Err(StakePoolError::StakeListAndPoolOutOfDate.into());
    }

    stake_pool.check_validator_list(validator_list_info)?;
    check_account_owner(validator_list_info, program_id)?;
    let mut validator_list_data = validator_list_info.data.borrow_mut();
    let (header, mut validator_list) = ValidatorListHeader::deserialize_vec(&mut validator_list_data)?;
    if !header.is_valid() {
        return Err(StakePoolError::InvalidState.into());
    }

    let (_, stake) = get_stake_state(stake_account_info)?;
    let vote_account_address = stake.delegation.voter_pubkey;
    let maybe_validator_stake_info = validator_list.find_mut::<ValidatorStakeInfo, _>(|x| {
        ValidatorStakeInfo::memcmp_pubkey(x, &vote_account_address)
    });
    if maybe_validator_stake_info.is_none() {
        return Err(StakePoolError::ValidatorNotFound.into());
    }
    let validator_stake_info = maybe_validator_stake_info.unwrap();
    check_validator_stake_address(
        program_id,
        stake_pool_info.key,
        stake_account_info.key,
        &vote_account_address,
        NonZeroU32::new(validator_stake_info.validator_seed_suffix.into()),
    )?;

    if validator_stake_info.status != StakeStatus::Active.into() {
        return Err(StakePoolError::ValidatorNotFound.into());
    }

    let new_status = if u64::from(validator_stake_info.transient_stake_lamports) > 0 {
        check_transient_stake_address(
            program_id,
            stake_pool_info.key,
            transient_stake_account_info.key,
            &vote_account_address,
            validator_stake_info.transient_seed_suffix.into(),
        )?;

        match get_stake_state(transient_stake_account_info) {
            Ok((meta, stake)) if stake_is_usable_by_pool(&meta, withdraw_authority_info.key, &stake_pool.lockup) => {
                if stake.delegation.deactivation_epoch == Epoch::MAX {
                    Self::stake_deactivate(
                        transient_stake_account_info.clone(),
                        clock_info.clone(),
                        withdraw_authority_info.clone(),
                        stake_pool_info.key,
                        AUTHORITY_WITHDRAW,
                        stake_pool.stake_withdraw_bump_seed,
                    )?;
                }
            }
            _ => (),
        }
        StakeStatus::DeactivatingAll
    } else {
        StakeStatus::DeactivatingValidator
    };

    if stake.delegation.deactivation_epoch == Epoch::MAX {
        Self::stake_deactivate(
            stake_account_info.clone(),
            clock_info.clone(),
            withdraw_authority_info.clone(),
            stake_pool_info.key,
            AUTHORITY_WITHDRAW,
            stake_pool.stake_withdraw_bump_seed,
        )?;
    }

    validator_stake_info.status = new_status.into();

    if stake_pool.preferred_deposit_validator_vote_address == Some(vote_account_address) {
        stake_pool.preferred_deposit_validator_vote_address = None;
    }
    if stake_pool.preferred_withdraw_validator_vote_address == Some(vote_account_address) {
        stake_pool.preferred_withdraw_validator_vote_address = None;
    }
    borsh::to_writer(&mut stake_pool_info.data.borrow_mut()[..], &stake_pool)?;

    Ok(())
}'''

THINKING_6 = """Analyzing `process_remove_validator_from_pool` from the SPL Stake Pool program.

1. **Function overview**: Lets the pool's staker authority mark a validator for removal, deactivating its stake (and any pending transient stake) so it can eventually be cleaned up.

2. **Missing Key Check**: `stake_pool.check_authority_withdraw(withdraw_authority_info.key, ...)` and `stake_pool.check_staker(staker_info)?` both validate that the caller holds the correct pool authority before any state changes. A privileged operation is correctly gated.

3. **Type Confusion**: `check_account_owner(stake_pool_info, program_id)` plus `try_from_slice_unchecked::<StakePool>(...)` followed by `stake_pool.is_valid()` — despite the name, "unchecked" here refers to skipping Borsh's internal length prefix, not skipping validation; the subsequent `is_valid()` call performs the actual discriminator check. The validator list undergoes the same owner-check + header.is_valid() pattern. `get_stake_state(...)` is a typed helper. `check_validator_stake_address` / `check_transient_stake_address` verify the provided stake accounts' addresses against their canonical PDA derivation before use.

4. **CPI Reentrancy**: `Self::stake_deactivate(...)` performs a CPI into the native Solana Stake program. The native Stake program cannot invoke arbitrary callbacks into this program — it has no mechanism to call back into an invoking program — so there is no reentrancy attack surface here regardless of whether local bookkeeping (`validator_stake_info.status`, preferred-validator resets) is written before or after this call. No fund transfer occurs in this function either.

5. **Unchecked External Calls**: Both `Self::stake_deactivate(...)` calls are invoked with `?`, propagating any failure.

6. **Integer Overflow/Underflow**: No arithmetic on financial values; only equality/epoch comparisons.

7. **Bump Seed Canonicalization**: `stake_pool.stake_withdraw_bump_seed` is a bump stored on-chain when the pool was created, not a value supplied fresh by the caller for this instruction — matching the "verified against a stored value" safe pattern. `check_validator_stake_address`/`check_transient_stake_address` independently re-derive and check the stake account addresses.

8. **Denial of Service**: There is an explicit re-processing guard: `if validator_stake_info.status != StakeStatus::Active.into() { return Err(...) }` prevents removing an already-removed validator twice. The list lookup (`find_mut`) is bounded by the pool's existing validator count, not by unbounded attacker input.

All seven categories are properly covered."""

CONTENT_6 = """The contract correctly gates this privileged removal operation behind staker/withdraw-authority checks and safely deactivates stake via the non-reentrant native Stake program.
### Vulnerability: Not Vulnerable
### Explanation: Caller authority is validated via check_authority_withdraw/check_staker before any state change, all stake account addresses are checked against their canonical PDA derivation, the CPI target (native Stake program) cannot reenter the caller, and an explicit status guard prevents double-removal.
### Risk: No significant risk identified."""


CODE_7 = '''fn process_cleanup_removed_validator_entries(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let stake_pool_info = next_account_info(account_info_iter)?;
    let validator_list_info = next_account_info(account_info_iter)?;

    check_account_owner(stake_pool_info, program_id)?;
    let stake_pool = try_from_slice_unchecked::<StakePool>(&stake_pool_info.data.borrow())?;
    if !stake_pool.is_valid() {
        return Err(StakePoolError::InvalidState.into());
    }
    stake_pool.check_validator_list(validator_list_info)?;

    check_account_owner(validator_list_info, program_id)?;
    let mut validator_list_data = validator_list_info.data.borrow_mut();
    let (header, mut validator_list) = ValidatorListHeader::deserialize_vec(&mut validator_list_data)?;
    if !header.is_valid() {
        return Err(StakePoolError::InvalidState.into());
    }

    validator_list.retain::<ValidatorStakeInfo, _>(|x| !ValidatorStakeInfo::is_removed(x))?;

    if stake_pool_info.is_writable {
        let mut stake_pool = try_from_slice_unchecked::<StakePool>(&stake_pool_info.data.borrow())?;

        if let Some(preferred_deposit) = stake_pool.preferred_deposit_validator_vote_address {
            let maybe_validator = validator_list.find::<ValidatorStakeInfo, _>(|x| {
                ValidatorStakeInfo::memcmp_pubkey(x, &preferred_deposit)
            });
            let should_reset = match maybe_validator {
                Some(validator) => !matches!(validator.status.try_into(), Ok(StakeStatus::Active)),
                None => true,
            };
            if should_reset {
                stake_pool.preferred_deposit_validator_vote_address = None;
            }
        }

        if let Some(preferred_withdraw) = stake_pool.preferred_withdraw_validator_vote_address {
            let maybe_validator = validator_list.find::<ValidatorStakeInfo, _>(|x| {
                ValidatorStakeInfo::memcmp_pubkey(x, &preferred_withdraw)
            });
            let should_reset = match maybe_validator {
                Some(validator) => !matches!(validator.status.try_into(), Ok(StakeStatus::Active)),
                None => true,
            };
            if should_reset {
                stake_pool.preferred_withdraw_validator_vote_address = None;
            }
        }

        borsh::to_writer(&mut stake_pool_info.data.borrow_mut()[..], &stake_pool)?;
    }

    Ok(())
}'''

THINKING_7 = """Analyzing `process_cleanup_removed_validator_entries` from the SPL Stake Pool program.

1. **Function overview**: A permissionless maintenance instruction that prunes fully-removed validator entries from the validator list and resets stale preferred-validator pointers.

2. **Missing Key Check**: This is a harmless, permissionless bookkeeping operation by design — it only deletes already-dead list entries and clears pointers that no longer resolve to an active validator. There is no fund movement or privileged state change that would require an authority check.

3. **Type Confusion**: Both `stake_pool_info` and `validator_list_info` go through `check_account_owner` plus the `is_valid()`/`header.is_valid()` discriminator checks before their data is trusted.

4. **CPI Reentrancy**: No cross-program invocation occurs anywhere in this function.

5. **Unchecked External Calls**: Not applicable — no CPI calls.

6. **Integer Overflow/Underflow**: No arithmetic operations are performed at all.

7. **Bump Seed Canonicalization**: No PDA derivation occurs in this function.

8. **Denial of Service**: This function exists specifically to *prevent* unbounded growth of the validator list (removing dead entries), and the `retain()` call operates over the pool's existing (bounded) validator list, not unbounded attacker-supplied input. Not an initializer, so re-initialization guards do not apply.

All categories are either explicitly satisfied or not applicable to this permissionless cleanup instruction."""

CONTENT_7 = """The contract performs only harmless, permissionless list maintenance with no fund movement, CPI, or arithmetic.
### Vulnerability: Not Vulnerable
### Explanation: Both accounts are owner- and discriminator-checked before use, the operation only prunes already-removed entries and stale pointers, and there is no CPI or arithmetic surface to exploit.
### Risk: No significant risk identified."""


CODE_8 = '''#[inline(never)]
fn process_init_obligation(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let obligation_info = next_account_info(account_info_iter)?;
    let lending_market_info = next_account_info(account_info_iter)?;
    let obligation_owner_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(next_account_info(account_info_iter)?)?;
    let rent = &Rent::from_account_info(next_account_info(account_info_iter)?)?;
    let token_program_id = next_account_info(account_info_iter)?;

    assert_rent_exempt(rent, obligation_info)?;
    let mut obligation = assert_uninitialized::<Obligation>(obligation_info)?;
    if obligation_info.owner != program_id {
        msg!("Obligation provided is not owned by the lending program");
        return Err(LendingError::InvalidAccountOwner.into());
    }

    let lending_market = LendingMarket::unpack(&lending_market_info.data.borrow())?;
    if lending_market_info.owner != program_id {
        msg!("Lending market provided is not owned by the lending program");
        return Err(LendingError::InvalidAccountOwner.into());
    }
    if &lending_market.token_program_id != token_program_id.key {
        msg!("Lending market token program does not match the token program provided");
        return Err(LendingError::InvalidTokenProgram.into());
    }

    if !obligation_owner_info.is_signer {
        msg!("Obligation owner provided must be a signer");
        return Err(LendingError::InvalidSigner.into());
    }

    obligation.init(InitObligationParams {
        current_slot: clock.slot,
        lending_market: *lending_market_info.key,
        owner: *obligation_owner_info.key,
        deposits: vec![],
        borrows: vec![],
    });
    Obligation::pack(obligation, &mut obligation_info.data.borrow_mut())?;

    Ok(())
}'''

THINKING_8 = """Analyzing `process_init_obligation` from the SPL Token Lending program.

1. **Function overview**: Initializes a new borrower "obligation" account (empty deposits/borrows) tied to a specific lending market and owner.

2. **Missing Key Check**: `if !obligation_owner_info.is_signer { return Err(InvalidSigner) }` explicitly requires the claimed owner to sign the initialization, preventing anyone else from creating an obligation on their behalf with a mismatched owner.

3. **Type Confusion**: `assert_uninitialized::<Obligation>(obligation_info)` performs a typed check that the account is not already an initialized `Obligation` (and implicitly validates its layout). `obligation_info.owner != program_id` and `lending_market_info.owner != program_id` explicitly verify program ownership of both accounts before trusting their contents. `LendingMarket::unpack(...)` is a typed deserialization, and the token program is cross-checked against the value stored in the lending market (`&lending_market.token_program_id != token_program_id.key`).

4. **CPI Reentrancy**: No cross-program invocation occurs in this function.

5. **Unchecked External Calls**: Not applicable — no CPI.

6. **Integer Overflow/Underflow**: No arithmetic; only struct field assignment with empty vectors.

7. **Bump Seed Canonicalization**: No PDA derivation in this function (the obligation account is a separately-created keypair account, not a program-derived address here).

8. **Denial of Service**: Two explicit guards are present: `assert_rent_exempt(rent, obligation_info)?` (rent-exemption check) and `assert_uninitialized::<Obligation>(obligation_info)?` (re-initialization guard) — both of the specific mitigations called for in the DoS category are implemented directly.

All seven categories are properly addressed, with the re-initialization and rent-exemption guards being textbook-correct examples of the required DoS mitigations."""

CONTENT_8 = """The contract explicitly guards against re-initialization and non-rent-exempt accounts, and requires the claimed owner's signature before creating the obligation.
### Vulnerability: Not Vulnerable
### Explanation: assert_uninitialized and assert_rent_exempt guard the initialization, ownership of both input accounts is verified against the program id, and the obligation owner must sign, preventing spoofed ownership.
### Risk: No significant risk identified."""


CODE_9 = '''#[inline(never)]
fn process_deposit_obligation_collateral(
    program_id: &Pubkey,
    collateral_amount: u64,
    accounts: &[AccountInfo],
) -> ProgramResult {
    if collateral_amount == 0 {
        msg!("Collateral amount provided cannot be zero");
        return Err(LendingError::InvalidAmount.into());
    }

    let account_info_iter = &mut accounts.iter();
    let source_collateral_info = next_account_info(account_info_iter)?;
    let destination_collateral_info = next_account_info(account_info_iter)?;
    let deposit_reserve_info = next_account_info(account_info_iter)?;
    let obligation_info = next_account_info(account_info_iter)?;
    let lending_market_info = next_account_info(account_info_iter)?;
    let obligation_owner_info = next_account_info(account_info_iter)?;
    let user_transfer_authority_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(next_account_info(account_info_iter)?)?;
    let token_program_id = next_account_info(account_info_iter)?;

    let lending_market = LendingMarket::unpack(&lending_market_info.data.borrow())?;
    if lending_market_info.owner != program_id {
        return Err(LendingError::InvalidAccountOwner.into());
    }
    if &lending_market.token_program_id != token_program_id.key {
        return Err(LendingError::InvalidTokenProgram.into());
    }

    let deposit_reserve = Reserve::unpack(&deposit_reserve_info.data.borrow())?;
    if deposit_reserve_info.owner != program_id {
        return Err(LendingError::InvalidAccountOwner.into());
    }
    if &deposit_reserve.lending_market != lending_market_info.key {
        return Err(LendingError::InvalidAccountInput.into());
    }
    if &deposit_reserve.collateral.supply_pubkey == source_collateral_info.key {
        return Err(LendingError::InvalidAccountInput.into());
    }
    if &deposit_reserve.collateral.supply_pubkey != destination_collateral_info.key {
        return Err(LendingError::InvalidAccountInput.into());
    }
    if deposit_reserve.last_update.is_stale(clock.slot)? {
        return Err(LendingError::ReserveStale.into());
    }
    if deposit_reserve.config.loan_to_value_ratio == 0 {
        return Err(LendingError::ReserveCollateralDisabled.into());
    }

    let mut obligation = Obligation::unpack(&obligation_info.data.borrow())?;
    if obligation_info.owner != program_id {
        return Err(LendingError::InvalidAccountOwner.into());
    }
    if &obligation.lending_market != lending_market_info.key {
        return Err(LendingError::InvalidAccountInput.into());
    }
    if &obligation.owner != obligation_owner_info.key {
        return Err(LendingError::InvalidObligationOwner.into());
    }
    if !obligation_owner_info.is_signer {
        return Err(LendingError::InvalidSigner.into());
    }

    obligation
        .find_or_add_collateral_to_deposits(*deposit_reserve_info.key)?
        .deposit(collateral_amount)?;
    obligation.last_update.mark_stale();
    Obligation::pack(obligation, &mut obligation_info.data.borrow_mut())?;

    spl_token_transfer(TokenTransferParams {
        source: source_collateral_info.clone(),
        destination: destination_collateral_info.clone(),
        amount: collateral_amount,
        authority: user_transfer_authority_info.clone(),
        authority_signer_seeds: &[],
        token_program: token_program_id.clone(),
    })?;

    Ok(())
}'''

THINKING_9 = """Analyzing `process_deposit_obligation_collateral` from the SPL Token Lending program.

1. **Function overview**: Moves collateral tokens from a user's account into the reserve's collateral supply and records the deposit against the user's obligation.

2. **Missing Key Check**: `if &obligation.owner != obligation_owner_info.key` cross-checks the stored owner against the provided account, and `if !obligation_owner_info.is_signer` requires that owner to sign. The actual token movement is performed via `spl_token_transfer` with `user_transfer_authority_info` as authority, which the SPL Token program itself will validate as a signer/delegate during the CPI.

3. **Type Confusion**: Every account touched is unpacked with a typed helper and then explicitly checked for program ownership: `LendingMarket::unpack` + owner check, `Reserve::unpack` + owner check, `Obligation::unpack` + owner check. Cross-references are also verified (deposit reserve's `lending_market` field must match the provided lending market; obligation's `lending_market` must also match; the destination collateral account must equal the reserve's registered collateral supply, and the source must specifically NOT be that same supply account, preventing self-referential confusion).

4. **CPI Reentrancy**: Local state is fully updated — `obligation.find_or_add_collateral_to_deposits(...).deposit(...)`, `obligation.last_update.mark_stale()`, and `Obligation::pack(...)` writing the updated obligation back to the account — all BEFORE the `spl_token_transfer(...)` CPI at the end. This is the correct checks-effects-interactions ordering: state is committed first, then the external call is made.

5. **Unchecked External Calls**: `spl_token_transfer(...)?` propagates its result with `?`.

6. **Integer Overflow/Underflow**: `collateral_amount == 0` is explicitly rejected up front. No raw arithmetic operators appear directly in this function on the collateral amount (the increment happens inside the `.deposit(...)` helper, which is not shown here, so I rely on the explicit zero-amount guard and the absence of any raw `+`/`-`/`*` in this function body).

7. **Bump Seed Canonicalization**: No PDA derivation occurs in this function.

8. **Denial of Service**: `deposit_reserve.last_update.is_stale(clock.slot)?` explicitly rejects stale reserve data — the freshness-check mitigation called for in the DoS category. Not an initializer, no unbounded loops.

All seven categories are addressed, with especially thorough cross-account validation."""

CONTENT_9 = """The contract cross-validates every account relationship (lending market, reserve, obligation, collateral supply) before mutating state, updates its own bookkeeping before the token transfer CPI, and rejects stale reserve data.
### Vulnerability: Not Vulnerable
### Explanation: Obligation ownership and signer status are checked, all accounts are unpacked with typed helpers and owner-checked, local obligation state is updated before the CPI (correct checks-effects-interactions order), and stale reserve data is explicitly rejected.
### Risk: No significant risk identified."""


CODE_10 = '''pub fn process_recover_nested(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let nested_associated_token_account_info = next_account_info(account_info_iter)?;
    let nested_token_mint_info = next_account_info(account_info_iter)?;
    let destination_associated_token_account_info = next_account_info(account_info_iter)?;
    let owner_associated_token_account_info = next_account_info(account_info_iter)?;
    let owner_token_mint_info = next_account_info(account_info_iter)?;
    let wallet_account_info = next_account_info(account_info_iter)?;
    let spl_token_program_info = next_account_info(account_info_iter)?;
    let spl_token_program_id = spl_token_program_info.key;

    let (owner_associated_token_address, bump_seed) = get_associated_token_address_and_bump_seed_internal(
        wallet_account_info.key,
        owner_token_mint_info.key,
        program_id,
        spl_token_program_id,
    );
    if owner_associated_token_address != *owner_associated_token_account_info.key {
        return Err(ProgramError::InvalidSeeds);
    }

    let (nested_associated_token_address, _) = get_associated_token_address_and_bump_seed_internal(
        owner_associated_token_account_info.key,
        nested_token_mint_info.key,
        program_id,
        spl_token_program_id,
    );
    if nested_associated_token_address != *nested_associated_token_account_info.key {
        return Err(ProgramError::InvalidSeeds);
    }

    let (destination_associated_token_address, _) = get_associated_token_address_and_bump_seed_internal(
        wallet_account_info.key,
        nested_token_mint_info.key,
        program_id,
        spl_token_program_id,
    );
    if destination_associated_token_address != *destination_associated_token_account_info.key {
        return Err(ProgramError::InvalidSeeds);
    }

    if !wallet_account_info.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    if owner_token_mint_info.owner != spl_token_program_id {
        return Err(ProgramError::IllegalOwner);
    }

    let (amount, decimals) = {
        if owner_associated_token_account_info.owner != spl_token_program_id {
            return Err(ProgramError::IllegalOwner);
        }
        let owner_account_data = owner_associated_token_account_info.data.borrow();
        let owner_account = StateWithExtensions::<Account>::unpack(&owner_account_data)?;
        if owner_account.base.owner != *wallet_account_info.key {
            return Err(AssociatedTokenAccountError::InvalidOwner.into());
        }

        if nested_associated_token_account_info.owner != spl_token_program_id {
            return Err(ProgramError::IllegalOwner);
        }
        let nested_account_data = nested_associated_token_account_info.data.borrow();
        let nested_account = StateWithExtensions::<Account>::unpack(&nested_account_data)?;
        if nested_account.base.owner != *owner_associated_token_account_info.key {
            return Err(AssociatedTokenAccountError::InvalidOwner.into());
        }
        let amount = nested_account.base.amount;

        if nested_token_mint_info.owner != spl_token_program_id {
            return Err(ProgramError::IllegalOwner);
        }
        let nested_mint_data = nested_token_mint_info.data.borrow();
        let nested_mint = StateWithExtensions::<Mint>::unpack(&nested_mint_data)?;
        let decimals = nested_mint.base.decimals;
        (amount, decimals)
    };

    let owner_associated_token_account_signer_seeds: &[&[_]] = &[
        &wallet_account_info.key.to_bytes(),
        &spl_token_program_id.to_bytes(),
        &owner_token_mint_info.key.to_bytes(),
        &[bump_seed],
    ];
    invoke_signed(
        &spl_token_2022_interface::instruction::transfer_checked(
            spl_token_program_id,
            nested_associated_token_account_info.key,
            nested_token_mint_info.key,
            destination_associated_token_account_info.key,
            owner_associated_token_account_info.key,
            &[],
            amount,
            decimals,
        )?,
        &[
            nested_associated_token_account_info.clone(),
            nested_token_mint_info.clone(),
            destination_associated_token_account_info.clone(),
            owner_associated_token_account_info.clone(),
            spl_token_program_info.clone(),
        ],
        &[owner_associated_token_account_signer_seeds],
    )?;

    invoke_signed(
        &spl_token_2022_interface::instruction::close_account(
            spl_token_program_id,
            nested_associated_token_account_info.key,
            wallet_account_info.key,
            owner_associated_token_account_info.key,
            &[],
        )?,
        &[
            nested_associated_token_account_info.clone(),
            wallet_account_info.clone(),
            owner_associated_token_account_info.clone(),
            spl_token_program_info.clone(),
        ],
        &[owner_associated_token_account_signer_seeds],
    )
}'''

THINKING_10 = """Analyzing `process_recover_nested` from the SPL Associated Token Account program.

1. **Function overview**: Recovers tokens mistakenly sent to a "nested" associated token account (an ATA owned by another ATA, a common user error) by transferring the balance out to the wallet's correct top-level ATA and closing the nested account.

2. **Missing Key Check**: `if !wallet_account_info.is_signer { return Err(MissingRequiredSignature) }` explicitly requires the wallet owner to sign before anything is moved.

3. **Type Confusion**: All three token accounts involved (owner ATA, nested ATA, destination ATA) are validated by independently re-deriving their expected addresses via `get_associated_token_address_and_bump_seed_internal` and comparing against the provided accounts (`InvalidSeeds` on mismatch) — this prevents substituting an unrelated account for any of the three roles. Ownership by the token program is checked (`.owner != spl_token_program_id`) before every `StateWithExtensions::<Account>::unpack` / `<Mint>::unpack` call, and the on-chain `owner`/`amount`/`decimals` fields are cross-checked against the expected wallet and hierarchy (owner ATA must belong to the wallet, nested ATA must belong to the owner ATA).

4. **CPI Reentrancy**: The transfer CPI happens first, then the close CPI — the nested account's balance is emptied before it is closed, which is the correct order (closing first would either fail or strand funds, not create a reentrancy issue, but the chosen order is also the only sound one for a closing operation).

5. **Unchecked External Calls**: Both `invoke_signed(...)` calls propagate errors — the first via `?`, the second as the function's own tail-position return value (its `Result` is returned directly, not discarded).

6. **Integer Overflow/Underflow**: `amount` and `decimals` are read directly from account data and passed through to the transfer instruction unmodified; no arithmetic is performed on them.

7. **Bump Seed Canonicalization**: `get_associated_token_address_and_bump_seed_internal` is the canonical ATA derivation function (equivalent to `find_program_address` for this account type) and the returned `bump_seed` is used exactly as computed, not overridden by caller input.

8. **Denial of Service**: Not an initializer; no loops; no oracle reads.

All seven categories are properly handled, with particularly thorough re-derivation checks on every account involved to prevent substitution attacks."""

CONTENT_10 = """The contract independently re-derives and validates every associated-token-account address involved before moving funds, requires the wallet owner's signature, and transfers out the full balance before closing the nested account.
### Vulnerability: Not Vulnerable
### Explanation: Owner, nested, and destination ATA addresses are all re-derived and checked against canonical seeds, the wallet must sign, account ownership and hierarchy are validated via typed unpacking before any CPI, and both CPI results are propagated.
### Risk: No significant risk identified."""


TRAIN_EXAMPLES = [
    make_example("generic_safe_006_stakepool_remove_validator", CODE_6, THINKING_6, CONTENT_6),
    make_example("generic_safe_007_stakepool_cleanup_removed", CODE_7, THINKING_7, CONTENT_7),
    make_example("generic_safe_008_lending_init_obligation", CODE_8, THINKING_8, CONTENT_8),
    make_example("generic_safe_009_lending_deposit_obligation_collateral", CODE_9, THINKING_9, CONTENT_9),
]

VAL_EXAMPLES = [
    make_example("generic_safe_010_ata_recover_nested", CODE_10, THINKING_10, CONTENT_10),
]


def append_jsonl(path, examples):
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Appended {len(examples)} examples to {path}")


if __name__ == "__main__":
    append_jsonl(TRAIN_PATH, TRAIN_EXAMPLES)
    append_jsonl(VAL_PATH, VAL_EXAMPLES)

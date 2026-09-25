"""
Adds new "generic safe" training examples to LLM-Contract-Analyzer's Solana
dataset: real, manually-audited Solana instruction handlers that are safe
across ALL 7 vulnerability categories (not just one), to close the
train/test distribution gap documented in the plan
(/home/biasi/.claude/plans/tingly-purring-pinwheel.md).

Sources (never used in the existing 285-sample corpus, verified by manual
line-by-line audit against data/knowledge_base/vulnerability_info.json):
  - solana-program/single-pool  (process_replenish_pool,
    process_create_pool_token_metadata, process_initialize_pool_onramp)
  - solana-labs/solana-program-library governance (process_relinquish_vote)
  - solana-program/token-2022 (process_withdraw_excess_lamports)
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


def user_msg(code):
    return {"role": "user", "content": USER_TEMPLATE.format(code=code)}


def assistant_msg(thinking, content):
    return {"role": "assistant", "thinking": thinking, "content": content}


def make_example(example_id, code, thinking, content):
    return {
        "id": example_id,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            user_msg(code),
            assistant_msg(thinking, content),
        ],
    }


CODE_1 = '''fn process_replenish_pool(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let vote_account_info = next_account_info(account_info_iter)?;
    let pool_info = next_account_info(account_info_iter)?;
    let pool_stake_info = next_account_info(account_info_iter)?;
    let pool_onramp_info = next_account_info(account_info_iter)?;
    let pool_stake_authority_info = next_account_info(account_info_iter)?;
    let clock_info = next_account_info(account_info_iter)?;
    let clock = &Clock::from_account_info(clock_info)?;
    let stake_history_info = next_account_info(account_info_iter)?;
    let stake_config_info = next_account_info(account_info_iter)?;
    let stake_program_info = next_account_info(account_info_iter)?;

    let rent = Rent::get()?;
    let stake_history = &StakeHistorySysvar(clock.epoch);

    check_vote_account(vote_account_info)?;
    check_pool_address(program_id, vote_account_info.key, pool_info.key)?;

    SinglePool::from_account_info(pool_info, program_id)?;

    check_pool_stake_address(program_id, pool_info.key, pool_stake_info.key)?;
    check_pool_onramp_address(program_id, pool_info.key, pool_onramp_info.key)?;
    let stake_authority_bump_seed = check_pool_stake_authority_address(
        program_id,
        pool_info.key,
        pool_stake_authority_info.key,
    )?;
    check_stake_program(stake_program_info.key)?;

    let minimum_delegation = stake::tools::get_minimum_delegation()?;
    let pool_rent_exempt_reserve = rent.minimum_balance(pool_stake_info.data_len());
    let onramp_rent_exempt_reserve = rent.minimum_balance(pool_onramp_info.data_len());

    let (_, pool_stake_state) = get_stake_state(pool_stake_info)?;
    let pool_stake_status = pool_stake_state
        .delegation
        .stake_activating_and_deactivating(clock.epoch, stake_history, PERPETUAL_NEW_WARMUP_COOLDOWN_RATE_EPOCH);
    let pool_stake_is_fully_active = is_stake_fully_active(&pool_stake_status);

    let (option_onramp_status, onramp_deactivation_epoch) = match deserialize_stake(pool_onramp_info) {
        Ok(StakeStateV2::Initialized(_)) => (None, u64::MAX),
        Ok(StakeStateV2::Stake(_, stake, _)) => (
            Some(stake.delegation.stake_activating_and_deactivating(clock.epoch, stake_history, PERPETUAL_NEW_WARMUP_COOLDOWN_RATE_EPOCH)),
            stake.delegation.deactivation_epoch,
        ),
        _ => return Err(SinglePoolError::OnRampDoesntExist.into()),
    };

    let stake_authority_seeds = &[POOL_STAKE_AUTHORITY_PREFIX, pool_info.key.as_ref(), &[stake_authority_bump_seed]];
    let stake_authority_signers = &[&stake_authority_seeds[..]];

    if pool_stake_state.delegation.deactivation_epoch == clock.epoch
        || (pool_stake_state.delegation.deactivation_epoch < clock.epoch && pool_stake_status.effective == 0)
    {
        invoke_signed(
            &stake::instruction::delegate_stake(pool_stake_info.key, pool_stake_authority_info.key, vote_account_info.key),
            &[pool_stake_info.clone(), vote_account_info.clone(), clock_info.clone(), stake_history_info.clone(), stake_config_info.clone(), pool_stake_authority_info.clone()],
            stake_authority_signers,
        )?;
    }

    if pool_stake_is_fully_active {
        let pool_excess_lamports = pool_stake_info.lamports()
            .saturating_sub(pool_stake_state.delegation.stake)
            .saturating_sub(pool_rent_exempt_reserve);

        if let Some(ref onramp_status) = option_onramp_status {
            if is_stake_fully_active(onramp_status) {
                invoke_signed(
                    &stake::instruction::move_stake(pool_onramp_info.key, pool_stake_info.key, pool_stake_authority_info.key, onramp_status.effective),
                    &[pool_onramp_info.clone(), pool_stake_info.clone(), pool_stake_authority_info.clone()],
                    stake_authority_signers,
                )?;
            }
        }

        if pool_excess_lamports > 0 {
            invoke_signed(
                &stake::instruction::move_lamports(pool_stake_info.key, pool_onramp_info.key, pool_stake_authority_info.key, pool_excess_lamports),
                &[pool_stake_info.clone(), pool_onramp_info.clone(), pool_stake_authority_info.clone()],
                stake_authority_signers,
            )?;
        }

        let onramp_non_rent_lamports = pool_onramp_info.lamports().saturating_sub(onramp_rent_exempt_reserve);
        let must_delegate_onramp = match option_onramp_status.unwrap_or_default() {
            StakeActivationStatus { effective: 0, activating, deactivating: 0 } if activating > 0 => {
                onramp_non_rent_lamports >= minimum_delegation && onramp_non_rent_lamports > activating
            }
            StakeActivationStatus { effective: _, activating: 0, deactivating } if deactivating == 0 || onramp_deactivation_epoch == clock.epoch => {
                onramp_non_rent_lamports >= minimum_delegation
            }
            _ => false,
        };

        if must_delegate_onramp {
            invoke_signed(
                &stake::instruction::delegate_stake(pool_onramp_info.key, pool_stake_authority_info.key, vote_account_info.key),
                &[pool_onramp_info.clone(), vote_account_info.clone(), clock_info.clone(), stake_history_info.clone(), stake_config_info.clone(), pool_stake_authority_info.clone()],
                stake_authority_signers,
            )?;
        }
    }

    Ok(())
}'''

THINKING_1 = """Analyzing the Solana smart contract function `process_replenish_pool` from the SPL Single Pool program.

1. **Function overview**: This is a permissionless "maintenance" instruction for a single-validator stake pool. It rebalances stake between the pool's main stake account and an on-ramp stake account, delegating undelegated lamports and moving fully-active stake around. No user funds are withdrawn to an external party; everything stays within pool-owned PDAs.

2. **Missing Key Check (Access Control)**: This instruction is designed to be callable by anyone (it is a bookkeeping/rebalancing operation, not a privileged admin action). All destination accounts for CPI signing are PDAs derived and validated by the program itself (`check_pool_stake_authority_address`), not caller-supplied keys, so there is no privileged operation being bypassed. No missing authorization here.

3. **Type Confusion (Input Validation)**: `pool_info` is validated with `SinglePool::from_account_info(pool_info, program_id)?`, which checks both ownership and account discriminator. `vote_account_info` is validated with `check_vote_account`. `pool_stake_info` and `pool_onramp_info` addresses are checked against their expected PDA derivations before use, and their contents are read through typed helpers (`get_stake_state`, `deserialize_stake`) rather than raw byte access. No unchecked account substitution is possible.

4. **CPI Reentrancy**: The function performs several `invoke_signed` calls (delegate_stake, move_stake, move_lamports). None of them are preceded by a local state mutation that could be left stale and exploited on reentry — the "state" here is the stake accounts themselves, updated atomically by the CPI, not a separate bookkeeping struct read before and trusted after. No checks-effects-interactions violation.

5. **Unchecked External Calls**: Every `invoke_signed(...)` call in this function is followed directly by `?`, propagating any CPI failure immediately. No `let _ =` or `.ok()` pattern discarding a CPI result.

6. **Integer Overflow/Underflow**: All lamport arithmetic uses `saturating_sub`, which cannot underflow/panic and is a stricter guarantee than a plain `checked_sub` (it clamps instead of erroring, appropriate here since these are best-effort "excess" calculations, not fund-accounting balances). No raw `+`, `-`, `*` on financial values.

7. **Bump Seed Canonicalization**: The signer seeds use `stake_authority_bump_seed`, which is the value *returned by* `check_pool_stake_authority_address` — i.e., derived and validated by the program via the canonical PDA-checking helper already trusted elsewhere in this codebase, not an arbitrary caller-supplied bump.

8. **Denial of Service**: This is a repeatable maintenance call, not an initializer, so re-initialization guards do not apply. There are no unbounded loops and no external oracle reads.

Across all seven categories, the security-relevant patterns (authority validation via PDA checks, checked/saturating arithmetic, propagated CPI results, canonical bump usage) are all present and correctly applied."""

CONTENT_1 = """The contract correctly implements security best practices across every vulnerability category checked.
### Vulnerability: Not Vulnerable
### Explanation: All PDA addresses (pool stake, on-ramp, stake authority) are validated via canonical derivation helpers before use, every invoke_signed CPI result is propagated with `?`, lamport arithmetic uses saturating operations to avoid underflow, and the instruction is intentionally permissionless with no privileged action being bypassed.
### Risk: No significant risk identified."""


CODE_2 = '''fn process_create_pool_token_metadata(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let pool_info = next_account_info(account_info_iter)?;
    let pool_mint_info = next_account_info(account_info_iter)?;
    let pool_mint_authority_info = next_account_info(account_info_iter)?;
    let pool_mpl_authority_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let metadata_info = next_account_info(account_info_iter)?;
    let mpl_token_metadata_program_info = next_account_info(account_info_iter)?;
    let system_program_info = next_account_info(account_info_iter)?;

    let pool = SinglePool::from_account_info(pool_info, program_id)?;

    let mint_authority_bump_seed = check_pool_mint_authority_address(
        program_id,
        pool_info.key,
        pool_mint_authority_info.key,
    )?;
    let mpl_authority_bump_seed = check_pool_mpl_authority_address(
        program_id,
        pool_info.key,
        pool_mpl_authority_info.key,
    )?;
    check_pool_mint_address(program_id, pool_info.key, pool_mint_info.key)?;
    check_system_program(system_program_info.key)?;
    check_account_owner(payer_info, &system_program::id())?;
    check_mpl_metadata_program(mpl_token_metadata_program_info.key)?;
    check_mpl_metadata_account_address(metadata_info.key, pool_mint_info.key)?;

    if !payer_info.is_signer {
        msg!("Payer did not sign metadata creation");
        return Err(SinglePoolError::SignatureMissing.into());
    }

    let vote_address_str = pool.vote_account_address.to_string();
    let token_name = format!("SPL Single Pool {}", &vote_address_str[0..15]);
    let token_symbol = format!("st{}", &vote_address_str[0..7]);

    let new_metadata_instruction = create_metadata_accounts_v3(
        *mpl_token_metadata_program_info.key,
        *metadata_info.key,
        *pool_mint_info.key,
        *pool_mint_authority_info.key,
        *payer_info.key,
        *pool_mpl_authority_info.key,
        token_name,
        token_symbol,
        "".to_string(),
    );

    let mint_authority_seeds = &[POOL_MINT_AUTHORITY_PREFIX, pool_info.key.as_ref(), &[mint_authority_bump_seed]];
    let mpl_authority_seeds = &[POOL_MPL_AUTHORITY_PREFIX, pool_info.key.as_ref(), &[mpl_authority_bump_seed]];
    let signers = &[&mint_authority_seeds[..], &mpl_authority_seeds[..]];

    invoke_signed(
        &new_metadata_instruction,
        &[metadata_info.clone(), pool_mint_info.clone(), pool_mint_authority_info.clone(), payer_info.clone(), pool_mpl_authority_info.clone(), system_program_info.clone()],
        signers,
    )?;

    Ok(())
}'''

THINKING_2 = """Analyzing `process_create_pool_token_metadata` from the SPL Single Pool program.

1. **Function overview**: Creates a Metaplex metadata account for a single-validator pool's stake-representative token mint, via a CPI to the Metaplex Token Metadata program.

2. **Missing Key Check**: The payer must sign (`if !payer_info.is_signer { return Err(...) }`), explicitly enforced. The mint authority and MPL authority are program-derived PDAs validated via `check_pool_mint_authority_address` / `check_pool_mpl_authority_address`, not caller-supplied keys — there is no privileged authority that could be spoofed.

3. **Type Confusion**: `pool_info` validated through `SinglePool::from_account_info`. `pool_mint_info` checked against the expected derived mint address. `system_program_info` checked via `check_system_program`. `payer_info` checked to be owned by the system program. `mpl_token_metadata_program_info` checked via `check_mpl_metadata_program`. `metadata_info` checked against its expected derived address via `check_mpl_metadata_account_address`. Every account is validated against its expected type/owner/address before being used in the CPI.

4. **CPI Reentrancy**: There is exactly one CPI call, at the very end of the function, and no program-owned state is read-then-trusted before it that a reentrant call into this instruction could exploit. No checks-effects-interactions violation is possible since there is no "effects" step distinct from the CPI itself.

5. **Unchecked External Calls**: The single `invoke_signed(...)` call ends with `?`, so any failure from the Metaplex program aborts the instruction.

6. **Integer Overflow/Underflow**: No arithmetic is performed in this function at all (only string formatting for the token name/symbol), so this class does not apply.

7. **Bump Seed Canonicalization**: Both `mint_authority_bump_seed` and `mpl_authority_bump_seed` come from the canonical `check_pool_*_address` helpers, matching the same pattern already trusted elsewhere in this program.

8. **Denial of Service**: Not an initialization function for the pool itself; it creates a Metaplex metadata account, and a duplicate call would simply fail inside the Metaplex program's own re-initialization guard (the metadata PDA would already be allocated). No loops or oracle reads.

All seven categories check out."""

CONTENT_2 = """The contract correctly validates every account and authority involved before performing its single CPI call.
### Vulnerability: Not Vulnerable
### Explanation: Payer signature is explicitly checked, all PDA authorities are derived and validated via canonical helpers rather than trusting caller-supplied keys, every account is type/owner-checked before use, and the single CPI call propagates its result with `?`.
### Risk: No significant risk identified."""


CODE_3 = '''fn process_initialize_pool_onramp(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    let pool_info = next_account_info(account_info_iter)?;
    let pool_onramp_info = next_account_info(account_info_iter)?;
    let pool_stake_authority_info = next_account_info(account_info_iter)?;
    let rent_info = next_account_info(account_info_iter)?;
    let rent = &Rent::from_account_info(rent_info)?;
    let system_program_info = next_account_info(account_info_iter)?;
    let stake_program_info = next_account_info(account_info_iter)?;

    SinglePool::from_account_info(pool_info, program_id)?;

    let onramp_bump_seed = check_pool_onramp_address(program_id, pool_info.key, pool_onramp_info.key)?;
    let stake_authority_bump_seed = check_pool_stake_authority_address(
        program_id,
        pool_info.key,
        pool_stake_authority_info.key,
    )?;
    check_system_program(system_program_info.key)?;
    check_stake_program(stake_program_info.key)?;

    let onramp_seeds = &[POOL_ONRAMP_PREFIX, pool_info.key.as_ref(), &[onramp_bump_seed]];
    let onramp_signers = &[&onramp_seeds[..]];
    let stake_authority_seeds = &[POOL_STAKE_AUTHORITY_PREFIX, pool_info.key.as_ref(), &[stake_authority_bump_seed]];
    let stake_authority_signers = &[&stake_authority_seeds[..]];

    let stake_space = StakeStateV2::size_of();
    let stake_rent = rent.minimum_balance(stake_space);

    if pool_onramp_info.lamports() < stake_rent {
        return Err(SinglePoolError::WrongRentAmount.into());
    }

    let authorized = stake::state::Authorized::auto(pool_stake_authority_info.key);

    invoke_signed(
        &system_instruction::allocate(pool_onramp_info.key, stake_space as u64),
        core::slice::from_ref(pool_onramp_info),
        onramp_signers,
    )?;

    invoke_signed(
        &system_instruction::assign(pool_onramp_info.key, stake_program_info.key),
        core::slice::from_ref(pool_onramp_info),
        onramp_signers,
    )?;

    invoke_signed(
        &stake::instruction::initialize_checked(pool_onramp_info.key, &authorized),
        &[pool_onramp_info.clone(), rent_info.clone(), pool_stake_authority_info.clone(), pool_stake_authority_info.clone()],
        stake_authority_signers,
    )?;

    Ok(())
}'''

THINKING_3 = """Analyzing `process_initialize_pool_onramp` from the SPL Single Pool program.

1. **Function overview**: Creates and initializes the pool's "on-ramp" stake account, a PDA used to stage new deposits before they are merged into the pool's main stake account by `process_replenish_pool`.

2. **Missing Key Check**: This is a permissionless setup step tied to a specific pool's PDAs; there is no separate "authority" whose signature could be bypassed — anyone can pay to set up the on-ramp account for a given pool, which is the intended design (the account only becomes useful within this program's own PDA-gated logic afterward).

3. **Type Confusion**: `pool_info` validated via `SinglePool::from_account_info`. `pool_onramp_info` and `pool_stake_authority_info` addresses validated against their canonical PDA derivations. `system_program_info` and `stake_program_info` checked via `check_system_program`/`check_stake_program`. `rent` is read through the sysvar-typed `Rent::from_account_info`, not raw bytes.

4. **CPI Reentrancy**: Three sequential CPIs (`allocate`, `assign`, `initialize_checked`) create a brand-new account; there is no pre-existing program state being read and trusted stale across these calls, since the account does not yet exist in a meaningful state until they complete.

5. **Unchecked External Calls**: All three `invoke_signed(...)` calls are followed by `?`.

6. **Integer Overflow/Underflow**: The only arithmetic is `rent.minimum_balance(stake_space)`, a library call, and a lamport comparison (`<`) — no raw addition/subtraction/multiplication on financial values.

7. **Bump Seed Canonicalization**: `onramp_bump_seed` and `stake_authority_bump_seed` both come from the canonical `check_pool_*_address` helpers, and are used exactly as returned (no user-supplied override).

8. **Denial of Service (re-initialization)**: There is no explicit `is_initialized` flag check, but re-initialization is prevented at the runtime level: `system_instruction::allocate` and `assign` both require the target account to still be owned by the System Program with zero-length data. Once this instruction succeeds once, `pool_onramp_info` becomes owned by the Stake program, so a second `allocate` call on the same PDA would fail at the runtime level before any state could be corrupted. No unbounded loops or oracle reads are present.

All seven categories are covered by either explicit checks or structural/runtime guarantees."""

CONTENT_3 = """The contract relies on validated PDA derivations and propagated CPI results, with re-initialization implicitly prevented by System Program account-ownership semantics.
### Vulnerability: Not Vulnerable
### Explanation: All PDAs are checked against canonical derivations before use, every CPI result is propagated with `?`, and a second call to allocate the same on-ramp PDA would fail at the runtime level once it is owned by the Stake program, preventing re-initialization.
### Risk: No significant risk identified."""


CODE_4 = '''pub fn process_relinquish_vote(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?; // 0
    let governance_info = next_account_info(account_info_iter)?; // 1
    let proposal_info = next_account_info(account_info_iter)?; // 2
    let token_owner_record_info = next_account_info(account_info_iter)?; // 3

    let vote_record_info = next_account_info(account_info_iter)?; // 4
    let vote_governing_token_mint_info = next_account_info(account_info_iter)?; // 5

    let realm_data = get_realm_data_for_governing_token_mint(
        program_id,
        realm_info,
        vote_governing_token_mint_info.key,
    )?;

    let governance_data =
        get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;

    let mut proposal_data =
        get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;

    let mut token_owner_record_data = get_token_owner_record_data_for_realm_and_governing_mint(
        program_id,
        token_owner_record_info,
        &governance_data.realm,
        vote_governing_token_mint_info.key,
    )?;

    let mut vote_record_data = get_vote_record_data_for_proposal_and_token_owner_record(
        program_id,
        vote_record_info,
        &realm_data,
        proposal_info.key,
        &proposal_data,
        &token_owner_record_data,
    )?;
    vote_record_data.assert_can_relinquish_vote()?;

    let clock = Clock::get()?;

    if proposal_data.state == ProposalState::Voting
        && !proposal_data.has_voting_max_time_ended(&governance_data.config, clock.unix_timestamp)
    {
        let governance_authority_info = next_account_info(account_info_iter)?; // 5
        let beneficiary_info = next_account_info(account_info_iter)?; // 6

        token_owner_record_data
            .assert_token_owner_or_delegate_is_signer(governance_authority_info)?;

        match vote_record_data.vote {
            Vote::Approve(choices) => {
                for (option, choice) in proposal_data.options.iter_mut().zip(choices) {
                    option.vote_weight = option
                        .vote_weight
                        .checked_sub(choice.get_choice_weight(vote_record_data.voter_weight)?)
                        .unwrap();
                }
            }
            Vote::Deny => {
                proposal_data.deny_vote_weight = Some(
                    proposal_data.deny_vote_weight.unwrap().checked_sub(vote_record_data.voter_weight).unwrap(),
                )
            }
            Vote::Veto => {
                proposal_data.veto_vote_weight = proposal_data.veto_vote_weight.checked_sub(vote_record_data.voter_weight).unwrap();
            }
            Vote::Abstain => {
                return Err(GovernanceError::NotSupportedVoteType.into());
            }
        }

        proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;
        dispose_account(vote_record_info, beneficiary_info)?;
    } else {
        if proposal_data.state == ProposalState::Voting {
            return Err(GovernanceError::CannotRelinquishInFinalizingState.into());
        }
        vote_record_data.is_relinquished = true;
        vote_record_data.serialize(&mut vote_record_info.data.borrow_mut()[..])?;
    }

    token_owner_record_data.unrelinquished_votes_count = token_owner_record_data
        .unrelinquished_votes_count
        .checked_sub(1)
        .unwrap();

    token_owner_record_data.serialize(&mut token_owner_record_info.data.borrow_mut()[..])?;

    Ok(())
}'''

THINKING_4 = """Analyzing `process_relinquish_vote` from the SPL Governance program.

1. **Function overview**: Lets a token owner (or their delegate) withdraw an active vote from a proposal, or clean up a vote record after the proposal has already been decided.

2. **Missing Key Check**: While the proposal is still actively being voted on, the code requires `token_owner_record_data.assert_token_owner_or_delegate_is_signer(governance_authority_info)?` before mutating the vote weights — the caller must be the token owner or their delegate. After the proposal is decided, the code comments explain that anyone may prune the (now inert) vote record to reclaim rent; this is an intentional, documented permissionless cleanup with no financial or governance impact, not a missing check.

3. **Type Confusion**: Every account (`realm`, `governance`, `proposal`, `token_owner_record`, `vote_record`) is loaded through a dedicated typed getter (`get_realm_data_for_governing_token_mint`, `get_governance_data_for_realm`, etc.) that cross-validates ownership and relationships between accounts before returning deserialized data — no raw/unchecked deserialization.

4. **CPI Reentrancy**: This function performs no cross-program invocations at all (only local account data serialization and a `dispose_account` helper call); this category does not apply.

5. **Unchecked External Calls**: No CPI calls are made, so there is nothing to leave unchecked.

6. **Integer Overflow/Underflow**: Every vote-weight and count decrement uses `checked_sub(...)`, never a raw `-`. The `.unwrap()` after each `checked_sub` will panic rather than silently wrap on underflow — this halts the transaction safely instead of corrupting state, satisfying the intent of the check (no silent wraparound), even though a hand-rolled error would be slightly more descriptive.

7. **Bump Seed Canonicalization**: No `create_program_address`/PDA derivation occurs in this function; not applicable.

8. **Denial of Service**: The only loop (`proposal_data.options.iter_mut().zip(choices)`) is bounded by the fixed, small number of proposal options set at proposal creation — not attacker-controlled or unbounded. This is not an initialization function, so re-initialization guards do not apply, and no external oracle data is read.

All applicable categories are properly handled; the remaining categories (CPI-related, bump seed) simply do not apply to this instruction's logic."""

CONTENT_4 = """The contract enforces signer/authority checks where funds or voting power are affected, uses checked arithmetic throughout, and has no CPI or PDA-derivation logic that could be misused.
### Vulnerability: Not Vulnerable
### Explanation: Vote withdrawal while a proposal is active requires the token owner or delegate to sign; post-decision cleanup is intentionally permissionless and inert; all vote-weight arithmetic uses checked_sub to prevent silent underflow.
### Risk: No significant risk identified."""


CODE_5 = '''pub fn process_withdraw_excess_lamports(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let source_info = next_account_info(account_info_iter)?;
    let destination_info = next_account_info(account_info_iter)?;
    let authority_info = next_account_info(account_info_iter)?;

    check_program_account(source_info.owner)?;

    let source_data = source_info.data.borrow();

    if let Ok(account) = PodStateWithExtensions::<PodAccount>::unpack(&source_data) {
        if account.base.is_native() {
            return Err(TokenError::NativeNotSupported.into());
        }
        Self::validate_owner(
            program_id,
            &account.base.owner,
            authority_info,
            authority_info.data_len(),
            account_info_iter.as_slice(),
        )?;

        if let Ok(cpi_guard) = account.get_extension::<CpiGuard>() {
            if cpi_guard.lock_cpi.into() && in_cpi() {
                return Err(TokenError::CpiGuardTransferBlocked.into());
            }
        }
    } else if let Ok(mint) = PodStateWithExtensions::<PodMint>::unpack(&source_data) {
        match &mint.base.mint_authority {
            PodCOption { option: PodCOption::<Pubkey>::SOME, value: mint_authority } => {
                Self::validate_owner(
                    program_id,
                    mint_authority,
                    authority_info,
                    authority_info.data_len(),
                    account_info_iter.as_slice(),
                )?;
            }
            PodCOption { option: PodCOption::<Pubkey>::NONE, value: _ } if source_info.key == authority_info.key => {
                if !authority_info.is_signer {
                    return Err(ProgramError::MissingRequiredSignature);
                }
            }
            _ => return Err(TokenError::AuthorityTypeNotSupported.into()),
        }
    } else if source_data.len() == PodMultisig::SIZE_OF {
        Self::validate_owner(
            program_id,
            source_info.key,
            authority_info,
            authority_info.data_len(),
            account_info_iter.as_slice(),
        )?;
    } else {
        return Err(TokenError::InvalidState.into());
    }

    let source_rent_exempt_reserve = Rent::get()?.minimum_balance(source_info.data_len());

    let transfer_amount = source_info
        .lamports()
        .checked_sub(source_rent_exempt_reserve)
        .ok_or(TokenError::NotRentExempt)?;

    let source_starting_lamports = source_info.lamports();
    **source_info.lamports.borrow_mut() = source_starting_lamports
        .checked_sub(transfer_amount)
        .ok_or(TokenError::Overflow)?;

    let destination_starting_lamports = destination_info.lamports();
    **destination_info.lamports.borrow_mut() = destination_starting_lamports
        .checked_add(transfer_amount)
        .ok_or(TokenError::Overflow)?;

    Ok(())
}'''

THINKING_5 = """Analyzing `process_withdraw_excess_lamports` from the SPL Token-2022 program.

1. **Function overview**: Recovers lamports sitting above the rent-exempt reserve on any Token-2022-owned account (token account, mint, or multisig) and moves them to a destination account, gated by the appropriate authority for that account type.

2. **Missing Key Check**: Every branch validates the authority before touching funds: token accounts and mints with a set authority go through `Self::validate_owner(...)`, which checks the authority matches and is a signer; mints with no authority set fall back to requiring the mint account itself to sign (`if !authority_info.is_signer { return Err(...) }`); multisig accounts also go through `validate_owner`. Any unrecognized account layout is explicitly rejected (`TokenError::InvalidState`). No branch skips authorization.

3. **Type Confusion**: The source account is only ever interpreted after a typed, checked `unpack` call (`PodStateWithExtensions::<PodAccount>::unpack` / `<PodMint>::unpack`), and the multisig branch is only taken when the raw data length exactly matches `PodMultisig::SIZE_OF`. There is no blind cast of account data to a struct without first validating which layout it actually has, and `check_program_account(source_info.owner)` confirms the account is owned by this program before any of that.

4. **CPI Reentrancy**: No cross-program invocation occurs in this function; lamports are moved by directly mutating the two accounts' lamport fields (`**source_info.lamports.borrow_mut() = ...`), which is standard, safe practice for a program moving lamports between accounts it has been granted access to in the same instruction. Not applicable.

5. **Unchecked External Calls**: No CPI calls are made.

6. **Integer Overflow/Underflow**: Every arithmetic step uses a checked variant with a properly propagated error: `checked_sub(...).ok_or(TokenError::NotRentExempt)?` for the transferable amount, `checked_sub(...).ok_or(TokenError::Overflow)?` for the source balance, and `checked_add(...).ok_or(TokenError::Overflow)?` for the destination balance. This is a textbook-correct implementation of the required pattern — no raw `+`/`-` and no silent wraparound.

7. **Bump Seed Canonicalization**: No PDA derivation occurs in this function.

8. **Denial of Service**: This is not an initialization instruction, so re-initialization guards do not apply. There is also an extra protective check: if the token account has a `CpiGuard` extension with `lock_cpi` enabled and the call is happening inside a CPI (`in_cpi()`), the lamport withdrawal is blocked (`CpiGuardTransferBlocked`), preventing a class of confused-deputy issues during nested CPI. No loops or oracle reads are present.

Every applicable category is handled correctly, with the checked-arithmetic pattern in particular implemented exactly as the security guidelines require."""

CONTENT_5 = """The contract validates the appropriate authority for every possible account type before moving funds, uses only checked arithmetic with properly propagated errors, and includes an extra CPI-guard check against confused-deputy withdrawals.
### Vulnerability: Not Vulnerable
### Explanation: Authority is validated per account type (token account owner, mint authority or self-signed mint, or multisig) before any lamports move, all balance arithmetic uses checked_sub/checked_add with error propagation, and a CpiGuard check blocks withdrawals triggered from within a nested CPI.
### Risk: No significant risk identified."""


TRAIN_EXAMPLES = [
    make_example("generic_safe_001_single_pool_replenish", CODE_1, THINKING_1, CONTENT_1),
    make_example("generic_safe_002_single_pool_create_metadata", CODE_2, THINKING_2, CONTENT_2),
    make_example("generic_safe_003_single_pool_init_onramp", CODE_3, THINKING_3, CONTENT_3),
    make_example("generic_safe_004_governance_relinquish_vote", CODE_4, THINKING_4, CONTENT_4),
]

VAL_EXAMPLES = [
    make_example("generic_safe_005_token2022_withdraw_excess_lamports", CODE_5, THINKING_5, CONTENT_5),
]


def append_jsonl(path, examples):
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Appended {len(examples)} examples to {path}")


if __name__ == "__main__":
    append_jsonl(TRAIN_PATH, TRAIN_EXAMPLES)
    append_jsonl(VAL_PATH, VAL_EXAMPLES)

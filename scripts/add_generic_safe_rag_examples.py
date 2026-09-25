"""
Third batch: RAG-formatted "generic safe" training examples.

Diagnosis (see plan): the FT model's true-negative rate improved for the
no-RAG configuration as generic-safe examples were added (v1->v2->v3: 19->
22->24) but got slightly WORSE for the +RAG configuration (18->16->15).
Root cause: ALL 234 existing training examples (including the 10 generic-
safe ones added so far) use the plain no-RAG prompt. The model never saw a
training example where a RAG checklist was presented and correctly
dismissed. This script reuses the same 10 already-audited real, safe
contracts, but now wraps them in the exact RAG prompt/checklist format used
at inference time (see src/evaluation/utils/prompts.py and rag.py), with
CoT reasoning that explicitly evaluates each checklist item before
concluding "Not Vulnerable".

The checklist for each example uses REAL entries from
data/knowledge_base/rag_contracts.jsonl (the same knowledge base used by
the actual RAG pipeline), so the format is byte-for-byte representative of
what the model will see at inference time.
"""

import json
import os

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TRAIN_PATH = os.path.join(REPO_DIR, "data", "training", "dataset_think_format.jsonl")
VAL_PATH = os.path.join(REPO_DIR, "data", "training", "validation_dataset.jsonl")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")

with open(os.path.join(KB_DIR, "vulnerability_info.json"), encoding="utf-8") as f:
    VULN_INFO = json.load(f)

RAG_CONTRACTS = [json.loads(l) for l in open(os.path.join(KB_DIR, "rag_contracts.jsonl"), encoding="utf-8")]
BY_VULN = {}
for c in RAG_CONTRACTS:
    BY_VULN.setdefault(c["vulnerability"], []).append(c)

DEVELOPER_PROMPT_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
Your task is to analyze Rust code precisely and systematically to identify security vulnerabilities, leveraging additional contextual information retrieved via RAG.

Required behavior:
- Perform a structured reasoning phase inside <think>...</think> tags.
  - For each RAG-suggested vulnerability, check whether the code enforces the required security invariant.
  - Also remain open to detecting other vulnerabilities not present in the RAG list.
- After </think>, provide the final judgment inside <final>...</final> tags with this structure:
  - A short summary sentence.
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it could be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

If no vulnerability is found, write "### Vulnerability: Not Vulnerable".
Do NOT include any extra commentary outside these tags."""

USER_TEMPLATE_RAG = """Here is a vulnerability checklist (from RAG) with priority guidance.
Use it to steer your audit, but also look for issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following Solana smart contract:

Contract Code:

```rust
{code}
```"""


def build_checklist(entries):
    """entries: list of (vuln_key, score) tuples, matching build_rag_checklist's format."""
    checklist = ""
    for i, (vuln_key, score) in enumerate(entries):
        info = VULN_INFO[vuln_key]
        example = BY_VULN[vuln_key][0]
        if i < 3:
            checklist += f"""
    {i+1}. **{info['name']}** ({score})
        - Description: {info['description']}
        - Preconditions: {info['precondition']}
        - Security check: {info['security_check']}
        - Example vulnerable pattern:
          ```rust
    {example['vulnerable_part'].strip()}
          ```
"""
        else:
            checklist += f"\n    {i+1}. **{info['name']}** ({score})\n"
    return checklist


def make_example(example_id, code, checklist_entries, thinking, content):
    checklist = build_checklist(checklist_entries)
    return {
        "id": example_id,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT_RAG},
            {"role": "user", "content": USER_TEMPLATE_RAG.format(checklist=checklist, code=code)},
            {"role": "assistant", "thinking": thinking, "content": content},
        ],
    }


# ---------------------------------------------------------------------------
# Reuse the same 10 real, audited contracts from batch 1 and batch 2.
# ---------------------------------------------------------------------------
from add_generic_safe_examples import CODE_1, CODE_2, CODE_3, CODE_4, CODE_5  # noqa: E402
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "batch2", os.path.join(os.path.dirname(__file__), "add_generic_safe_examples_batch2.py")
)
_batch2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch2)
CODE_6, CODE_7, CODE_8, CODE_9, CODE_10 = (
    _batch2.CODE_6, _batch2.CODE_7, _batch2.CODE_8, _batch2.CODE_9, _batch2.CODE_10,
)


def rag_thinking(checklist_items_analysis, other_categories_note):
    """checklist_items_analysis: list of (vuln_display_name, verdict_text) for the 2-3 checklisted items."""
    lines = ["The RAG checklist suggests checking the following vulnerabilities first; I evaluate each against the code, then broaden to the remaining categories.\n"]
    for name, verdict in checklist_items_analysis:
        lines.append(f"**Checking {name} (from checklist)**: {verdict}\n")
    lines.append(other_categories_note)
    return "\n".join(lines)


CONTENT_TEMPLATE = """The RAG checklist items were each evaluated against the code and none apply; the remaining vulnerability categories were also checked with no findings.
### Vulnerability: Not Vulnerable
### Explanation: {explanation}
### Risk: No significant risk identified."""


EXAMPLES = [
    dict(
        id="generic_safe_rag_001_single_pool_replenish",
        code=CODE_1,
        checklist=[("bump_seed", "0.7842"), ("cpi_reentrancy", "0.7513"), ("missing_key_check", "0.6910")],
        thinking=rag_thinking(
            [
                ("Bump Seed Canonicalization", "The checklist's vulnerable pattern shows a user-supplied bump seed used directly with `create_program_address`. In this function, `stake_authority_bump_seed` instead comes from `check_pool_stake_authority_address`, the program's own canonical-derivation helper — the bump is derived and validated by the program itself, not accepted from caller input. This checklist item does not apply."),
                ("CPI Reentrancy", "The checklist's pattern shows a CPI happening before a critical state update on funds. Here, the `invoke_signed` calls (delegate_stake, move_stake, move_lamports) target the native Stake program, which has no mechanism to call back into this program, so there is no reentrancy surface regardless of ordering. Not applicable."),
                ("Missing Key Check", "The checklist's pattern shows a privileged function with no signer/authority check. This instruction is intentionally permissionless maintenance (rebalancing pool stake) with no privileged action being bypassed — all signer PDAs used for CPI are program-derived and validated, not caller-supplied. Not applicable."),
            ],
            "Continuing to the remaining categories: Type Confusion (pool and vote accounts validated via typed helpers `SinglePool::from_account_info`/`check_vote_account`), Unchecked External Calls (every invoke_signed uses `?`), Integer Overflow (only `saturating_sub` is used, never raw arithmetic), and Denial of Service (this is a repeatable maintenance call, not an initializer, with no unbounded loops or oracle reads). No issues found in any category.",
        ),
        explanation="All three checklist items (bump seed, CPI reentrancy, missing key check) rely on patterns absent here: bump seeds come from canonical derivation helpers, the CPI target is the non-reentrant native Stake program, and the instruction is intentionally permissionless with no privileged action skipped.",
    ),
    dict(
        id="generic_safe_rag_002_single_pool_create_metadata",
        code=CODE_2,
        checklist=[("missing_key_check", "0.7205"), ("type_confusion", "0.6844"), ("bump_seed", "0.6512")],
        thinking=rag_thinking(
            [
                ("Missing Key Check", "The checklist's vulnerable pattern shows a privileged function callable without verifying the caller. Here, `payer_info.is_signer` is explicitly checked and returns an error if false, and the mint/MPL authorities are program-derived PDAs (not caller-supplied keys) validated via `check_pool_mint_authority_address`/`check_pool_mpl_authority_address`. Not applicable."),
                ("Type Confusion", "The checklist's pattern shows an account used without verifying its owner or address. Here, every account (`pool_info`, `pool_mint_info`, `system_program_info`, `payer_info`, `mpl_token_metadata_program_info`, `metadata_info`) is checked against its expected owner or canonical derived address before use. Not applicable."),
                ("Bump Seed Canonicalization", "Both `mint_authority_bump_seed` and `mpl_authority_bump_seed` are returned by the program's own canonical `check_pool_*_address` helpers, not supplied by the caller. Not applicable."),
            ],
            "Continuing to the remaining categories: CPI Reentrancy and Unchecked External Calls (the single invoke_signed call is the last statement and propagates its result with `?`; no state is left stale before it), Integer Overflow (no arithmetic is performed at all), and Denial of Service (a duplicate call would fail inside the Metaplex program's own guard against an already-initialized metadata PDA). No issues found.",
        ),
        explanation="Payer signature and all account addresses/owners are explicitly validated before the CPI, both PDA bumps come from canonical derivation helpers, and no arithmetic or re-initializable state exists in this function.",
    ),
    dict(
        id="generic_safe_rag_003_single_pool_init_onramp",
        code=CODE_3,
        checklist=[("dos", "0.7677"), ("bump_seed", "0.7021"), ("unchecked_calls", "0.6433")],
        thinking=rag_thinking(
            [
                ("Denial of Service", "The checklist's vulnerable pattern shows an initialization function missing an is_initialized guard, allowing re-initialization to overwrite existing state. Here there is no explicit flag check, but `system_instruction::allocate` requires the target PDA to still be owned by the System Program with zero-length data; once this instruction succeeds, `pool_onramp_info` becomes owned by the Stake program, so a second `allocate` on the same PDA fails at the runtime level before any state could be overwritten. This is a structural, not a flag-based, re-initialization guard, and it satisfies the same intent."),
                ("Bump Seed Canonicalization", "`onramp_bump_seed` and `stake_authority_bump_seed` both come from the canonical `check_pool_*_address` helpers, used exactly as returned. Not applicable."),
                ("Unchecked External Calls", "All three `invoke_signed` calls (`allocate`, `assign`, `initialize_checked`) are followed by `?`. Not applicable."),
            ],
            "Continuing to the remaining categories: Missing Key Check (this is intentionally permissionless setup, no privileged authority bypassed), Type Confusion (all accounts checked against canonical derivations or sysvar-typed accessors), CPI Reentrancy (account is being freshly created, no pre-existing state to exploit via reentry), and Integer Overflow (only a rent-balance comparison, no raw arithmetic). No issues found.",
        ),
        explanation="Re-initialization is prevented structurally by System Program account-ownership semantics rather than an explicit flag, all PDA bumps are canonical, and every CPI result is propagated with `?`.",
    ),
    dict(
        id="generic_safe_rag_004_governance_relinquish_vote",
        code=CODE_4,
        checklist=[("missing_key_check", "0.7390"), ("integer_overflow", "0.6981"), ("type_confusion", "0.6520")],
        thinking=rag_thinking(
            [
                ("Missing Key Check", "The checklist's vulnerable pattern shows a privileged state change with no signer check. While the proposal is actively being voted on, this code requires `token_owner_record_data.assert_token_owner_or_delegate_is_signer(governance_authority_info)?` before changing vote weights. After the proposal is decided, anyone may prune the now-inert vote record — an intentional, documented, harmless cleanup with no financial or voting-power impact, not a missing check."),
                ("Integer Overflow/Underflow", "The checklist's vulnerable pattern shows raw subtraction that can silently underflow. Here every decrement uses `checked_sub(...)`, and the trailing `.unwrap()` causes a panic (safe abort) rather than a silent wraparound if the checked operation would underflow — this matches the intent of the check even though a custom error would be more descriptive."),
                ("Type Confusion", "The checklist's vulnerable pattern shows an account used without owner/discriminator validation. Here, every account (`realm`, `governance`, `proposal`, `token_owner_record`, `vote_record`) is loaded through a dedicated typed getter that cross-validates ownership and relationships before returning deserialized data. Not applicable."),
            ],
            "Continuing to the remaining categories: CPI Reentrancy and Unchecked External Calls (no CPI is performed anywhere in this function), Bump Seed Canonicalization (no PDA derivation occurs here), and Denial of Service (the only loop is bounded by the fixed, small number of proposal options, not attacker-controlled). No issues found.",
        ),
        explanation="Vote withdrawal while active requires the token owner or delegate to sign, post-decision cleanup is intentionally permissionless and inert, all arithmetic uses checked_sub, and every account is loaded through cross-validating typed getters.",
    ),
    dict(
        id="generic_safe_rag_005_token2022_withdraw_excess_lamports",
        code=CODE_5,
        checklist=[("integer_overflow", "0.8103"), ("missing_key_check", "0.7256"), ("type_confusion", "0.6877")],
        thinking=rag_thinking(
            [
                ("Integer Overflow/Underflow", "The checklist's vulnerable pattern shows raw arithmetic on lamport balances. Here every step uses a checked variant with a properly propagated error: `checked_sub(...).ok_or(TokenError::NotRentExempt)?` for the transferable amount, `checked_sub(...).ok_or(TokenError::Overflow)?` for the source balance, and `checked_add(...).ok_or(TokenError::Overflow)?` for the destination — a textbook-correct implementation, not a raw `+`/`-`."),
                ("Missing Key Check", "The checklist's vulnerable pattern shows a fund-moving operation without authority validation. Here, every possible account type branch (token account, mint with authority, mint without authority requiring self-signature, multisig) explicitly validates the authority via `Self::validate_owner(...)` or an explicit signer check before any lamports move. Not applicable."),
                ("Type Confusion", "The checklist's vulnerable pattern shows blind casting of account data. Here, the source account is only interpreted after a typed, checked `unpack` call (`PodStateWithExtensions::<PodAccount>::unpack` / `<PodMint>::unpack`), with an explicit `TokenError::InvalidState` rejection for anything that doesn't match a recognized layout. Not applicable."),
            ],
            "Continuing to the remaining categories: CPI Reentrancy and Unchecked External Calls (no CPI occurs; lamports are moved by directly mutating the two accounts' own lamport fields), Bump Seed Canonicalization (no PDA derivation here), and Denial of Service (not an initializer; an extra CpiGuard check even blocks withdrawal during a nested CPI). No issues found.",
        ),
        explanation="All balance arithmetic uses checked_sub/checked_add with propagated errors, authority is validated per account-type branch before any funds move, and every account is type-checked via a validated unpack before use.",
    ),
    dict(
        id="generic_safe_rag_006_stakepool_remove_validator",
        code=CODE_6,
        checklist=[("missing_key_check", "0.7534"), ("cpi_reentrancy", "0.6822"), ("bump_seed", "0.6390")],
        thinking=rag_thinking(
            [
                ("Missing Key Check", "The checklist's vulnerable pattern shows a privileged operation missing an authority check. Here, `stake_pool.check_authority_withdraw(...)` and `stake_pool.check_staker(staker_info)?` both validate the caller's authority before any state change. Not applicable."),
                ("CPI Reentrancy", "The checklist's vulnerable pattern shows state mutated after an exploitable CPI. Here, `Self::stake_deactivate(...)` invokes the native Stake program, which cannot call back into this program — there is no reentrancy surface regardless of the ordering of the local `validator_stake_info.status` update relative to the CPI, and no funds are transferred in this function."),
                ("Bump Seed Canonicalization", "`stake_pool.stake_withdraw_bump_seed` is a bump stored on-chain at pool creation time, not freshly supplied by the caller for this instruction — it matches the accepted 'verified against a stored value' pattern. `check_validator_stake_address`/`check_transient_stake_address` further re-derive and check the stake account addresses independently."),
            ],
            "Continuing to the remaining categories: Type Confusion (stake pool and validator list both go through owner-check plus is_valid()/header.is_valid() discriminator checks), Unchecked External Calls (both stake_deactivate calls use `?`), Integer Overflow (only epoch/equality comparisons, no arithmetic), and Denial of Service (an explicit status guard rejects removing an already-removed validator twice). No issues found.",
        ),
        explanation="Caller authority is validated via check_authority_withdraw/check_staker, the CPI target is the non-reentrant native Stake program, and the stake-withdraw bump is a stored, previously-validated value rather than fresh caller input.",
    ),
    dict(
        id="generic_safe_rag_007_stakepool_cleanup_removed",
        code=CODE_7,
        checklist=[("dos", "0.7048"), ("missing_key_check", "0.6512"), ("type_confusion", "0.6205")],
        thinking=rag_thinking(
            [
                ("Denial of Service", "The checklist's vulnerable pattern shows unbounded growth or missing cleanup guards leading to resource exhaustion. This function is itself the cleanup mechanism: `validator_list.retain(...)` removes already-dead entries, and the operation is bounded by the pool's existing (bounded) validator list, not unbounded attacker input. It actively prevents the DoS class it was checked against rather than exhibiting it."),
                ("Missing Key Check", "The checklist's vulnerable pattern shows a privileged action with no authority check. This is a harmless, permissionless bookkeeping operation by design — it only deletes already-removed entries and clears stale preferred-validator pointers, with no fund movement or privileged state change requiring authorization."),
                ("Type Confusion", "The checklist's vulnerable pattern shows account data trusted without validation. Both `stake_pool_info` and `validator_list_info` go through `check_account_owner` plus `is_valid()`/`header.is_valid()` discriminator checks before use."),
            ],
            "Continuing to the remaining categories: CPI Reentrancy and Unchecked External Calls (no CPI occurs anywhere in this function), Bump Seed Canonicalization (no PDA derivation here), and Integer Overflow (no arithmetic is performed at all). No issues found.",
        ),
        explanation="This instruction is itself a DoS-prevention cleanup mechanism operating over a bounded list, requires no privileged authority since it only removes already-dead data, and both accounts are owner/discriminator-checked before use.",
    ),
    dict(
        id="generic_safe_rag_008_lending_init_obligation",
        code=CODE_8,
        checklist=[("dos", "0.7891"), ("missing_key_check", "0.7102"), ("type_confusion", "0.6688")],
        thinking=rag_thinking(
            [
                ("Denial of Service", "The checklist's vulnerable pattern shows an initializer missing is_initialized and rent-exemption checks. Here, both are explicitly present: `assert_rent_exempt(rent, obligation_info)?` and `assert_uninitialized::<Obligation>(obligation_info)?` — the two specific mitigations the checklist calls for are implemented directly."),
                ("Missing Key Check", "The checklist's vulnerable pattern shows initialization without verifying the claimed owner. Here, `if !obligation_owner_info.is_signer { return Err(InvalidSigner) }` explicitly requires the claimed owner to sign, preventing anyone else from creating an obligation with a mismatched owner."),
                ("Type Confusion", "The checklist's vulnerable pattern shows accounts trusted without owner/type validation. Here, `obligation_info.owner != program_id` and `lending_market_info.owner != program_id` are both explicitly checked, `LendingMarket::unpack(...)` is a typed deserialization, and the token program is cross-checked against the value stored in the lending market."),
            ],
            "Continuing to the remaining categories: CPI Reentrancy and Unchecked External Calls (no CPI occurs in this function), Bump Seed Canonicalization (the obligation account is a separately-created keypair account, not derived here), and Integer Overflow (only struct field assignment with empty vectors, no arithmetic). No issues found.",
        ),
        explanation="Both DoS mitigations (rent-exemption and uninitialized-state checks) are implemented directly, the claimed owner must sign, and both input accounts are explicitly owner- and type-checked before use.",
    ),
    dict(
        id="generic_safe_rag_009_lending_deposit_obligation_collateral",
        code=CODE_9,
        checklist=[("unchecked_calls", "0.7266"), ("cpi_reentrancy", "0.7012"), ("integer_overflow", "0.6544")],
        thinking=rag_thinking(
            [
                ("Unchecked External Calls", "The checklist's vulnerable pattern shows a CPI result discarded with `let _ =`. Here, `spl_token_transfer(...)?` propagates its result with `?`."),
                ("CPI Reentrancy", "The checklist's vulnerable pattern shows a CPI performed before local state is updated, allowing stale-state exploitation. Here the order is reversed and correct: `obligation.find_or_add_collateral_to_deposits(...).deposit(...)`, `obligation.last_update.mark_stale()`, and `Obligation::pack(...)` (writing the obligation back to its account) all happen BEFORE the `spl_token_transfer` CPI at the end — exactly the checks-effects-interactions ordering the check requires."),
                ("Integer Overflow/Underflow", "The checklist's vulnerable pattern shows raw arithmetic on token amounts. `collateral_amount == 0` is explicitly rejected up front, and no raw `+`/`-`/`*` operator appears directly in this function on the collateral amount (the actual increment happens inside the `.deposit(...)` helper, not shown here, so this assessment is based on the absence of any raw arithmetic in the visible code plus the explicit zero-amount guard)."),
            ],
            "Continuing to the remaining categories: Missing Key Check (obligation ownership is cross-checked against the provided owner account, which must also sign), Type Confusion (every account -- lending market, reserve, obligation -- is unpacked with a typed helper and owner-checked, with cross-references between them also validated), Bump Seed Canonicalization (no PDA derivation occurs here), and Denial of Service (stale reserve data is explicitly rejected via `is_stale(clock.slot)`). No issues found.",
        ),
        explanation="The CPI result is propagated with `?`, local obligation state is committed before the token-transfer CPI (correct checks-effects-interactions order), and no raw arithmetic appears in this function beyond an explicit zero-amount guard.",
    ),
    dict(
        id="generic_safe_rag_010_ata_recover_nested",
        code=CODE_10,
        checklist=[("type_confusion", "0.7622"), ("bump_seed", "0.7134"), ("unchecked_calls", "0.6601")],
        thinking=rag_thinking(
            [
                ("Type Confusion", "The checklist's vulnerable pattern shows an account substituted without validating its derived address or ownership. Here, all three token accounts (owner ATA, nested ATA, destination ATA) are validated by independently re-deriving their expected addresses via `get_associated_token_address_and_bump_seed_internal` and rejecting any mismatch, plus ownership-by-token-program and on-chain owner/hierarchy checks before any `StateWithExtensions` unpack."),
                ("Bump Seed Canonicalization", "`get_associated_token_address_and_bump_seed_internal` is the canonical ATA derivation function (the associated-token-account equivalent of `find_program_address`), and the returned `bump_seed` is used exactly as computed, never overridden by caller input."),
                ("Unchecked External Calls", "Both `invoke_signed` calls propagate their result — the first via `?`, and the second as the function's own tail-position return value, which is not discarded."),
            ],
            "Continuing to the remaining categories: Missing Key Check (the wallet owner must sign, checked explicitly), CPI Reentrancy (the transfer CPI empties the nested account's balance before the close CPI, the only sound order for this operation, and neither CPI target can call back into this program in a way that would matter here), and Integer Overflow (amount and decimals are read from account data and passed through unmodified, no arithmetic performed). No issues found.",
        ),
        explanation="Every associated-token-account address involved is independently re-derived and validated against canonical seeds before use, the wallet must sign, and both CPI results are propagated rather than discarded.",
    ),
]


def build():
    train_examples = []
    val_examples = []
    for i, ex in enumerate(EXAMPLES):
        content = CONTENT_TEMPLATE.format(explanation=ex["explanation"])
        example = make_example(ex["id"], ex["code"], ex["checklist"], ex["thinking"], content)
        if i == len(EXAMPLES) - 1:
            val_examples.append(example)
        else:
            train_examples.append(example)
    return train_examples, val_examples


def append_jsonl(path, examples):
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Appended {len(examples)} examples to {path}")


if __name__ == "__main__":
    train_examples, val_examples = build()
    append_jsonl(TRAIN_PATH, train_examples)
    append_jsonl(VAL_PATH, val_examples)

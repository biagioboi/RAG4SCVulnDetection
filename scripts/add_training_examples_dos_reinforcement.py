"""
Targeted DoS training reinforcement, driven by failure analysis of the v11
model on the 97-contract expanded test set: DoS recall was 0/6 (no-RAG) and
1/6 (RAG k=2) -- the weakest category by far. Inspecting the raw responses
showed the model's own <think> reasoning frequently identified the correct
underlying mechanism (e.g. "the data_is_empty() reinitialization guard has
been removed, so this instruction can be replayed") but then labeled the
verdict as Unchecked External Calls or Type Confusion instead of Denial of
Service -- a labeling/taxonomy problem compounded by the RAG checklist
consistently ranking DoS near the bottom (a retrieval-quality issue already
diagnosed and only partially addressed this session).

Source: governance/program/src/processor/process_set_token_owner_record_lock.rs
(never used anywhere in training or the test set -- verified by grep).
SAFE (real): rejects the lock outright if it is already expired
(`lock.is_expired(clock.unix_timestamp)`) before storing it -- a staleness
check, matching vulnerability_info.json's DoS definition ("oracle/price data
includes staleness checks with maximum age thresholds"). VULNERABLE: removes
the expiry check, letting an already-expired lock be persisted, which then
keeps blocking token-owner-record operations indefinitely (the whole point
of a lock) since nothing ever purges it.
"""

import json
import os

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
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


CODE_DOS_SAFE = """pub fn process_set_token_owner_record_lock(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    lock_id: u8,
    expiry: Option<UnixTimestamp>,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let realm_config_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let token_owner_record_lock_authority_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;

    let rent = Rent::get()?;
    let clock = Clock::get()?;

    if !token_owner_record_lock_authority_info.is_signer {
        return Err(GovernanceError::TokenOwnerRecordLockAuthorityMustSign.into());
    }

    let token_owner_record_lock = TokenOwnerRecordLock {
        lock_id,
        authority: *token_owner_record_lock_authority_info.key,
        expiry,
    };

    // Reject the lock outright if it is already expired, instead of storing
    // stale lock data that would never get cleaned up.
    if token_owner_record_lock.is_expired(clock.unix_timestamp) {
        return Err(GovernanceError::ExpiredTokenOwnerRecordLock.into());
    }

    let realm_data = get_realm_data(program_id, realm_info)?;
    let realm_config_data =
        get_realm_config_data_for_realm(program_id, realm_config_info, realm_info.key)?;

    let mut token_owner_record_data = get_token_owner_record_data_for_realm(
        program_id,
        token_owner_record_info,
        &realm_config_data.realm,
    )?;

    if !realm_config_data
        .get_token_config(&realm_data, &token_owner_record_data.governing_token_mint)?
        .lock_authorities
        .contains(token_owner_record_lock_authority_info.key)
    {
        return Err(GovernanceError::InvalidTokenOwnerRecordLockAuthority.into());
    }

    token_owner_record_data.remove_expired_locks(clock.unix_timestamp);
    token_owner_record_data.upsert_lock(token_owner_record_lock);

    token_owner_record_data.serialize_with_resize(
        token_owner_record_info,
        payer_info,
        system_info,
        &rent,
    )?;

    Ok(())
}"""

CODE_DOS_VULN = """pub fn process_set_token_owner_record_lock(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    lock_id: u8,
    expiry: Option<UnixTimestamp>,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let realm_config_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let token_owner_record_lock_authority_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;

    let rent = Rent::get()?;
    let clock = Clock::get()?;

    if !token_owner_record_lock_authority_info.is_signer {
        return Err(GovernanceError::TokenOwnerRecordLockAuthorityMustSign.into());
    }

    let token_owner_record_lock = TokenOwnerRecordLock {
        lock_id,
        authority: *token_owner_record_lock_authority_info.key,
        expiry,
    };

    // No check that the lock is already expired before storing it.

    let realm_data = get_realm_data(program_id, realm_info)?;
    let realm_config_data =
        get_realm_config_data_for_realm(program_id, realm_config_info, realm_info.key)?;

    let mut token_owner_record_data = get_token_owner_record_data_for_realm(
        program_id,
        token_owner_record_info,
        &realm_config_data.realm,
    )?;

    if !realm_config_data
        .get_token_config(&realm_data, &token_owner_record_data.governing_token_mint)?
        .lock_authorities
        .contains(token_owner_record_lock_authority_info.key)
    {
        return Err(GovernanceError::InvalidTokenOwnerRecordLockAuthority.into());
    }

    token_owner_record_data.upsert_lock(token_owner_record_lock);

    token_owner_record_data.serialize_with_resize(
        token_owner_record_info,
        payer_info,
        system_info,
        &rent,
    )?;

    Ok(())
}"""

THINKING_DOS_VULN = """Analyzing the Solana smart contract function `process_set_token_owner_record_lock`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for unbounded state and staleness guards.

2. **Expected security checks for V10 (Denial of Service)**:
   - Check 1: initialization functions reject re-initialization
   - Check 2: accounts verify rent exemption status
   - Check 3: oracle/price/time-based data includes staleness checks with maximum age thresholds
   - Check 4: loops/collections are bounded

3. **Code inspection**: Examining the implementation of `process_set_token_owner_record_lock`:
   - The reference implementation rejects a lock outright if `token_owner_record_lock.is_expired(clock.unix_timestamp)` before it is ever stored. Here that check is absent: an authority can set a lock with an `expiry` timestamp that is already in the past, and the code stores it anyway via `upsert_lock`.
   - Note this is distinct from a Missing Key Check or Type Confusion issue: the signer and lock-authority checks are both present and correct; the missing piece is specifically the time-based staleness guard on the lock's own data.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing staleness validation: an authority (malicious or careless) can persist an already-expired lock, and since nothing else in this instruction purges expired locks, the record stays encumbered by a lock that serves no purpose except to block legitimate token-owner-record operations that check for active locks, until some other instruction happens to call remove_expired_locks.

5. **Severity assessment**: MEDIUM - This vulnerability can be directly exploited by a malicious actor with access to a lock authority."""

CONTENT_DOS_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V10 - Denial of Service (DoS)
### Explanation: The lock's expiry is never checked before storage, so an already-expired lock can be persisted via upsert_lock. This leaves the token owner record blocked by a stale lock that this instruction never cleans up.
### Risk: MEDIUM - This vulnerability can be exploited to disrupt normal contract operation."""

THINKING_DOS_SAFE = """Analyzing the Solana smart contract function `process_set_token_owner_record_lock`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for unbounded state and staleness guards.

2. **Expected security checks for V10 (Denial of Service)**:
   - Check 1: initialization functions reject re-initialization
   - Check 2: accounts verify rent exemption status
   - Check 3: oracle/price/time-based data includes staleness checks with maximum age thresholds
   - Check 4: loops/collections are bounded

3. **Code inspection**: Examining the implementation of `process_set_token_owner_record_lock`:
   - Before the new lock is stored, `token_owner_record_lock.is_expired(clock.unix_timestamp)` is checked and the instruction is rejected if the lock would already be expired. Existing expired locks are also proactively purged via `remove_expired_locks` on every call.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V10 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract validates lock staleness appropriately."""

CONTENT_DOS_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: New locks are rejected if already expired, and existing expired locks are purged on every call via remove_expired_locks.
### Risk: No significant risk identified."""


def main():
    train_samples = [
        sample("dos_vulnerable_215_gov_set_token_owner_record_lock", CODE_DOS_VULN, THINKING_DOS_VULN, CONTENT_DOS_VULN),
        sample("dos_safe_215_gov_set_token_owner_record_lock", CODE_DOS_SAFE, THINKING_DOS_SAFE, CONTENT_DOS_SAFE),
    ]
    with open(TRAIN_PATH, "a", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Appended {len(train_samples)} samples to {TRAIN_PATH}")


if __name__ == "__main__":
    main()

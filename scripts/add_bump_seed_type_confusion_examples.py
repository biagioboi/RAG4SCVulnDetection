"""
Add new real-code SAFE/VULNERABLE pairs for the two weakest-recall categories
found in the k-sweep ablation (v7/v7b): Bump Seed Canonicalization and Type
Confusion. Both categories already had 16/16 vulnerable examples in training
(same count as the strong categories), so the fix is not "more of the same"
but new, previously-unused real functions -- to give the model more distinct
manifestations of each pattern to generalize from.

Methodology (same as dataset_construction/dataset_batch*/README.md):
  - SAFE: real function, extracted and simplified for length, security
    pattern documented inline.
  - VULNERABLE: "synthesized vulnerable variant" -- same function with the
    specific security check swapped for the unsafe pattern (user-supplied
    bump instead of find_program_address; raw try_from_slice instead of the
    type-checked accessor), same technique as batch1 sample 12
    (process_initialize / Bump Seed) and the V4 samples throughout.

Source: solana-labs/solana-program-library/governance/program/src/processor/
  - process_create_token_owner_record.rs  (Bump Seed)
  - process_create_native_treasury.rs      (Bump Seed, held out for validation)
  - process_sign_off_proposal.rs           (Type Confusion)
  - process_finalize_vote.rs               (Type Confusion)
None of these 4 functions were previously used anywhere in the dataset
(verified by grep against dataset_final.json and dataset_think_format.jsonl).
process_relinquish_vote was deliberately NOT reused here even though it also
fits Type Confusion, because it is already in training as a SAFE
(Not Vulnerable) generic_safe example -- reusing it for a modified vulnerable
variant would put two labels on near-identical code.
"""

import json
import os

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
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
# Pair 1: process_create_token_owner_record -- Bump Seed
# Source: governance/program/src/processor/process_create_token_owner_record.rs
# ============================================================

CODE_1_SAFE = """pub fn process_create_token_owner_record(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governing_token_owner_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governing_token_mint_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent = Rent::get()?;

    let realm_data = get_realm_data(program_id, realm_info)?;
    realm_data.assert_is_valid_governing_token_mint(governing_token_mint_info.key)?;

    if !token_owner_record_info.data_is_empty() {
        return Err(GovernanceError::TokenOwnerRecordAlreadyExists.into());
    }

    let token_owner_record_data = TokenOwnerRecordV2 {
        account_type: GovernanceAccountType::TokenOwnerRecordV2,
        realm: *realm_info.key,
        governing_token_owner: *governing_token_owner_info.key,
        governing_token_deposit_amount: 0,
        governing_token_mint: *governing_token_mint_info.key,
        governance_delegate: None,
        unrelinquished_votes_count: 0,
        outstanding_proposal_count: 0,
        version: TOKEN_OWNER_RECORD_LAYOUT_VERSION,
        reserved: [0; 6],
        reserved_v2: [0; 124],
        locks: vec![],
    };

    // create_and_serialize_account_signed derives the PDA internally via
    // Pubkey::find_program_address(seeds, program_id), verifies the result
    // matches token_owner_record_info.key, and signs the creation CPI with
    // the canonical bump it returns.
    create_and_serialize_account_signed(
        payer_info,
        token_owner_record_info,
        &token_owner_record_data,
        &get_token_owner_record_address_seeds(
            realm_info.key,
            governing_token_mint_info.key,
            governing_token_owner_info.key,
        ),
        program_id,
        system_info,
        &rent,
        0,
    )
}"""

CODE_1_VULN = """pub fn process_create_token_owner_record(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    bump_seed: u8,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governing_token_owner_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governing_token_mint_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent = Rent::get()?;

    let realm_data = get_realm_data(program_id, realm_info)?;
    realm_data.assert_is_valid_governing_token_mint(governing_token_mint_info.key)?;

    if !token_owner_record_info.data_is_empty() {
        return Err(GovernanceError::TokenOwnerRecordAlreadyExists.into());
    }

    // Caller supplies the bump seed directly instead of it being derived
    // canonically via find_program_address.
    let seeds = get_token_owner_record_address_seeds(
        realm_info.key,
        governing_token_mint_info.key,
        governing_token_owner_info.key,
    );
    let mut seeds_with_bump: Vec<&[u8]> = seeds.to_vec();
    let bump = [bump_seed];
    seeds_with_bump.push(&bump);
    let expected_address = Pubkey::create_program_address(&seeds_with_bump, program_id)?;
    if expected_address != *token_owner_record_info.key {
        return Err(ProgramError::InvalidSeeds);
    }

    let token_owner_record_data = TokenOwnerRecordV2 {
        account_type: GovernanceAccountType::TokenOwnerRecordV2,
        realm: *realm_info.key,
        governing_token_owner: *governing_token_owner_info.key,
        governing_token_deposit_amount: 0,
        governing_token_mint: *governing_token_mint_info.key,
        governance_delegate: None,
        unrelinquished_votes_count: 0,
        outstanding_proposal_count: 0,
        version: TOKEN_OWNER_RECORD_LAYOUT_VERSION,
        reserved: [0; 6],
        reserved_v2: [0; 124],
        locks: vec![],
    };

    create_and_serialize_account_with_bump(
        payer_info,
        token_owner_record_info,
        &token_owner_record_data,
        &seeds_with_bump,
        program_id,
        system_info,
        &rent,
        0,
    )
}"""

THINKING_1_SAFE = """Analyzing the Solana smart contract function `process_create_token_owner_record`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for PDA derivation and bump seed handling.

2. **Expected security checks for V9 (Bump Seed)**:
   - Check 1: use of find_program_address (returns canonical bump)
   - Check 2: bump seed stored and verified on subsequent calls
   - Check 3: no raw create_program_address without bump validation

3. **Code inspection**: Examining the implementation of `process_create_token_owner_record`:
   - The TokenOwnerRecord PDA is created via `create_and_serialize_account_signed`, which derives the address internally with `find_program_address(seeds, program_id)`, confirms the result matches `token_owner_record_info.key`, and only then signs the account-creation CPI with the canonical bump. No bump seed is ever accepted from instruction data.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V9 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles PDA derivation and bump seed canonicalization appropriately."""

CONTENT_1_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The TokenOwnerRecord PDA is derived via find_program_address inside create_and_serialize_account_signed, which enforces the canonical bump before creating and signing for the account.
### Risk: No significant risk identified."""

THINKING_1_VULN = """Analyzing the Solana smart contract function `process_create_token_owner_record`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for PDA derivation and bump seed handling.

2. **Expected security checks for V9 (Bump Seed)**:
   - Check 1: use of find_program_address (returns canonical bump)
   - Check 2: bump seed stored and verified on subsequent calls
   - Check 3: no raw create_program_address without bump validation

3. **Code inspection**: Examining the implementation of `process_create_token_owner_record`:
   - The function takes `bump_seed` directly as an instruction argument and calls `Pubkey::create_program_address(&seeds_with_bump, program_id)` with it, instead of deriving the canonical bump via `find_program_address`. It only checks that the resulting address matches the provided account key, not that the bump itself is canonical.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing or inadequate PDA derivation and bump seed handling. An attacker can supply a different, non-canonical bump value that still satisfies `create_program_address` for a distinct off-curve address, allowing creation of multiple, colliding "TokenOwnerRecord" PDAs for the same (realm, mint, owner) tuple.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_1_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V9 - Bump Seed Canonicalization
### Explanation: TokenOwnerRecord PDA derived with a caller-supplied bump via create_program_address instead of find_program_address. Attacker can create a non-canonical duplicate TokenOwnerRecord for the same realm/mint/owner tuple.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""


# ============================================================
# Pair 2: process_create_native_treasury -- Bump Seed (validation set)
# Source: governance/program/src/processor/process_create_native_treasury.rs
# ============================================================

CODE_2_SAFE = """pub fn process_create_native_treasury(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let native_treasury_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;

    let rent = Rent::get()?;

    assert_is_valid_governance(program_id, governance_info)?;

    let native_treasury_data = NativeTreasury {};

    // create_and_serialize_account_with_owner_signed derives the treasury PDA
    // via find_program_address(seeds, program_id) and signs the creation CPI
    // with the canonical bump it returns.
    create_and_serialize_account_with_owner_signed(
        payer_info,
        native_treasury_info,
        &native_treasury_data,
        &get_native_treasury_address_seeds(governance_info.key),
        program_id,
        &system_program::id(),
        system_info,
        &rent,
        0,
    )?;

    Ok(())
}"""

CODE_2_VULN = """pub fn process_create_native_treasury(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    bump_seed: u8,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let native_treasury_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;

    let rent = Rent::get()?;

    assert_is_valid_governance(program_id, governance_info)?;

    // Bump seed comes from instruction data instead of find_program_address,
    // so the canonical-address guarantee is gone.
    let seeds = get_native_treasury_address_seeds(governance_info.key);
    let mut seeds_with_bump: Vec<&[u8]> = seeds.to_vec();
    let bump = [bump_seed];
    seeds_with_bump.push(&bump);
    let expected_address = Pubkey::create_program_address(&seeds_with_bump, program_id)?;
    if expected_address != *native_treasury_info.key {
        return Err(ProgramError::InvalidSeeds);
    }

    let native_treasury_data = NativeTreasury {};
    create_and_serialize_account_with_bump(
        payer_info,
        native_treasury_info,
        &native_treasury_data,
        &seeds_with_bump,
        program_id,
        system_info,
        &rent,
        0,
    )?;

    Ok(())
}"""

THINKING_2_SAFE = """Analyzing the Solana smart contract function `process_create_native_treasury`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for PDA derivation and bump seed handling.

2. **Expected security checks for V9 (Bump Seed)**:
   - Check 1: use of find_program_address (returns canonical bump)
   - Check 2: bump seed stored and verified on subsequent calls
   - Check 3: no raw create_program_address without bump validation

3. **Code inspection**: Examining the implementation of `process_create_native_treasury`:
   - The NativeTreasury PDA is created via `create_and_serialize_account_with_owner_signed`, which derives the address internally with `find_program_address(seeds, program_id)` before signing the creation CPI with the canonical bump. No bump is accepted from the caller.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V9 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles PDA derivation and bump seed canonicalization appropriately."""

CONTENT_2_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The NativeTreasury PDA is derived via find_program_address inside create_and_serialize_account_with_owner_signed, which enforces the canonical bump before creating and signing for the account.
### Risk: No significant risk identified."""

THINKING_2_VULN = """Analyzing the Solana smart contract function `process_create_native_treasury`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for PDA derivation and bump seed handling.

2. **Expected security checks for V9 (Bump Seed)**:
   - Check 1: use of find_program_address (returns canonical bump)
   - Check 2: bump seed stored and verified on subsequent calls
   - Check 3: no raw create_program_address without bump validation

3. **Code inspection**: Examining the implementation of `process_create_native_treasury`:
   - The function accepts `bump_seed` as an instruction argument and calls `Pubkey::create_program_address` with it directly instead of deriving the canonical bump via `find_program_address`. Only the resulting address is checked against the account key, not the canonicality of the bump itself.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing or inadequate PDA derivation and bump seed handling. An attacker can pick a non-canonical bump that produces a distinct, still-valid off-curve address, creating a duplicate "treasury" PDA for the same governance account that the program will still treat as authentic.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_2_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V9 - Bump Seed Canonicalization
### Explanation: NativeTreasury PDA derived with a caller-supplied bump via create_program_address instead of find_program_address. Attacker can create a non-canonical duplicate treasury PDA for the same governance account.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""


# ============================================================
# Pair 3: process_sign_off_proposal -- Type Confusion
# Source: governance/program/src/processor/process_sign_off_proposal.rs
# ============================================================

CODE_3_SAFE = """pub fn process_sign_off_proposal(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let signatory_info = next_account_info(account_info_iter)?;
    let signatory_record_info = next_account_info(account_info_iter)?;

    let clock = Clock::get()?;

    assert_is_valid_realm(program_id, realm_info)?;
    let governance_data = get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;
    let mut proposal_data = get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_sign_off()?;

    // get_signatory_record_data_for_seeds checks the account's account_type
    // tag equals SignatoryRecordV2/V1 AND re-derives its PDA from
    // ['governance', proposal, signatory] to confirm signatory_record_info is
    // genuinely the record for THIS proposal and THIS signatory.
    let mut signatory_record_data = get_signatory_record_data_for_seeds(
        program_id,
        signatory_record_info,
        proposal_info.key,
        signatory_info.key,
    )?;
    signatory_record_data.assert_can_sign_off(signatory_info)?;

    signatory_record_data.signed_off = true;
    signatory_record_data.serialize(&mut signatory_record_info.data.borrow_mut()[..])?;

    if proposal_data.signatories_signed_off_count == 0 {
        proposal_data.signing_off_at = Some(clock.unix_timestamp);
        proposal_data.state = ProposalState::SigningOff;
    }
    proposal_data.signatories_signed_off_count =
        proposal_data.signatories_signed_off_count.checked_add(1).unwrap();

    if proposal_data.signatories_signed_off_count == proposal_data.signatories_count {
        proposal_data.voting_at = Some(clock.unix_timestamp);
        proposal_data.voting_at_slot = Some(clock.slot);
        proposal_data.state = ProposalState::Voting;
    }

    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;
    Ok(())
}"""

CODE_3_VULN = """pub fn process_sign_off_proposal(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let signatory_info = next_account_info(account_info_iter)?;
    let signatory_record_info = next_account_info(account_info_iter)?;

    let clock = Clock::get()?;

    assert_is_valid_realm(program_id, realm_info)?;
    let governance_data = get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;
    let mut proposal_data = get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_sign_off()?;

    // Deserializes the raw account bytes directly instead of calling
    // get_signatory_record_data_for_seeds, so neither the account_type tag
    // nor the (proposal, signatory) PDA binding is checked.
    let mut signatory_record_data =
        SignatoryRecordV2::try_from_slice(&signatory_record_info.data.borrow())?;
    signatory_record_data.assert_can_sign_off(signatory_info)?;

    signatory_record_data.signed_off = true;
    signatory_record_data.serialize(&mut signatory_record_info.data.borrow_mut()[..])?;

    if proposal_data.signatories_signed_off_count == 0 {
        proposal_data.signing_off_at = Some(clock.unix_timestamp);
        proposal_data.state = ProposalState::SigningOff;
    }
    proposal_data.signatories_signed_off_count =
        proposal_data.signatories_signed_off_count.checked_add(1).unwrap();

    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;
    Ok(())
}"""

THINKING_3_SAFE = """Analyzing the Solana smart contract function `process_sign_off_proposal`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for input validation and account type verification.

2. **Expected security checks for V4 (Type Confusion)**:
   - Check 1: account type/discriminator validation
   - Check 2: account ownership verification (e.g., account.owner == expected_program)
   - Check 3: data deserialization with proper type checks

3. **Code inspection**: Examining the implementation of `process_sign_off_proposal`:
   - The signatory record is loaded through `get_signatory_record_data_for_seeds`, which checks the account's `account_type` discriminator matches SignatoryRecordV2/V1 AND re-derives the PDA from `['governance', proposal, signatory]` to confirm the account genuinely belongs to this proposal and this signatory before any field is trusted.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V4 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles input validation and account type verification appropriately."""

CONTENT_3_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The signatory record is loaded via get_signatory_record_data_for_seeds, which validates the account_type discriminator and re-derives the PDA from the proposal and signatory keys before use.
### Risk: No significant risk identified."""

THINKING_3_VULN = """Analyzing the Solana smart contract function `process_sign_off_proposal`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for input validation and account type verification.

2. **Expected security checks for V4 (Type Confusion)**:
   - Check 1: account type/discriminator validation
   - Check 2: account ownership verification (e.g., account.owner == expected_program)
   - Check 3: data deserialization with proper type checks

3. **Code inspection**: Examining the implementation of `process_sign_off_proposal`:
   - `signatory_record_info` is deserialized directly with `SignatoryRecordV2::try_from_slice(...)`, bypassing `get_signatory_record_data_for_seeds`. Neither the `account_type` discriminator nor the PDA binding to this specific (proposal, signatory) pair is checked.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing or inadequate input validation and account type verification. An attacker can pass any account whose byte layout matches SignatoryRecordV2 -- e.g. a signatory record belonging to a different proposal, or one they fully control -- and have the program treat it as their authentic signatory record, forging a sign-off.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_3_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V4 - Type Confusion (Input Validation)
### Explanation: signatory_record_info is deserialized with a raw try_from_slice instead of the type-checked get_signatory_record_data_for_seeds accessor, skipping both the account_type discriminator check and the (proposal, signatory) PDA binding. An attacker can substitute a foreign or forged account to fake a sign-off.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""


# ============================================================
# Pair 4: process_finalize_vote -- Type Confusion
# Source: governance/program/src/processor/process_finalize_vote.rs
# ============================================================

CODE_4_SAFE = """pub fn process_finalize_vote(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let proposal_owner_record_info = next_account_info(account_info_iter)?;
    let governing_token_mint_info = next_account_info(account_info_iter)?;

    let clock = Clock::get()?;

    // Every account is loaded through a typed getter that checks the
    // account_type discriminator AND that it belongs to this realm/
    // governance/governing-mint combination before returning deserialized
    // state.
    let realm_data = get_realm_data_for_governing_token_mint(
        program_id, realm_info, governing_token_mint_info.key,
    )?;
    let mut governance_data =
        get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;
    let mut proposal_data = get_proposal_data_for_governance_and_governing_mint(
        program_id, proposal_info, governance_info.key, governing_token_mint_info.key,
    )?;
    let mut proposal_owner_record_data = get_token_owner_record_data_for_proposal_owner(
        program_id, proposal_owner_record_info, &proposal_data.token_owner_record,
    )?;

    proposal_data.finalize_vote(clock.unix_timestamp)?;
    proposal_owner_record_data.decrease_outstanding_proposal_count();

    proposal_owner_record_data.serialize(&mut proposal_owner_record_info.data.borrow_mut()[..])?;
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;
    governance_data.active_proposal_count = governance_data.active_proposal_count.saturating_sub(1);
    governance_data.serialize(&mut governance_info.data.borrow_mut()[..])?;

    Ok(())
}"""

CODE_4_VULN = """pub fn process_finalize_vote(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let proposal_owner_record_info = next_account_info(account_info_iter)?;
    let governing_token_mint_info = next_account_info(account_info_iter)?;

    let clock = Clock::get()?;

    let realm_data = get_realm_data_for_governing_token_mint(
        program_id, realm_info, governing_token_mint_info.key,
    )?;
    let mut governance_data =
        get_governance_data_for_realm(program_id, governance_info, realm_info.key)?;

    // Proposal and owner-record state are deserialized directly from raw
    // bytes, skipping the typed getters' checks that account_type matches
    // and that the accounts actually belong to this governance/mint pair.
    let mut proposal_data = ProposalV2::try_from_slice(&proposal_info.data.borrow())?;
    let mut proposal_owner_record_data =
        TokenOwnerRecordV2::try_from_slice(&proposal_owner_record_info.data.borrow())?;

    proposal_data.finalize_vote(clock.unix_timestamp)?;
    proposal_owner_record_data.decrease_outstanding_proposal_count();

    proposal_owner_record_data.serialize(&mut proposal_owner_record_info.data.borrow_mut()[..])?;
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;
    governance_data.active_proposal_count = governance_data.active_proposal_count.saturating_sub(1);
    governance_data.serialize(&mut governance_info.data.borrow_mut()[..])?;

    Ok(())
}"""

THINKING_4_SAFE = """Analyzing the Solana smart contract function `process_finalize_vote`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for input validation and account type verification.

2. **Expected security checks for V4 (Type Confusion)**:
   - Check 1: account type/discriminator validation
   - Check 2: account ownership verification (e.g., account.owner == expected_program)
   - Check 3: data deserialization with proper type checks

3. **Code inspection**: Examining the implementation of `process_finalize_vote`:
   - `proposal_data` is loaded via `get_proposal_data_for_governance_and_governing_mint` and `proposal_owner_record_data` via `get_token_owner_record_data_for_proposal_owner` -- both check the account_type discriminator and cross-validate the accounts belong to this specific governance/proposal/mint combination before any field is trusted.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V4 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles input validation and account type verification appropriately."""

CONTENT_4_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: proposal_data and proposal_owner_record_data are loaded through typed getters that validate the account_type discriminator and confirm each account belongs to this governance/proposal/mint combination.
### Risk: No significant risk identified."""

THINKING_4_VULN = """Analyzing the Solana smart contract function `process_finalize_vote`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for input validation and account type verification.

2. **Expected security checks for V4 (Type Confusion)**:
   - Check 1: account type/discriminator validation
   - Check 2: account ownership verification (e.g., account.owner == expected_program)
   - Check 3: data deserialization with proper type checks

3. **Code inspection**: Examining the implementation of `process_finalize_vote`:
   - `proposal_data` and `proposal_owner_record_data` are deserialized directly with `try_from_slice`, bypassing `get_proposal_data_for_governance_and_governing_mint` and `get_token_owner_record_data_for_proposal_owner`. Neither the account_type discriminator nor the binding to this specific governance/mint/proposal combination is checked.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing or inadequate input validation and account type verification. An attacker can substitute a proposal or token-owner-record account from a completely different realm/governance that happens to share the same byte layout, causing the wrong proposal to be finalized or the wrong owner's outstanding-proposal count to be decremented.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_4_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V4 - Type Confusion (Input Validation)
### Explanation: proposal_info and proposal_owner_record_info are deserialized with raw try_from_slice instead of the type-checked accessors, skipping the account_type discriminator check and the binding to this governance/mint/proposal combination. An attacker can substitute a foreign account with matching layout to finalize the wrong proposal or corrupt an unrelated owner's record.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""


def main():
    train_samples = [
        sample("bump_seed_vulnerable_211_gov_create_token_owner_record", CODE_1_VULN, THINKING_1_VULN, CONTENT_1_VULN),
        sample("bump_seed_safe_211_gov_create_token_owner_record", CODE_1_SAFE, THINKING_1_SAFE, CONTENT_1_SAFE),
        sample("type_confusion_vulnerable_211_gov_sign_off_proposal", CODE_3_VULN, THINKING_3_VULN, CONTENT_3_VULN),
        sample("type_confusion_safe_211_gov_sign_off_proposal", CODE_3_SAFE, THINKING_3_SAFE, CONTENT_3_SAFE),
        sample("type_confusion_vulnerable_212_gov_finalize_vote", CODE_4_VULN, THINKING_4_VULN, CONTENT_4_VULN),
        sample("type_confusion_safe_212_gov_finalize_vote", CODE_4_SAFE, THINKING_4_SAFE, CONTENT_4_SAFE),
    ]
    val_samples = [
        sample("bump_seed_vulnerable_212_gov_create_native_treasury", CODE_2_VULN, THINKING_2_VULN, CONTENT_2_VULN),
        sample("bump_seed_safe_212_gov_create_native_treasury", CODE_2_SAFE, THINKING_2_SAFE, CONTENT_2_SAFE),
    ]

    with open(TRAIN_PATH, "a", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(VAL_PATH, "a", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Appended {len(train_samples)} samples to {TRAIN_PATH}")
    print(f"Appended {len(val_samples)} samples to {VAL_PATH}")


if __name__ == "__main__":
    main()

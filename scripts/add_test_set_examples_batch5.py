"""
Fifth batch of test-set expansion, continuing to grow the still-thin
categories (Integer Overflow, Bump Seed, DoS, Type Confusion) plus one more
Missing Key Check example, from governance functions never used anywhere in
training or the existing test set (verified by grep). One function
(process_insert_transaction) is legitimately reused for two different
pairs/categories with two different single-line modifications -- same
precedent as batch2's stake-pool process_withdraw_obligation_collateral
(Bump Seed + CPI Reentrancy from one real function).

Sources:
  - process_insert_transaction.rs: checked_add(1).unwrap() on
    transactions_next_index/transactions_count (Integer Overflow pair), AND
    the data_is_empty() reinitialization guard on proposal_transaction_info
    (DoS pair).
  - process_create_governance.rs: PDA created via
    create_and_serialize_account_signed with seeds from
    get_governance_address_seeds (canonical, internally find_program_address)
    (Bump Seed pair).
  - process_refund_proposal_deposit.rs: deposit state loaded via the typed
    get_proposal_deposit_data_for_proposal_and_deposit_payer getter, which
    cross-validates the deposit belongs to this exact (proposal, payer) pair
    (Type Confusion pair).
  - process_set_governance_config.rs: requires governance_info.is_signer
    (the governance PDA can only reconfigure itself via a proposal-executed
    CPI, never directly) (Missing Key Check pair).
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

# ============================================================
# 1. Integer Overflow -- governance/process_insert_transaction.rs
# ============================================================

CODE_IO_SAFE = """pub fn process_insert_transaction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    option_index: u8,
    instruction_index: u16,
    instructions: Vec<InstructionData>,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governance_authority_info = next_account_info(account_info_iter)?;
    let proposal_transaction_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent_sysvar_info = next_account_info(account_info_iter)?;
    let rent = &Rent::from_account_info(rent_sysvar_info)?;

    if !proposal_transaction_info.data_is_empty() {
        return Err(GovernanceError::TransactionAlreadyExists.into());
    }

    let _governance_data = get_governance_data(program_id, governance_info)?;
    let mut proposal_data =
        get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_edit_instructions()?;

    let token_owner_record_data = get_token_owner_record_data_for_proposal_owner(
        program_id,
        token_owner_record_info,
        &proposal_data.token_owner_record,
    )?;
    token_owner_record_data.assert_token_owner_or_delegate_is_signer(governance_authority_info)?;

    let option = &mut proposal_data.options[option_index as usize];

    match instruction_index.cmp(&option.transactions_next_index) {
        Ordering::Greater => return Err(GovernanceError::InvalidTransactionIndex.into()),
        Ordering::Equal => {
            // checked_add rejects the instruction outright once the counter
            // hits u16::MAX instead of silently wrapping back to 0.
            option.transactions_next_index = option.transactions_next_index.checked_add(1).unwrap();
        }
        Ordering::Less => {}
    }

    option.transactions_count = option.transactions_count.checked_add(1).unwrap();
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    let proposal_transaction_data = ProposalTransactionV2 {
        account_type: GovernanceAccountType::ProposalTransactionV2,
        option_index,
        transaction_index: instruction_index,
        legacy: 0,
        instructions,
        executed_at: None,
        execution_status: TransactionExecutionStatus::None,
        proposal: *proposal_info.key,
        reserved_v2: [0; 8],
    };

    create_and_serialize_account_signed::<ProposalTransactionV2>(
        payer_info,
        proposal_transaction_info,
        &proposal_transaction_data,
        &get_proposal_transaction_address_seeds(
            proposal_info.key,
            &option_index.to_le_bytes(),
            &instruction_index.to_le_bytes(),
        ),
        program_id,
        system_info,
        rent,
        0,
    )?;

    Ok(())
}"""

CODE_IO_VULN = """pub fn process_insert_transaction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    option_index: u8,
    instruction_index: u16,
    instructions: Vec<InstructionData>,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governance_authority_info = next_account_info(account_info_iter)?;
    let proposal_transaction_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent_sysvar_info = next_account_info(account_info_iter)?;
    let rent = &Rent::from_account_info(rent_sysvar_info)?;

    if !proposal_transaction_info.data_is_empty() {
        return Err(GovernanceError::TransactionAlreadyExists.into());
    }

    let _governance_data = get_governance_data(program_id, governance_info)?;
    let mut proposal_data =
        get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_edit_instructions()?;

    let token_owner_record_data = get_token_owner_record_data_for_proposal_owner(
        program_id,
        token_owner_record_info,
        &proposal_data.token_owner_record,
    )?;
    token_owner_record_data.assert_token_owner_or_delegate_is_signer(governance_authority_info)?;

    let option = &mut proposal_data.options[option_index as usize];

    match instruction_index.cmp(&option.transactions_next_index) {
        Ordering::Greater => return Err(GovernanceError::InvalidTransactionIndex.into()),
        Ordering::Equal => {
            // Raw increment instead of checked_add: once the counter reaches
            // u16::MAX it silently wraps back to 0 instead of erroring.
            option.transactions_next_index += 1;
        }
        Ordering::Less => {}
    }

    option.transactions_count += 1;
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    let proposal_transaction_data = ProposalTransactionV2 {
        account_type: GovernanceAccountType::ProposalTransactionV2,
        option_index,
        transaction_index: instruction_index,
        legacy: 0,
        instructions,
        executed_at: None,
        execution_status: TransactionExecutionStatus::None,
        proposal: *proposal_info.key,
        reserved_v2: [0; 8],
    };

    create_and_serialize_account_signed::<ProposalTransactionV2>(
        payer_info,
        proposal_transaction_info,
        &proposal_transaction_data,
        &get_proposal_transaction_address_seeds(
            proposal_info.key,
            &option_index.to_le_bytes(),
            &instruction_index.to_le_bytes(),
        ),
        program_id,
        system_info,
        rent,
        0,
    )?;

    Ok(())
}"""

# ============================================================
# 2. Denial of Service -- same function, reinitialization-guard angle
# ============================================================

CODE_DOS_SAFE = CODE_IO_SAFE  # the data_is_empty() reinit guard is present here too

CODE_DOS_VULN = """pub fn process_insert_transaction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    option_index: u8,
    instruction_index: u16,
    instructions: Vec<InstructionData>,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;
    let proposal_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let governance_authority_info = next_account_info(account_info_iter)?;
    let proposal_transaction_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent_sysvar_info = next_account_info(account_info_iter)?;
    let rent = &Rent::from_account_info(rent_sysvar_info)?;

    // The check that proposal_transaction_info.data_is_empty() (i.e. this
    // slot was not already used) has been removed, so this instruction can
    // be replayed against an existing ProposalTransactionV2 account.
    let _governance_data = get_governance_data(program_id, governance_info)?;
    let mut proposal_data =
        get_proposal_data_for_governance(program_id, proposal_info, governance_info.key)?;
    proposal_data.assert_can_edit_instructions()?;

    let token_owner_record_data = get_token_owner_record_data_for_proposal_owner(
        program_id,
        token_owner_record_info,
        &proposal_data.token_owner_record,
    )?;
    token_owner_record_data.assert_token_owner_or_delegate_is_signer(governance_authority_info)?;

    let option = &mut proposal_data.options[option_index as usize];

    match instruction_index.cmp(&option.transactions_next_index) {
        Ordering::Greater => return Err(GovernanceError::InvalidTransactionIndex.into()),
        Ordering::Equal => {
            option.transactions_next_index = option.transactions_next_index.checked_add(1).unwrap();
        }
        Ordering::Less => {}
    }

    option.transactions_count = option.transactions_count.checked_add(1).unwrap();
    proposal_data.serialize(&mut proposal_info.data.borrow_mut()[..])?;

    let proposal_transaction_data = ProposalTransactionV2 {
        account_type: GovernanceAccountType::ProposalTransactionV2,
        option_index,
        transaction_index: instruction_index,
        legacy: 0,
        instructions,
        executed_at: None,
        execution_status: TransactionExecutionStatus::None,
        proposal: *proposal_info.key,
        reserved_v2: [0; 8],
    };

    create_and_serialize_account_signed::<ProposalTransactionV2>(
        payer_info,
        proposal_transaction_info,
        &proposal_transaction_data,
        &get_proposal_transaction_address_seeds(
            proposal_info.key,
            &option_index.to_le_bytes(),
            &instruction_index.to_le_bytes(),
        ),
        program_id,
        system_info,
        rent,
        0,
    )?;

    Ok(())
}"""

# ============================================================
# 3. Bump Seed Canonicalization -- governance/process_create_governance.rs
# ============================================================

CODE_BUMP_SAFE = """pub fn process_create_governance(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    config: GovernanceConfig,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let governance_seed_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent = Rent::get()?;
    let create_authority_info = next_account_info(account_info_iter)?;

    assert_valid_create_governance_args(program_id, &config, realm_info)?;

    let realm_data = get_realm_data(program_id, realm_info)?;
    realm_data.assert_create_authority_can_create_governance(
        program_id,
        realm_info.key,
        token_owner_record_info,
        create_authority_info,
        account_info_iter,
    )?;

    let governance_data = GovernanceV2 {
        account_type: GovernanceAccountType::GovernanceV2,
        realm: *realm_info.key,
        governance_seed: *governance_seed_info.key,
        config,
        reserved1: 0,
        reserved_v2: Reserved119::default(),
        required_signatories_count: 0,
        active_proposal_count: 0,
    };

    // create_and_serialize_account_signed derives the PDA internally via
    // find_program_address(seeds, program_id) and signs the creation CPI
    // with the canonical bump it returns.
    create_and_serialize_account_signed::<GovernanceV2>(
        payer_info,
        governance_info,
        &governance_data,
        &get_governance_address_seeds(realm_info.key, governance_seed_info.key),
        program_id,
        system_info,
        &rent,
        0,
    )?;

    Ok(())
}"""

CODE_BUMP_VULN = """pub fn process_create_governance(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    config: GovernanceConfig,
    bump_seed: u8,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let realm_info = next_account_info(account_info_iter)?;
    let governance_info = next_account_info(account_info_iter)?;
    let governance_seed_info = next_account_info(account_info_iter)?;
    let token_owner_record_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_info = next_account_info(account_info_iter)?;
    let rent = Rent::get()?;
    let create_authority_info = next_account_info(account_info_iter)?;

    assert_valid_create_governance_args(program_id, &config, realm_info)?;

    let realm_data = get_realm_data(program_id, realm_info)?;
    realm_data.assert_create_authority_can_create_governance(
        program_id,
        realm_info.key,
        token_owner_record_info,
        create_authority_info,
        account_info_iter,
    )?;

    let governance_data = GovernanceV2 {
        account_type: GovernanceAccountType::GovernanceV2,
        realm: *realm_info.key,
        governance_seed: *governance_seed_info.key,
        config,
        reserved1: 0,
        reserved_v2: Reserved119::default(),
        required_signatories_count: 0,
        active_proposal_count: 0,
    };

    // The bump seed comes from instruction data instead of
    // find_program_address, so the canonical-address guarantee is gone.
    let seeds = get_governance_address_seeds(realm_info.key, governance_seed_info.key);
    let mut seeds_with_bump: Vec<&[u8]> = seeds.to_vec();
    let bump = [bump_seed];
    seeds_with_bump.push(&bump);
    let expected_address = Pubkey::create_program_address(&seeds_with_bump, program_id)?;
    if expected_address != *governance_info.key {
        return Err(ProgramError::InvalidSeeds);
    }

    create_and_serialize_account_with_bump::<GovernanceV2>(
        payer_info,
        governance_info,
        &governance_data,
        &seeds_with_bump,
        program_id,
        system_info,
        &rent,
        0,
    )?;

    Ok(())
}"""

# ============================================================
# 4. Type Confusion -- governance/process_refund_proposal_deposit.rs
# ============================================================

CODE_TC_SAFE = """pub fn process_refund_proposal_deposit(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let proposal_info = next_account_info(account_info_iter)?;
    let proposal_deposit_info = next_account_info(account_info_iter)?;
    let proposal_deposit_payer_info = next_account_info(account_info_iter)?;

    let proposal_data = get_proposal_data(program_id, proposal_info)?;
    proposal_data.assert_can_refund_proposal_deposit()?;

    // get_proposal_deposit_data_for_proposal_and_deposit_payer checks the
    // account_type discriminator AND that the deposit record actually
    // belongs to THIS proposal and THIS payer before it can be disposed of.
    let _proposal_deposit_data = get_proposal_deposit_data_for_proposal_and_deposit_payer(
        program_id,
        proposal_deposit_info,
        proposal_info.key,
        proposal_deposit_payer_info.key,
    )?;

    dispose_account(proposal_deposit_info, proposal_deposit_payer_info)?;

    Ok(())
}"""

CODE_TC_VULN = """pub fn process_refund_proposal_deposit(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let proposal_info = next_account_info(account_info_iter)?;
    let proposal_deposit_info = next_account_info(account_info_iter)?;
    let proposal_deposit_payer_info = next_account_info(account_info_iter)?;

    let proposal_data = get_proposal_data(program_id, proposal_info)?;
    proposal_data.assert_can_refund_proposal_deposit()?;

    // Deserializes the deposit record directly from raw bytes, skipping
    // get_proposal_deposit_data_for_proposal_and_deposit_payer, so neither
    // the account_type discriminator nor the binding to this specific
    // (proposal, payer) pair is checked.
    let _proposal_deposit_data =
        ProposalDeposit::try_from_slice(&proposal_deposit_info.data.borrow())?;

    dispose_account(proposal_deposit_info, proposal_deposit_payer_info)?;

    Ok(())
}"""

# ============================================================
# 5. Missing Key Check -- governance/process_set_governance_config.rs
# ============================================================

CODE_MKC_SAFE = """pub fn process_set_governance_config(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    config: GovernanceConfig,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;

    // Only the governance PDA itself, signing via a CPI from an executed
    // proposal, may authorize a change to its own config.
    if !governance_info.is_signer {
        return Err(GovernanceError::GovernancePdaMustSign.into());
    };

    assert_is_valid_governance_config(&config)?;

    let mut governance_data = get_governance_data(program_id, governance_info)?;
    governance_data.config = config;
    governance_data.serialize(&mut governance_info.data.borrow_mut()[..])?;

    Ok(())
}"""

CODE_MKC_VULN = """pub fn process_set_governance_config(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    config: GovernanceConfig,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let governance_info = next_account_info(account_info_iter)?;

    // No check that governance_info is a signer -- anyone can pass the
    // governance account and rewrite its config directly, bypassing the
    // requirement that config changes only happen via an executed proposal.
    assert_is_valid_governance_config(&config)?;

    let mut governance_data = get_governance_data(program_id, governance_info)?;
    governance_data.config = config;
    governance_data.serialize(&mut governance_info.data.borrow_mut()[..])?;

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
        ("integer_overflow", "V8", CODE_IO_VULN, CODE_IO_SAFE),
        ("dos", "V10", CODE_DOS_VULN, CODE_DOS_SAFE),
        ("bump_seed", "V9", CODE_BUMP_VULN, CODE_BUMP_SAFE),
        ("type_confusion", "V4", CODE_TC_VULN, CODE_TC_SAFE),
        ("missing_key_check", "V1", CODE_MKC_VULN, CODE_MKC_SAFE),
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

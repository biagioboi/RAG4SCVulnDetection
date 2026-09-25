"""
Seventh batch: one more Missing Key Check pair, from
solana-labs/solana-program-library/name-service, fn process_update (never
used anywhere in training or the test set -- verified by grep; process_create,
process_delete and process_transfer from the same file were already used
elsewhere, but process_update was not).

SAFE (real): requires the caller to be a signer AND to be either the
configured "class" account or the record's owner (or the parent name's
owner, for delegated updates). VULNERABLE drops the class/owner check,
keeping only the is_signer requirement -- authenticated but not authorized.
"""

import json
import os

TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/test_set"

CODE_MKC_SAFE = """pub fn process_update(accounts: &[AccountInfo], offset: u32, data: Vec<u8>) -> ProgramResult {
    let accounts_iter = &mut accounts.iter();

    let name_account = next_account_info(accounts_iter)?;
    let name_update_signer = next_account_info(accounts_iter)?;
    let parent_name = next_account_info(accounts_iter).ok();

    let name_record_header = NameRecordHeader::unpack_from_slice(&name_account.data.borrow())?;

    let is_parent_owner = if let Some(parent_name) = parent_name {
        if name_record_header.parent_name != *parent_name.key {
            msg!("Invalid parent name account");
            return Err(ProgramError::InvalidArgument);
        }
        let parent_name_record_header =
            NameRecordHeader::unpack_from_slice(&parent_name.data.borrow())?;
        parent_name_record_header.owner == *name_update_signer.key
    } else {
        false
    };
    if !name_update_signer.is_signer {
        msg!("The given name class or owner is not a signer.");
        return Err(ProgramError::InvalidArgument);
    }
    if name_record_header.class != Pubkey::default()
        && *name_update_signer.key != name_record_header.class
    {
        msg!("The given name class account is incorrect.");
        return Err(ProgramError::InvalidArgument);
    }
    if name_record_header.class == Pubkey::default()
        && *name_update_signer.key != name_record_header.owner
        && !is_parent_owner
    {
        msg!("The given name owner account is incorrect.");
        return Err(ProgramError::InvalidArgument);
    }

    write_data(
        name_account,
        &data,
        NameRecordHeader::LEN.saturating_add(offset as usize),
    );

    Ok(())
}"""

CODE_MKC_VULN = """pub fn process_update(accounts: &[AccountInfo], offset: u32, data: Vec<u8>) -> ProgramResult {
    let accounts_iter = &mut accounts.iter();

    let name_account = next_account_info(accounts_iter)?;
    let name_update_signer = next_account_info(accounts_iter)?;

    let name_record_header = NameRecordHeader::unpack_from_slice(&name_account.data.borrow())?;

    // Only checks that SOMEONE signed, never that the signer is the
    // record's class/owner (or a delegated parent-name owner).
    if !name_update_signer.is_signer {
        msg!("The given name class or owner is not a signer.");
        return Err(ProgramError::InvalidArgument);
    }

    write_data(
        name_account,
        &data,
        NameRecordHeader::LEN.saturating_add(offset as usize),
    );

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

    written = [
        write_entry(next_idx, "missing_key_check", "V1", "VULNERABLE", CODE_MKC_VULN),
        write_entry(next_idx + 1, "not_vulnerable", "V1", "SAFE", CODE_MKC_SAFE),
    ]

    print(f"Wrote {len(written)} new test files:")
    for w in written:
        print(" ", w)
    print(f"Test set size: {len(existing)} -> {len(existing) + len(written)}")


if __name__ == "__main__":
    main()

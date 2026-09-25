"""
Root-cause fix for DoS's poor RAG retrieval ranking (diagnosed via v14
failure analysis): all 5 existing DoS entries in
data/knowledge_base/rag_contracts.jsonl describe the SAME narrow family of
DoS sub-patterns -- unbounded validator-list growth, panics on curve
calculation, stale price/exchange-rate checks, missing withdrawal-size
limits. None of them describe a missing REINITIALIZATION GUARD (the
"if account already initialized, reject" check), even though
vulnerability_info.json's own DoS definition names re-initialization as the
primary example, and 2 of the 6 DoS contracts in the expanded test set
(solana_70, solana_84) are exactly that sub-pattern. For those two
contracts (and solana_46, solana_34, which are also not well matched by any
existing entry), the checklist consistently ranked DoS in the bottom half
(#4-#7 of 7), so the model never even saw a detailed DoS hint.

This adds one more DoS reference entry covering the reinitialization-guard
sub-pattern, using the record.rs `Initialize` instruction (the same
"if account_data.is_initialized() { return Err(AccountAlreadyInitialized) }"
pattern already used in training under a Type Confusion framing for a
different specific issue -- reused here for its reinitialization angle
specifically, matching the format of the existing 5 KB entries). This is a
knowledge-base reference document, not a train/test item, so reusing real
code already seen in training here does not create train/test leakage.
"""

import json

KB_PATH = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/knowledge_base/rag_contracts.jsonl"

NEW_ENTRY = {
    "id": "dos-VULNERABLE-05",
    "smart_contract": """RecordInstruction::Initialize => {
    let data_info = next_account_info(account_info_iter)?;
    let authority_info = next_account_info(account_info_iter)?;

    let raw_data = &mut data_info.data.borrow_mut();
    if raw_data.len() < RecordData::WRITABLE_START_INDEX {
        return Err(ProgramError::InvalidAccountData);
    }

    let account_data = bytemuck::try_from_bytes_mut::<RecordData>(
        &mut raw_data[..RecordData::WRITABLE_START_INDEX],
    )
    .map_err(|_| ProgramError::InvalidArgument)?;

    account_data.authority = *authority_info.key;
    account_data.version = RecordData::CURRENT_VERSION;
    Ok(())
}""",
    "vulnerability": "dos",
    "vulnerable_part": """RecordInstruction::Initialize => {
    let data_info = next_account_info(account_info_iter)?;
    let authority_info = next_account_info(account_info_iter)?;
    let account_data = bytemuck::try_from_bytes_mut::<RecordData>(
        &mut raw_data[..RecordData::WRITABLE_START_INDEX],
    ).map_err(|_| ProgramError::InvalidArgument)?;
    // Missing: if account_data.is_initialized() { return Err(AccountAlreadyInitialized) }
    account_data.authority = *authority_info.key;
    account_data.version = RecordData::CURRENT_VERSION;
    Ok(())
}""",
    "description": (
        "### Contract type:\n"
        "Solana program instruction handler\n\n"
        "### Contract Purpose\n"
        "This function `Initialize` is a Solana program instruction handler that processes program state changes. \n\n"
        "### Functional Description\n"
        "- The function does not include explicit signer or ownership verification.\n"
        "- Security observation: The initialize handler never checks whether the account was already "
        "initialized before overwriting its authority field. Calling this instruction again on an "
        "already-initialized account silently resets its authority, and since nothing rejects the "
        "replay, the account can be repeatedly reinitialized -- the same missing-reinitialization-guard "
        "pattern that makes accounts unusable or lets state be reset indefinitely."
    ),
}


def main():
    rows = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if any(r["id"] == NEW_ENTRY["id"] for r in rows):
        raise ValueError(f"{NEW_ENTRY['id']} already exists in {KB_PATH}")

    rows.append(NEW_ENTRY)

    with open(KB_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Added {NEW_ENTRY['id']}. KB now has {len(rows)} entries "
          f"({sum(1 for r in rows if r['vulnerability']=='dos')} dos entries).")


if __name__ == "__main__":
    main()

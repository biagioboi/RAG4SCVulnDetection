"""
Fix the RAG knowledge base descriptions used for embedding-based retrieval
(data/knowledge_base/rag_contracts.jsonl, consumed by search_similar_contracts
in scripts/eval_qwen3_32b_solana*.py).

Root cause found by manually inspecting the two DoS-truth contracts the v11
model mislabeled as Type Confusion (solana_34, solana_70): the RAG checklist
built for both ranked "Type Confusion" or "Bump Seed Canonicalization" as the
#1 (most similar) reference item instead of DoS. Inspecting rag_contracts.jsonl
showed why -- every one of the 37 entries' "description" field (the text
that gets embedded for similarity search) opens with the IDENTICAL templated
boilerplate:

    ### Contract type:
    Solana program instruction handler

    ### Contract Purpose
    This function `X` is a Solana program instruction handler that ...

    ### Functional Description
    - The function does not include explicit signer or ownership verification.
    - Security observation: <the only category-specific content, often
      truncated>

That generic "does not include explicit signer or ownership verification"
line is present verbatim across ALL categories (verified: all 5/5 dos and
5/5 type_confusion entries contain it), so instructor-xl's embedding of the
full description is dominated by shared boilerplate text instead of the one
sentence that actually distinguishes the category. This also explains the
near-uniform (0.925-0.977) similarity scores found earlier this session by
diagnose_rag_similarity.py.

Fix: rebuild each entry's "description" to drop the boilerplate and lead
with the category name plus the (already extracted) category-specific
"Security observation" sentence, which is the only content that varies
between entries.
"""

import json
import re

KB_PATH = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer/data/knowledge_base/rag_contracts.jsonl"

DISPLAY_NAME = {
    "missing_key_check": "Missing Key Check (Access Control)",
    "type_confusion": "Type Confusion (Input Validation)",
    "cpi_reentrancy": "CPI Reentrancy",
    "unchecked_calls": "Unchecked External Calls",
    "integer_overflow": "Integer Overflow/Underflow",
    "bump_seed": "Bump Seed Canonicalization",
    "dos": "Denial of Service (DoS)",
}


def rebuild_description(entry):
    old = entry["description"]

    func_match = re.search(r"This function `([^`]+)`", old)
    func_name = func_match.group(1) if func_match else "this instruction handler"

    obs_match = re.search(r"Security observation:\s*(.+)", old, re.DOTALL)
    if not obs_match:
        raise ValueError(f"No security observation found for {entry['id']}")
    observation = obs_match.group(1).strip()

    display = DISPLAY_NAME.get(entry["vulnerability"], entry["vulnerability"])

    return (
        f"{display} vulnerability in the Solana instruction handler "
        f"`{func_name}`. {observation}"
    )


def main():
    rows = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    changed = 0
    for r in rows:
        new_desc = rebuild_description(r)
        if new_desc != r["description"]:
            r["description"] = new_desc
            changed += 1

    with open(KB_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Rewrote {changed}/{len(rows)} descriptions in {KB_PATH}")
    print("\nSample after rewrite:")
    for r in rows[:2]:
        print(" -", r["id"], "->", r["description"][:160])


if __name__ == "__main__":
    main()

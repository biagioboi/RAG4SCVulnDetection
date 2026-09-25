"""
Shared helper: builds the v6 static RAG checklist -- same static, full
7-category reference as v5, now enriched with ONE fixed, representative
vulnerable-code example per category (the shortest entry in
rag_contracts.jsonl for that category, chosen deterministically, NOT by
similarity to the contract under review).
"""

import json
import os

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")

with open(os.path.join(KB_DIR, "vulnerability_info.json"), encoding="utf-8") as f:
    VULN_INFO = json.load(f)

RAG_CONTRACTS = [json.loads(l) for l in open(os.path.join(KB_DIR, "rag_contracts.jsonl"), encoding="utf-8")]
_BY_VULN = {}
for c in RAG_CONTRACTS:
    _BY_VULN.setdefault(c["vulnerability"], []).append(c)

# Deterministically pick the shortest example per category -- a fixed,
# representative illustration, not a per-contract similarity match.
EXAMPLE_PER_VULN = {
    key: min(entries, key=lambda e: len(e["vulnerable_part"]))
    for key, entries in _BY_VULN.items()
}


def build_static_checklist():
    parts = []
    for i, (key, info) in enumerate(VULN_INFO.items()):
        example = EXAMPLE_PER_VULN.get(key)
        example_block = ""
        if example is not None:
            example_block = (
                f"   - Representative vulnerable pattern:\n"
                f"     ```rust\n     {example['vulnerable_part'].strip()}\n     ```\n"
            )
        parts.append(
            f"\n{i+1}. **{info['name']}**\n"
            f"   - Description: {info['description']}\n"
            f"   - Applies when: {info['precondition']}\n"
            f"   - Security check to verify: {info['security_check']}\n"
            f"{example_block}"
        )
    return "".join(parts)


STATIC_CHECKLIST = build_static_checklist()

if __name__ == "__main__":
    print(STATIC_CHECKLIST)
    print(f"\nTotal length: {len(STATIC_CHECKLIST)} chars")

"""
RAG-formatted "generic safe" training examples, v2 design (static full
checklist, no embedding similarity score -- see plan and conversation for
the redesign rationale: the previous embedding-based top-3 retrieval was
shown to carry no discriminative signal, 0.958 mean similarity on both safe
and vulnerable contracts).

Reuses the SAME 10 real, manually-audited contracts and the SAME analysis
(thinking/content) from add_generic_safe_examples.py and
add_generic_safe_examples_batch2.py -- that reasoning already independently
walks through all 7 categories against the code using the exact security
check criteria, which is precisely the behavior the new RAG system prompt
asks for. Only the prompt wrapper changes: instead of a similarity-scored
3-item checklist, it is now the full static 7-category reference.
"""

import json
import os
import importlib.util

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TRAIN_PATH = os.path.join(REPO_DIR, "data", "training", "dataset_think_format.jsonl")
VAL_PATH = os.path.join(REPO_DIR, "data", "training", "validation_dataset.jsonl")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")

with open(os.path.join(KB_DIR, "vulnerability_info.json"), encoding="utf-8") as f:
    VULN_INFO = json.load(f)


def build_static_checklist():
    parts = []
    for i, (key, info) in enumerate(VULN_INFO.items()):
        parts.append(
            f"\n{i+1}. **{info['name']}**\n"
            f"   - Description: {info['description']}\n"
            f"   - Applies when: {info['precondition']}\n"
            f"   - Security check to verify: {info['security_check']}\n"
        )
    return "".join(parts)


STATIC_CHECKLIST = build_static_checklist()

DEVELOPER_PROMPT_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
Your task is to analyze Rust code precisely and systematically to identify security vulnerabilities.

You are given a reference of all known vulnerability categories below. This is documentation to consult,
NOT a hint about which vulnerabilities are present in this specific contract -- most contracts are safe
with respect to most categories. Determine independently, from the code itself, which categories (if any)
actually apply, and use the reference only to confirm the exact security check for a category you already
suspect from your own analysis.

Required behavior:
- Perform a structured reasoning phase inside <think>...</think> tags.
  - Analyze the code's logic and structure first, on its own terms.
  - Only then check it against the reference categories that seem plausibly relevant, explaining why each
    checked category does or does not apply based on the code.
  - Do not assume a category applies just because it appears in the reference list.
- After </think>, provide the final judgment inside <final>...</final> tags with this structure:
  - A short summary sentence.
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it could be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

If no vulnerability is found, write "### Vulnerability: Not Vulnerable".
Do NOT include any extra commentary outside these tags."""

USER_TEMPLATE_RAG = """### Vulnerability reference (documentation, not a prediction for this contract)
{checklist}

Now perform a detailed security analysis of the following Solana smart contract, deciding independently
which (if any) of the categories above actually apply:

Contract Code:

```rust
{code}
```"""


def make_example(example_id, code, thinking, content):
    return {
        "id": example_id,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT_RAG},
            {"role": "user", "content": USER_TEMPLATE_RAG.format(checklist=STATIC_CHECKLIST, code=code)},
            {"role": "assistant", "thinking": thinking, "content": content},
        ],
    }


# --- Load the 10 already-written, already-audited examples from batch 1 & 2 ---
_spec1 = importlib.util.spec_from_file_location(
    "batch1", os.path.join(os.path.dirname(__file__), "add_generic_safe_examples.py")
)
_b1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(_b1)

_spec2 = importlib.util.spec_from_file_location(
    "batch2", os.path.join(os.path.dirname(__file__), "add_generic_safe_examples_batch2.py")
)
_b2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_b2)

SOURCES = [
    ("generic_safe_ragv2_001_single_pool_replenish", _b1.CODE_1, _b1.THINKING_1, _b1.CONTENT_1),
    ("generic_safe_ragv2_002_single_pool_create_metadata", _b1.CODE_2, _b1.THINKING_2, _b1.CONTENT_2),
    ("generic_safe_ragv2_003_single_pool_init_onramp", _b1.CODE_3, _b1.THINKING_3, _b1.CONTENT_3),
    ("generic_safe_ragv2_004_governance_relinquish_vote", _b1.CODE_4, _b1.THINKING_4, _b1.CONTENT_4),
    ("generic_safe_ragv2_005_token2022_withdraw_excess_lamports", _b1.CODE_5, _b1.THINKING_5, _b1.CONTENT_5),
    ("generic_safe_ragv2_006_stakepool_remove_validator", _b2.CODE_6, _b2.THINKING_6, _b2.CONTENT_6),
    ("generic_safe_ragv2_007_stakepool_cleanup_removed", _b2.CODE_7, _b2.THINKING_7, _b2.CONTENT_7),
    ("generic_safe_ragv2_008_lending_init_obligation", _b2.CODE_8, _b2.THINKING_8, _b2.CONTENT_8),
    ("generic_safe_ragv2_009_lending_deposit_obligation_collateral", _b2.CODE_9, _b2.THINKING_9, _b2.CONTENT_9),
    ("generic_safe_ragv2_010_ata_recover_nested", _b2.CODE_10, _b2.THINKING_10, _b2.CONTENT_10),
]


def build():
    train_examples = []
    val_examples = []
    for i, (eid, code, thinking, content) in enumerate(SOURCES):
        example = make_example(eid, code, thinking, content)
        if i == len(SOURCES) - 1:
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

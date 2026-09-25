"""
v12 phase 3/3: run the actual RAG-audit generation using the checklists
precomputed by phase 2 (Qwen3-Embedding-4B retrieval), score with the fixed
parser, and compare against v11's instructor-xl-based k=2 result (F1=0.611
on the same 81-contract test set) -- the only variable changed across v11
vs v12 is the embedding model used for retrieval; model weights, prompts,
parser, num_detailed=2 and the test set are identical.

No-RAG baseline is not re-run here: it doesn't use retrieval/embeddings at
all, so v11's no-RAG result (F1=0.457) is unaffected by the embedding model
and directly comparable.
"""

import os
import json
import re
import gc
import time
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v11")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "qwen3-32b-v12")

CHECKLISTS_PATH = os.path.join(RESULTS_DIR, "checklists_k2.json")
checklists = json.load(open(CHECKLISTS_PATH, "r", encoding="utf-8"))
print(f"Loaded {len(checklists)} precomputed checklists from {CHECKLISTS_PATH}")

SYSTEM_PROMPT_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
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
Do NOT include any extra commentary outside these tags.""".strip()

USER_PROMPT_RAG = """Here is a vulnerability checklist (from RAG) with priority guidance.
Use it to steer your audit, but also look for issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following Solana smart contract:

Contract Code:

```rust
{code}
```""".strip()

print("Prompts defined.")

# === Parser (unchanged from v7-v11) ===
KNOWN_VULNERABILITIES = [
    "Missing Key Check", "Access Control",
    "Type Confusion", "Input Validation",
    "CPI Reentrancy", "Reentrancy",
    "Unchecked External Calls", "Unchecked Calls",
    "Integer Overflow", "Integer Underflow", "Integer Flow",
    "Bump Seed", "Bump Seed Canonicalization",
    "Denial of Service", "DoS",
    "Not Vulnerable", "No security risk", "No vulnerabilities", "None detected",
]


def parse_response(response):
    match = re.search(r"<final>\s*(.*?)\s*</final>", response, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        think_end = re.search(r"</think>", response, re.IGNORECASE)
        content = response[think_end.end():] if think_end else response
        content = content.strip()

    vuln_lines = re.findall(r"###\s*Vulnerability:\s*(.+)", content, re.IGNORECASE)
    found = []
    for line in vuln_lines:
        for part in re.split(r",|;|\band\b", line, flags=re.IGNORECASE):
            part = part.strip().lower()
            if not part:
                continue
            for v in KNOWN_VULNERABILITIES:
                if v.lower() == part or v.lower() in part:
                    if v not in found:
                        found.append(v)
                    break
    return (True, found) if found else (False, [])


VULN_DISPLAY = {
    "missing_key_check": "Missing Key Check",
    "type_confusion": "Type Confusion",
    "cpi_reentrancy": "CPI Reentrancy",
    "unchecked_calls": "Unchecked External Calls",
    "integer_overflow": "Integer Overflow",
    "bump_seed": "Bump Seed",
    "dos": "Denial of Service",
    "not_vulnerable": "Not Vulnerable",
}

print("Parser ready.")

from unsloth import FastLanguageModel

print("Loading FT v11 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FT_MODEL_DIR, max_seq_length=8192, dtype=None, load_in_4bit=True,
    device_map={"": 0},
)
FastLanguageModel.for_inference(model)
print("Model loaded.")


def generate_response(messages, max_tokens=1024):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.95
    )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


VULNERABILITY_ALIASES = {
    "Missing Key Check": ["missing key check", "access control"],
    "Type Confusion": ["type confusion", "input validation"],
    "CPI Reentrancy": ["cpi reentrancy", "cpi", "reentrancy"],
    "Unchecked External Calls": ["unchecked external calls", "unchecked calls"],
    "Integer Overflow": ["integer overflow", "integer underflow", "integer flow"],
    "Bump Seed": ["bump seed", "bump seed canonicalization"],
    "Denial of Service": ["denial of service", "dos"],
    "Not Vulnerable": ["not vulnerable", "no security risk", "no vulnerabilities", "none detected"],
}

GROUND_TRUTH_MAP = {
    "integer_flow": "Integer Overflow",
    "cpi": "CPI Reentrancy",
}


def normalize_predictions(predictions):
    canonical_set = set()
    for pred in predictions:
        pred_lower = pred.lower().strip()
        for canonical, aliases in VULNERABILITY_ALIASES.items():
            if pred_lower in aliases:
                canonical_set.add(canonical)
                break
    return canonical_set


def compute_metrics(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in results:
        truth = r["ground_truth"]
        predictions = normalize_predictions(r["predictions"])

        if truth in GROUND_TRUTH_MAP:
            truth = GROUND_TRUTH_MAP[truth]

        if truth == "Not Vulnerable":
            if predictions == {"Not Vulnerable"} or len(predictions) == 0:
                tn += 1
            else:
                real_preds = predictions - {"Not Vulnerable"}
                fp += len(real_preds) if real_preds else 0
                for p in real_preds:
                    per_vuln[p]["fp"] += 1
        else:
            if truth in predictions:
                tp += 1
                per_vuln[truth]["tp"] += 1
                extras = predictions - {truth, "Not Vulnerable"}
                fp += len(extras)
                for e in extras:
                    per_vuln[e]["fp"] += 1
            else:
                fn += 1
                per_vuln[truth]["fn"] += 1
                wrong = predictions - {"Not Vulnerable"}
                fp += len(wrong)
                for e in wrong:
                    per_vuln[e]["fp"] += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_vulnerability": dict(per_vuln),
    }


results = []
start = time.time()
for idx, c in enumerate(checklists):
    truth_display = VULN_DISPLAY.get(c["ground_truth"], c["ground_truth"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": USER_PROMPT_RAG.format(checklist=c["checklist"], code=c["code"])},
    ]

    found_vulns = []
    response = ""
    for attempt in range(3):
        response = generate_response(messages)
        success, found_vulns = parse_response(response)
        if success:
            break

    results.append({
        "file": c["file"],
        "ground_truth": truth_display,
        "predictions": found_vulns,
        "checklist": c["checklist"],
        "top_categories": c["top_categories"],
        "response": response,
    })

    pred_str = ", ".join(found_vulns) if found_vulns else "PARSE_FAIL"
    print(f"  [{idx+1}/{len(checklists)}] {c['file']}: truth={truth_display} | pred={pred_str}")

metrics = compute_metrics(results)
path = os.path.join(RESULTS_DIR, "results_ft_rag_k2_qwen3embed.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump({"config": "ft_rag_k2_qwen3embed", "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

print(f"\nDone in {(time.time()-start)/60:.1f} min")
print(f"Results saved to {path}")
print("\n" + "=" * 80)
print("QWEN3-32B-SOLANA-V12 -- Qwen3-Embedding-4B RETRIEVAL (vs v11 instructor-xl)")
print("=" * 80)
print(f"v11 FT+RAG k=2 (instructor-xl): acc=0.6818 prec=0.6875 rec=0.5500 f1=0.6111")
print(f"v12 FT+RAG k=2 (Qwen3-Embed-4B): acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
      f"rec={metrics['recall']:.4f} f1={metrics['f1_score']:.4f}")
print("\nDone.")

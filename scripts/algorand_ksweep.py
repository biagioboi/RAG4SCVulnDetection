"""
Algorand checklist-size ablation, mirroring scripts/*solana*ksweep* methodology:
single variable k, fixed FT checkpoint, fixed 45-contract test set, fixed
parser, single greedy run per k (no sampling variance -- this isolates the
checklist-size variable exactly as done for Solana's k-sweep, Table
tab:solana_ksweep).

k=0 uses the no-RAG prompt/path entirely (no description-generation step).
k=1..8 use the RAG prompt with create_rag_checklist(num_relevant=k); 8 is
the full category count for Algorand (one more than Solana's 7), so k=8
shows every category in full detail.
"""

import os
import json
import time
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
FT_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen3-32B-pyteal-desc", "checkpoint-189")
OUT_DIR = os.path.join(BASE_DIR, "results_multisample", "algorand_ksweep")
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = list(range(0, 9))  # 0 (no RAG) .. 8 (all categories, full detail)

import sys
sys.path.insert(0, BASE_DIR)
from rag import search_similarity, parse_response, create_rag_checklist  # noqa: E402
from prompts import dev_description_prompt, user_description_prompt, user_prompt  # noqa: E402

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

SYSTEM_PROMPT_NO_RAG = """
You are an expert smart contract security auditor specialized in the Algorand blockchain and PyTeal.
Your task is to analyze PyTeal code precisely and systematically to identify security vulnerabilities.

### Required behavior:
- Always perform a structured, explicit reasoning phase first and put it inside the <think> block.
  - In <think>...</think> you must:
    - Summarize the contract's purpose and high-level architecture.
    - Inspect the logic block-by-block (or line-by-line for short snippets).
    - Note any suspicious patterns, missing authorization checks, unsafe transaction handling, or other issues with evidence (point to the code lines/constructs).
    - Use concise, technical language and show the chain of reasoning (why you suspect an issue).

- After </think>, provide the final judgment in the <final>...</final> block using this exact structure:
  - A short summary sentence (one or two lines).
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it can be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

### Constraints:
- Do NOT include any extra commentary, greetings, or meta-text.
- If no vulnerability is found, explicitly write "### Vulnerability: Not Vulnerable" and "### Risk: No significant risk identified".
- The <name> in "### Vulnerability:" MUST be exactly one of: Arbitrary delete, Arbitrary update, Unchecked Asset Close To, Unchecked Close Remainder To, Unchecked Rekey to, Unchecked Transaction Fee, Unchecked Asset Receiver, Unchecked Payment Receiver, Not Vulnerable.
""".strip()
USER_PROMPT_NO_RAG = user_prompt.strip()

SYSTEM_PROMPT_RAG = """
You are an expert smart contract security auditor specialized in the Algorand blockchain and PyTeal.
Your task is to analyze PyTeal code precisely and systematically to identify security vulnerabilities, **leveraging additional contextual information retrieved via RAG** (i.e., probable vulnerabilities provided to you).

### Required behavior:
- Always perform a structured, explicit reasoning phase first and put it inside the <think> block.
  - In <think>...</think> you must:
    - Summarize the contract's purpose and high-level architecture.
    - Inspect the logic block-by-block (or line-by-line for short snippets).
    - For each **RAG-suggested vulnerability**, check whether the code enforces the required security invariant; note any matches or gaps.
    - Also remain open to detecting **other vulnerabilities** not present in the RAG list.
    - Use concise, technical language and show the chain of reasoning (why you suspect an issue, how you trace it to specific code).

- After </think>, provide the final judgment in the <final>…</final> block using this exact structure:
  - A short summary sentence (one or two lines).
  - ### Vulnerability: <name or "None detected">
  - ### Explanation: <concise cause and how it could be exploited>
  - ### Risk: <severity (Critical / High / Medium / Low) and short impact statement>
  - ### Mitigation: <for each identified issue, a concrete fix or recommendation>

### Constraints:
- Do NOT include any extra commentary, greetings, or meta-text.
- If no vulnerability is found, explicitly write "### Vulnerability: Not Vulnerable" and "### Risk: No significant risk identified".
- Do NOT consider logical flaws as vulnerabilities.
""".strip()

USER_PROMPT_RAG = """
Here is a **vulnerability checklist** (from RAG) with your priority guidance. Use it to steer your audit, but also be ready to discover new issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following PyTeal smart contract using the reasoning structure.

Contract Code:

```python
{code}
```
""".strip()

print("Prompts defined.")

from unsloth import FastLanguageModel


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FT_MODEL_DIR, max_seq_length=4096, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(model, tokenizer, messages, seed, max_tokens=1536):
    torch.manual_seed(seed)
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_tokens, do_sample=False,
    )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


MAP_TRUTH_TO_DISPLAY = {
    "transaction_fee": "Unchecked Transaction Fee",
    "close_remainder_to": "Unchecked Close Remainder To",
    "asset_close_to": "Unchecked Asset Close To",
    "rekey_to": "Unchecked Rekey to",
    "arbitrary_delete": "Arbitrary delete",
    "arbitrary_update": "Arbitrary update",
    "Unchecked_Asset_Receiver": "Unchecked Asset Receiver",
    "Unchecked_Payment_Receiver": "Unchecked Payment Receiver",
    "no vuln": "Not Vulnerable",
}
SAFE_LABELS = {"Not Vulnerable", "No security risk", "No vulnerabilities"}


def compute_metrics(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        truth = MAP_TRUTH_TO_DISPLAY.get(r["ground_truth"], r["ground_truth"])
        preds = set(r["predictions"])
        preds_is_safe = len(preds) == 0 or preds.issubset(SAFE_LABELS)
        if truth == "Not Vulnerable":
            if preds_is_safe:
                tn += 1
            else:
                real = preds - SAFE_LABELS
                fp += len(real)
                for p in real:
                    per_vuln[p]["fp"] += 1
        else:
            if truth in preds:
                tp += 1
                per_vuln[truth]["tp"] += 1
                extras = preds - {truth} - SAFE_LABELS
                fp += len(extras)
                for e in extras:
                    per_vuln[e]["fp"] += 1
            else:
                fn += 1
                per_vuln[truth]["fn"] += 1
                wrong = preds - SAFE_LABELS
                fp += len(wrong)
                for e in wrong:
                    per_vuln[e]["fp"] += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    return {
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def run_k(model, tokenizer, k, seed_base=5000):
    results = []
    for idx, test_file in enumerate(test_files):
        with open(test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        code = content["smart_contract"]
        ground_truth = content["vulnerability"]
        seed = seed_base + idx

        if k == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
                {"role": "user", "content": USER_PROMPT_NO_RAG.format(code=code)},
            ]
            response = generate_response(model, tokenizer, messages, seed)
        else:
            desc_messages = [
                {"role": "system", "content": dev_description_prompt},
                {"role": "user", "content": user_description_prompt.format(code=code)},
            ]
            description = generate_response(model, tokenizer, desc_messages, seed, max_tokens=512)
            rag_contracts = search_similarity(description)
            checklist = create_rag_checklist(rag_contracts, num_relevant=k, with_few_shot=True)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": USER_PROMPT_RAG.format(checklist=checklist, code=code)},
            ]
            response = generate_response(model, tokenizer, messages, seed + 1)

        vulnerabilities = parse_response(response)
        results.append({
            "file": os.path.basename(test_file), "ground_truth": ground_truth,
            "predictions": vulnerabilities, "response": response,
        })
        print(f"  [k={k}] [{idx+1}/{len(test_files)}] {os.path.basename(test_file)}: "
              f"truth={ground_truth} | pred={vulnerabilities}")
    return results


print("Loading fine-tuned model (Qwen3-32B-PyTeal-Audit)...")
model_ft, tok_ft = load_model()
print("FT model loaded.\n")

all_rows = []
for k in K_VALUES:
    print("=" * 60)
    print(f"K={k} ({'no RAG' if k == 0 else f'{k} categor' + ('y' if k==1 else 'ies') + ' full detail'})")
    print("=" * 60)
    start = time.time()
    results_k = run_k(model_ft, tok_ft, k)
    metrics_k = compute_metrics(results_k)
    with open(os.path.join(OUT_DIR, f"results_k{k}.json"), "w", encoding="utf-8") as f:
        json.dump({"k": k, "metrics": metrics_k, "details": results_k}, f, indent=2, ensure_ascii=False)
    all_rows.append({"k": k, **metrics_k})
    print(f"k={k} done in {(time.time()-start)/60:.1f} min. "
          f"P={metrics_k['precision']} R={metrics_k['recall']} F1={metrics_k['f1_score']}")

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump({"rows": all_rows}, f, indent=2)

print("\n" + "=" * 80)
print("ALGORAND CHECKLIST-SIZE ABLATION -- SUMMARY")
print("=" * 80)
for row in all_rows:
    print(f"k={row['k']}: P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1_score']:.3f}")
print("\nDone.")

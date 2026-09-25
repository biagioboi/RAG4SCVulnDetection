"""
Sampling-robustness re-run of the flagship Algorand Qwen3-32B-FT + RAG
configuration (originally F1=0.88, results_original_rag_rerun/results_ft_rag.json).

The original eval_algorand_original_rag_rerun.py passed temperature=0.6,
top_p=0.95 to model.generate() WITHOUT do_sample=True. In transformers
4.57.x this makes the sampling arguments silently ignored (falls back to
greedy decoding), so every "run" was actually fully deterministic and a
single point estimate hid all decoding variance.

This script fixes that (do_sample=True, explicit per-call seeding for
reproducibility) and repeats the full 45-contract FT+RAG evaluation
N_SAMPLES times, saving each sample's results/metrics separately, then
reports mean +/- std across samples.
"""

import os
import json
import gc
import time
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
FT_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen3-32B-pyteal-desc", "checkpoint-189")
RESULTS_DIR = os.path.join(BASE_DIR, "results_multisample", "algorand_ft_rag")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES = 5
BASE_SEED = 1000

import sys
sys.path.insert(0, BASE_DIR)
from rag import search_similarity, parse_response, create_rag_checklist  # noqa: E402
from prompts import dev_description_prompt, user_description_prompt  # noqa: E402

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

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

print("Prompts defined (verbatim from test.ipynb).")

from unsloth import FastLanguageModel


def load_finetuned_model():
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
        input_ids=inputs, max_new_tokens=max_tokens,
        do_sample=True, temperature=0.6, top_p=0.95,
    )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


print("Model loading functions ready.")


def evaluate_rag(model, tokenizer, test_files, sample_idx):
    results = []
    for idx, test_file in enumerate(test_files):
        with open(test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        code = content["smart_contract"]
        ground_truth = content["vulnerability"]

        seed_desc = BASE_SEED + sample_idx * 10000 + idx * 2
        seed_audit = seed_desc + 1

        desc_messages = [
            {"role": "system", "content": dev_description_prompt},
            {"role": "user", "content": user_description_prompt.format(code=code)},
        ]
        description = generate_response(model, tokenizer, desc_messages, seed_desc, max_tokens=512)

        rag_contracts = search_similarity(description)
        checklist = create_rag_checklist(rag_contracts, with_few_shot=True)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": USER_PROMPT_RAG.format(checklist=checklist, code=code)},
        ]
        response = generate_response(model, tokenizer, messages, seed_audit)
        vulnerabilities = parse_response(response)

        results.append({
            "file": os.path.basename(test_file),
            "ground_truth": ground_truth,
            "predictions": vulnerabilities,
            "response": response,
        })
        print(f"  [sample {sample_idx}] [{idx+1}/{len(test_files)}] {os.path.basename(test_file)}: "
              f"truth={ground_truth} | pred={vulnerabilities}")
    return results


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
        "per_vulnerability": dict(per_vuln),
    }


print("Loading fine-tuned model (Qwen3-32B-PyTeal-Audit)...")
model_ft, tok_ft = load_finetuned_model()
print("FT model loaded.\n")

all_metrics = []
for s in range(N_SAMPLES):
    print("=" * 60)
    print(f"SAMPLE {s+1}/{N_SAMPLES}: Qwen3-FT + RAG (do_sample=True, seed base={BASE_SEED + s*10000})")
    print("=" * 60)
    start = time.time()
    results_s = evaluate_rag(model_ft, tok_ft, test_files, sample_idx=s)
    metrics_s = compute_metrics(results_s)
    path = os.path.join(RESULTS_DIR, f"results_sample{s}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sample": s, "metrics": metrics_s, "details": results_s}, f, indent=2, ensure_ascii=False)
    all_metrics.append(metrics_s)
    print(f"Sample {s} done in {(time.time()-start)/60:.1f} min. Metrics: {metrics_s['accuracy']=} "
          f"{metrics_s['precision']=} {metrics_s['recall']=} {metrics_s['f1_score']=}")

import statistics as stats

summary = {}
for key in ("accuracy", "precision", "recall", "f1_score"):
    vals = [m[key] for m in all_metrics]
    summary[key] = {
        "values": vals,
        "mean": round(stats.mean(vals), 4),
        "std": round(stats.stdev(vals), 4) if len(vals) > 1 else 0.0,
    }

with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump({"n_samples": N_SAMPLES, "per_sample_metrics": all_metrics, "summary": summary}, f, indent=2)

print("\n" + "=" * 80)
print(f"ALGORAND FT+RAG -- {N_SAMPLES}-SAMPLE ROBUSTNESS SUMMARY")
print("=" * 80)
for key, s in summary.items():
    print(f"{key:10s}: mean={s['mean']:.4f} std={s['std']:.4f} values={s['values']}")
print("\nOriginal (deterministic / greedy) point estimate: F1=0.8800 (P=0.9429 R=0.825 Acc=0.8085)")
print("\nDone.")

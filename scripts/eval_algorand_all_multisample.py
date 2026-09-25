"""
Full sampling-robustness sweep for Algorand: all 4 models (Qwen3-32B and
QwQ-32B, base and fine-tuned) x 2 conditions (no-RAG, RAG), 5 seeded samples
each with do_sample=True properly enabled (the original eval scripts passed
temperature/top_p WITHOUT do_sample=True, which transformers 4.57.x silently
treats as greedy decoding -- every previously reported number was actually a
single deterministic point estimate mislabeled as a sampled one).

Qwen3-32B-FT + RAG is skipped here (already covered by
eval_algorand_ft_rag_multisample.py, 5 samples in
results_multisample/algorand_ft_rag/).

Loads each of the 4 models once, runs both its no-RAG and RAG sample loops,
then unloads before moving to the next model, to minimize model-load
overhead across the 7 remaining (model, condition) combinations.
"""

import os
import json
import gc
import time
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
RESULTS_ROOT = os.path.join(BASE_DIR, "results_multisample")
os.makedirs(RESULTS_ROOT, exist_ok=True)

N_SAMPLES = 5

import sys
sys.path.insert(0, BASE_DIR)
from rag import search_similarity, parse_response, create_rag_checklist  # noqa: E402
from prompts import user_prompt  # noqa: E402

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

# NOTE: deliberately NOT using prompts.dev_prompt_think here -- its "###
# Vulnerability name (if any):" header breaks rag.parse_response's regex,
# which expects "### Vulnerability:" directly (no intervening word before
# the colon), causing it to capture "name:" instead of the actual verdict
# on the next line. Same required-behavior structure as the RAG system
# prompt below, minus anything RAG-specific, plus the explicit canonical
# vulnerability-name constraint so the base model's free-form naming still
# matches the fixed-vocabulary parser.
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

from prompts import dev_description_prompt, user_description_prompt  # noqa: E402

print("Prompts defined.")

from unsloth import FastLanguageModel


def load_model(model_name_or_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path, max_seq_length=4096, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


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


def run_no_rag(model, tokenizer, out_dir, seed_base):
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = []
    for s in range(N_SAMPLES):
        start = time.time()
        results = []
        for idx, test_file in enumerate(test_files):
            with open(test_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            code = content["smart_contract"]
            ground_truth = content["vulnerability"]
            seed = seed_base + s * 10000 + idx
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
                {"role": "user", "content": USER_PROMPT_NO_RAG.format(code=code)},
            ]
            response = generate_response(model, tokenizer, messages, seed)
            vulnerabilities = parse_response(response)
            results.append({
                "file": os.path.basename(test_file), "ground_truth": ground_truth,
                "predictions": vulnerabilities, "response": response,
            })
            print(f"  [no_rag][sample {s}] [{idx+1}/{len(test_files)}] "
                  f"{os.path.basename(test_file)}: truth={ground_truth} | pred={vulnerabilities}")
        metrics = compute_metrics(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[no_rag] Sample {s} done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


def run_rag(model, tokenizer, out_dir, seed_base):
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = []
    for s in range(N_SAMPLES):
        start = time.time()
        results = []
        for idx, test_file in enumerate(test_files):
            with open(test_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            code = content["smart_contract"]
            ground_truth = content["vulnerability"]
            seed_desc = seed_base + s * 10000 + idx * 2
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
                "file": os.path.basename(test_file), "ground_truth": ground_truth,
                "predictions": vulnerabilities, "response": response,
            })
            print(f"  [rag][sample {s}] [{idx+1}/{len(test_files)}] "
                  f"{os.path.basename(test_file)}: truth={ground_truth} | pred={vulnerabilities}")
        metrics = compute_metrics(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[rag] Sample {s} done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


def save_summary(out_dir, all_metrics):
    import statistics as stats
    summary = {}
    for key in ("accuracy", "precision", "recall", "f1_score"):
        vals = [m[key] for m in all_metrics]
        summary[key] = {
            "values": vals,
            "mean": round(stats.mean(vals), 4),
            "std": round(stats.stdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(all_metrics), "per_sample_metrics": all_metrics, "summary": summary}, f, indent=2)
    print(f"SUMMARY [{out_dir}]:")
    for key, s in summary.items():
        print(f"  {key:10s}: mean={s['mean']:.4f} std={s['std']:.4f}")


# === Run plan: (name, model_path, run_no_rag, run_rag, seed_base) ===
PLAN = [
    ("qwen3_base", "unsloth/Qwen3-32B", True, True, 10000),
    ("qwen3_ft", os.path.join(BASE_DIR, "models", "Qwen3-32B-pyteal-desc", "checkpoint-189"), True, False, 20000),
    ("qwq_base", "unsloth/QwQ-32B-unsloth-bnb-4bit", True, True, 30000),
    ("qwq_ft", os.path.join(BASE_DIR, "models", "QwQ-32B-pyteal-desc", "checkpoint-189"), True, True, 40000),
]

def already_done(out_dir):
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as f:
            return json.load(f).get("n_samples", 0) >= N_SAMPLES
    except Exception:
        return False


only_model = sys.argv[1] if len(sys.argv) > 1 else None
if only_model:
    PLAN = [p for p in PLAN if p[0] == only_model]
    if not PLAN:
        raise SystemExit(f"Unknown model name: {only_model}")

for name, path, do_no_rag, do_rag, seed_base in PLAN:
    norag_dir = os.path.join(RESULTS_ROOT, f"algorand_{name}_norag")
    rag_dir = os.path.join(RESULTS_ROOT, f"algorand_{name}_rag")
    need_no_rag = do_no_rag and not already_done(norag_dir)
    need_rag = do_rag and not already_done(rag_dir)

    print("=" * 70)
    print(f"MODEL: {name} ({path}) -- need_no_rag={need_no_rag} need_rag={need_rag}")
    print("=" * 70)

    if not need_no_rag and not need_rag:
        print(f"--- {name}: both configs already complete, skipping model load ---")
        continue

    model, tokenizer = load_model(path)
    if need_no_rag:
        print(f"--- {name}: no-RAG, {N_SAMPLES} samples ---")
        run_no_rag(model, tokenizer, norag_dir, seed_base)
    elif do_no_rag:
        print(f"--- {name}: no-RAG already complete, skipping ---")
    if need_rag:
        print(f"--- {name}: RAG, {N_SAMPLES} samples ---")
        run_rag(model, tokenizer, rag_dir, seed_base + 5000)
    elif do_rag:
        print(f"--- {name}: RAG already complete, skipping ---")
    unload_model(model, tokenizer)

print("\nAll Algorand multi-sample configs done.")

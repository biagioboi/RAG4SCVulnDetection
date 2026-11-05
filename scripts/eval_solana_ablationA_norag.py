"""
Major Concern 2 response, Solana instantiation: evaluate the Config A
(final-only, no CoT supervision) checkpoint on the same 97-contract Solana
test set, no-RAG condition, 5 seeded samples -- identical protocol to
Config D's existing results_multisample/solana_ft_norag (produced by
scripts/eval_solana_all_multisample.py). Only the no_rag path is reused
here, pointed at the ablation checkpoint; the RAG / base-model / QwQ
entries of that script's PLAN are irrelevant to this ablation and are not
rerun.

Usage: python scripts/eval_solana_ablationA_norag.py
"""

import os
import json
import re
import gc
import time
import statistics as stats
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-ablationA-final-only")
RESULTS_ROOT = os.path.join(BASE_DIR, "results_multisample")
OUT_DIR = os.path.join(RESULTS_ROOT, "solana_ablationA_norag")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 5
SEED_BASE = 80000
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "4"))

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.startswith("solana_") and f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

SYSTEM_PROMPT_NO_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
Your task is to analyze Rust code precisely and systematically to identify security vulnerabilities.

Required behavior:
- Perform a structured reasoning phase inside <think>...</think> tags.
- After </think>, provide the final judgment inside <final>...</final> tags with this structure:
  - A short summary sentence.
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it can be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

If no vulnerability is found, write "### Vulnerability: Not Vulnerable".
Do NOT include any extra commentary outside these tags.""".strip()

USER_PROMPT_NO_RAG = """Please perform a detailed security analysis of the following Solana smart contract.
Carefully examine its logic, identify any potential vulnerabilities:

Contract Code:

```rust
{code}
```""".strip()

print("Prompts defined.")

KNOWN_VULNERABILITIES = [
    "Missing Key Check", "Access Control", "Type Confusion", "Input Validation",
    "CPI Reentrancy", "Reentrancy", "Unchecked External Calls", "Unchecked Calls",
    "Integer Overflow", "Integer Underflow", "Integer Flow",
    "Bump Seed", "Bump Seed Canonicalization", "Denial of Service", "DoS",
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
    "missing_key_check": "Missing Key Check", "type_confusion": "Type Confusion",
    "cpi_reentrancy": "CPI Reentrancy", "unchecked_calls": "Unchecked External Calls",
    "integer_overflow": "Integer Overflow", "bump_seed": "Bump Seed",
    "dos": "Denial of Service", "not_vulnerable": "Not Vulnerable",
}

print("Parser ready.")

from unsloth import FastLanguageModel


def load_model(model_name_or_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path, max_seq_length=8192, dtype=None, load_in_4bit=True, device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_batch(model, tokenizer, messages_list, seed, max_tokens=1024):
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    enc = tokenizer.apply_chat_template(
        messages_list, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False, padding=True, return_dict=True,
    )
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc["attention_mask"].to("cuda")
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens,
        do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tokenizer.pad_token_id,
    )
    gen = outputs[:, input_ids.shape[-1]:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in gen]


def generate_with_retries(model, tokenizer, messages_list, seed_base, max_tokens=1024, max_attempts=3):
    n = len(messages_list)
    responses = [""] * n
    found_all = [[] for _ in range(n)]
    pending = list(range(n))
    for attempt in range(max_attempts):
        if not pending:
            break
        seed = seed_base + attempt * 100000
        sub_messages = [messages_list[i] for i in pending]
        sub_responses = generate_batch(model, tokenizer, sub_messages, seed, max_tokens=max_tokens)
        still_pending = []
        for local_i, resp in zip(pending, sub_responses):
            success, found = parse_response(resp)
            responses[local_i] = resp
            found_all[local_i] = found
            if not success:
                still_pending.append(local_i)
        pending = still_pending
    return responses, found_all


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
GROUND_TRUTH_MAP = {"integer_flow": "Integer Overflow", "cpi": "CPI Reentrancy"}


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
                fp += len(real_preds)
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
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_vulnerability": dict(per_vuln),
    }


def save_summary(out_dir, all_metrics):
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


def run_no_rag(model, tokenizer, out_dir, seed_base):
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = []
    for s in range(N_SAMPLES):
        start = time.time()
        results = [None] * len(test_files)
        for batch_start in range(0, len(test_files), BATCH_SIZE):
            batch_files = test_files[batch_start:batch_start + BATCH_SIZE]
            messages_list, truths = [], []
            for test_file in batch_files:
                with open(test_file, "r", encoding="utf-8") as f:
                    test_data = json.load(f)
                code = test_data["smart_contract"]
                truth_display = VULN_DISPLAY.get(test_data["vulnerability"], test_data["vulnerability"])
                truths.append(truth_display)
                messages_list.append([
                    {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
                    {"role": "user", "content": USER_PROMPT_NO_RAG.format(code=code)},
                ])
            seed = seed_base + s * 10000 + batch_start * 7
            responses, found_all = generate_with_retries(model, tokenizer, messages_list, seed)
            for local_i, test_file in enumerate(batch_files):
                idx = batch_start + local_i
                results[idx] = {
                    "file": os.path.basename(test_file), "ground_truth": truths[local_i],
                    "predictions": found_all[local_i], "response": responses[local_i],
                }
                print(f"  [no_rag][sample {s}] [{idx+1}/{len(test_files)}] "
                      f"{os.path.basename(test_file)}: truth={truths[local_i]} | pred={found_all[local_i]}")
        metrics = compute_metrics(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[no_rag] Sample {s} done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


def already_done(out_dir):
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as f:
            return json.load(f).get("n_samples", 0) >= N_SAMPLES
    except Exception:
        return False


if already_done(OUT_DIR):
    print(f"--- already complete, skipping ---")
else:
    model, tokenizer = load_model(FT_MODEL_DIR)
    run_no_rag(model, tokenizer, OUT_DIR, SEED_BASE)
    unload_model(model, tokenizer)

print("\nSolana ablation A no-RAG eval done.")

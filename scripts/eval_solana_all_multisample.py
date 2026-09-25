"""
Remaining sampling-robustness sweep for Solana: FT no-RAG, Base+RAG, and
Base no-RAG, all on the 97-contract test set, 5 seeded samples each with
do_sample=True properly enabled. FT + RAG (k=2) is skipped here (already
covered by eval_solana_ft_rag_multisample.py).

Running Base on the same 97-contract set the FT configs use (rather than
the legacy 59-contract set) lets the cross-language comparison table drop
its "different test set size" caveat for the Base, no-RAG row.
"""

import os
import json
import re
import gc
import time
import numpy as np
import statistics as stats
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v14")
RESULTS_ROOT = os.path.join(BASE_DIR, "results_multisample")
os.makedirs(RESULTS_ROOT, exist_ok=True)

N_SAMPLES = 5
NUM_DETAILED = 2

VULN_INFO_PATH = os.path.join(KB_DIR, "vulnerability_info.json")
RAG_CONTRACTS_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl")

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

SYSTEM_PROMPT_DESCRIPTION = """You are an expert smart contract security auditor specialized in Solana and Rust.
Describe the given Solana program with exactly these 3 sections:

### Contract type:
State whether this is a "Solana program instruction handler".

### Contract Purpose
One short paragraph stating the primary objective.

### Functional Description
Step-by-step description of the execution logic. Be explicit about which accounts are validated and how.
Do not include variable names or code syntax.""".strip()

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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading embedding model...")
embed_model = SentenceTransformer("hkunlp/instructor-xl")
print("Embedding model loaded.")

with open(VULN_INFO_PATH, "r", encoding="utf-8") as f:
    vuln_info = json.load(f)

rag_contracts = []
with open(RAG_CONTRACTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rag_contracts.append(json.loads(line))

instruction = "Represent the following functional description of a Solana smart contract, written in Rust:"
rag_embeddings = []
for contract in rag_contracts:
    emb = embed_model.encode([instruction, contract["description"]])
    rag_embeddings.append(emb)
rag_embeddings = np.array(rag_embeddings)
print(f"RAG knowledge base ready: {len(rag_contracts)} contracts")


def search_similar_contracts(query_description):
    query_emb = embed_model.encode([instruction, query_description]).reshape(1, -1)
    results = []
    for i, emb in enumerate(rag_embeddings):
        sim = cosine_similarity(emb.reshape(1, -1), query_emb)[0][0]
        results.append({
            "vulnerability": rag_contracts[i]["vulnerability"],
            "vulnerable_part": rag_contracts[i]["vulnerable_part"],
            "similarity": f"{sim:.4f}",
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    seen, unique_results = set(), []
    for r in results:
        if r["vulnerability"] not in seen:
            seen.add(r["vulnerability"])
            unique_results.append(r)
    return unique_results


def build_rag_checklist(similar_contracts, num_detailed=2):
    checklist = ""
    for i, contract in enumerate(similar_contracts):
        vuln_key = contract["vulnerability"]
        if vuln_key not in vuln_info:
            continue
        info = vuln_info[vuln_key]
        score = contract["similarity"]
        if i < num_detailed:
            checklist += f"""\n    {i+1}. **{info['name']}** ({score})\n        - Description: {info['description']}\n        - Preconditions: {info['precondition']}\n        - Security check: {info['security_check']}\n        - Example vulnerable pattern:\n          ```rust\n    {contract['vulnerable_part'].strip()}\n          ```\n"""
        else:
            checklist += f"""\n    {i+1}. **{info['name']}** ({score})\n"""
    return checklist


embed_model = embed_model.to("cpu")
torch.cuda.empty_cache()
gc.collect()

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


def generate_response(model, tokenizer, messages, seed, max_tokens=1024):
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
        results = []
        for idx, test_file in enumerate(test_files):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            code = test_data["smart_contract"]
            ground_truth = test_data["vulnerability"]
            truth_display = VULN_DISPLAY.get(ground_truth, ground_truth)
            seed = seed_base + s * 10000 + idx
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
                {"role": "user", "content": USER_PROMPT_NO_RAG.format(code=code)},
            ]
            found_vulns, response = [], ""
            for attempt in range(3):
                response = generate_response(model, tokenizer, messages, seed + attempt * 100000)
                success, found_vulns = parse_response(response)
                if success:
                    break
            results.append({
                "file": os.path.basename(test_file), "ground_truth": truth_display,
                "predictions": found_vulns, "response": response,
            })
            print(f"  [no_rag][sample {s}] [{idx+1}/{len(test_files)}] "
                  f"{os.path.basename(test_file)}: truth={truth_display} | pred={found_vulns}")
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
                test_data = json.load(f)
            code = test_data["smart_contract"]
            ground_truth = test_data["vulnerability"]
            truth_display = VULN_DISPLAY.get(ground_truth, ground_truth)

            seed_desc = seed_base + s * 10000 + idx * 4
            desc_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
                {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
            ]
            description = generate_response(model, tokenizer, desc_messages, seed_desc, max_tokens=512)
            similar = search_similar_contracts(description)
            checklist = build_rag_checklist(similar, num_detailed=NUM_DETAILED)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": USER_PROMPT_RAG.format(checklist=checklist, code=code)},
            ]
            found_vulns, response = [], ""
            for attempt in range(3):
                response = generate_response(model, tokenizer, messages, seed_desc + 1 + attempt)
                success, found_vulns = parse_response(response)
                if success:
                    break
            results.append({
                "file": os.path.basename(test_file), "ground_truth": truth_display,
                "predictions": found_vulns, "response": response,
            })
            print(f"  [rag][sample {s}] [{idx+1}/{len(test_files)}] "
                  f"{os.path.basename(test_file)}: truth={truth_display} | pred={found_vulns}")
        metrics = compute_metrics(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[rag] Sample {s} done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


QWQ_FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "QwQ-32B-Solana-Audit")

PLAN = [
    ("solana_ft_norag", FT_MODEL_DIR, "no_rag", 50000),
    ("solana_base_rag", "unsloth/Qwen3-32B", "rag", 60000),
    ("solana_base_norag", "unsloth/Qwen3-32B", "no_rag", 70000),
    ("solana_qwq_base_norag", "unsloth/QwQ-32B-unsloth-bnb-4bit", "no_rag", 80000),
    ("solana_qwq_base_rag", "unsloth/QwQ-32B-unsloth-bnb-4bit", "rag", 90000),
    ("solana_qwq_ft_norag", QWQ_FT_MODEL_DIR, "no_rag", 100000),
    ("solana_qwq_ft_rag", QWQ_FT_MODEL_DIR, "rag", 110000),
]

import sys
only_config = sys.argv[1] if len(sys.argv) > 1 else None
if only_config:
    PLAN = [p for p in PLAN if p[0] == only_config]
    if not PLAN:
        raise SystemExit(f"Unknown config name: {only_config}")


def already_done(out_dir):
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as f:
            return json.load(f).get("n_samples", 0) >= N_SAMPLES
    except Exception:
        return False


loaded_name, model, tokenizer = None, None, None
for out_name, model_path, mode, seed_base in PLAN:
    out_dir = os.path.join(RESULTS_ROOT, out_name)
    if already_done(out_dir):
        print(f"--- {out_name}: already complete, skipping ---")
        continue

    if loaded_name != model_path:
        if model is not None:
            unload_model(model, tokenizer)
        print("=" * 70)
        print(f"Loading model: {model_path}")
        print("=" * 70)
        model, tokenizer = load_model(model_path)
        loaded_name = model_path

    print(f"--- {out_name}: {mode}, {N_SAMPLES} samples ---")
    if mode == "no_rag":
        run_no_rag(model, tokenizer, out_dir, seed_base)
    else:
        run_rag(model, tokenizer, out_dir, seed_base)

if model is not None:
    unload_model(model, tokenizer)

print("\nAll Solana multi-sample configs done.")

"""
Full Qwen3-Coder-30B-A3B-Instruct sweep -- base/fine-tuned x no-RAG/RAG -- on
both Algorand and Solana, mirroring the Qwen3-32B/QwQ-32B/Gemma-4-31B
multi-sample studies. Runs in the main .venv (transformers==4.57.1 already
satisfies Qwen3-MoE's >=4.51.0 requirement).

Usage:
    source .venv/bin/activate
    python scripts/eval_qwen3coder_baseline.py [config_name]
"""

# Unsloth must be imported before transformers/peft (via any other import,
# including rag.py's sentence-transformers dependency below) or its
# optimizations -- critically, the fast MoE kernel patches this script
# depends on for Qwen3-Coder's 128-expert architecture -- are not fully
# applied. Confirmed via a direct comparison: with this ordering violated,
# a single 512-token generation that otherwise takes ~60-80s never
# completed in over 40 minutes.
from unsloth import FastLanguageModel  # noqa: E402,F401

import os
import re
import sys
import json
import gc
import time
import statistics as stats
from collections import defaultdict

import torch
import numpy as np

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
RESULTS_ROOT = os.path.join(BASE_DIR, "results_multisample")
os.makedirs(RESULTS_ROOT, exist_ok=True)

BASE_MODEL_ID = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
ALGORAND_FT_DIR = os.path.join(BASE_DIR, "models", "Qwen3-Coder-30B-pyteal-desc")
SOLANA_FT_DIR = os.path.join(REPO_DIR, "models", "Qwen3-Coder-30B-Solana-Audit")

N_SAMPLES = 5
MAX_NEW_TOKENS = 1536
NUM_DETAILED_SOLANA = 2

sys.path.insert(0, BASE_DIR)
from rag import search_similarity, parse_response as parse_response_algorand, create_rag_checklist  # noqa: E402
from prompts import user_prompt as USER_PROMPT_ALGORAND  # noqa: E402
from prompts import dev_description_prompt, user_description_prompt  # noqa: E402

USER_PROMPT_ALGORAND = USER_PROMPT_ALGORAND.strip()

SYSTEM_PROMPT_ALGORAND_NORAG = """
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

SYSTEM_PROMPT_ALGORAND_RAG = """
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

USER_PROMPT_ALGORAND_RAG = """
Here is a **vulnerability checklist** (from RAG) with your priority guidance. Use it to steer your audit, but also be ready to discover new issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following PyTeal smart contract using the reasoning structure.

Contract Code:

```python
{code}
```
""".strip()

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
SAFE_LABELS_ALGORAND = {"Not Vulnerable", "No security risk", "No vulnerabilities"}


def compute_metrics_algorand(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        truth = MAP_TRUTH_TO_DISPLAY.get(r["ground_truth"], r["ground_truth"])
        preds = set(r["predictions"])
        preds_is_safe = len(preds) == 0 or preds.issubset(SAFE_LABELS_ALGORAND)
        if truth == "Not Vulnerable":
            if preds_is_safe:
                tn += 1
            else:
                real = preds - SAFE_LABELS_ALGORAND
                fp += len(real)
                for p in real:
                    per_vuln[p]["fp"] += 1
        else:
            if truth in preds:
                tp += 1
                per_vuln[truth]["tp"] += 1
                extras = preds - {truth} - SAFE_LABELS_ALGORAND
                fp += len(extras)
                for e in extras:
                    per_vuln[e]["fp"] += 1
            else:
                fn += 1
                per_vuln[truth]["fn"] += 1
                wrong = preds - SAFE_LABELS_ALGORAND
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


# --- Solana prompts/parser (copied inline, matching eval_solana_all_multisample.py) ---
SYSTEM_PROMPT_SOLANA_NORAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
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

USER_PROMPT_SOLANA_NORAG = """Please perform a detailed security analysis of the following Solana smart contract.
Carefully examine its logic, identify any potential vulnerabilities:

Contract Code:

```rust
{code}
```""".strip()

SYSTEM_PROMPT_SOLANA_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
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

USER_PROMPT_SOLANA_RAG = """Here is a vulnerability checklist (from RAG) with priority guidance.
Use it to steer your audit, but also look for issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following Solana smart contract:

Contract Code:

```rust
{code}
```""".strip()

SYSTEM_PROMPT_SOLANA_DESCRIPTION = """You are an expert smart contract security auditor specialized in Solana and Rust.
Describe the given Solana program with exactly these 3 sections:

### Contract type:
State whether this is a "Solana program instruction handler".

### Contract Purpose
One short paragraph stating the primary objective.

### Functional Description
Step-by-step description of the execution logic. Be explicit about which accounts are validated and how.
Do not include variable names or code syntax.""".strip()

KNOWN_VULNERABILITIES_SOLANA = [
    "Missing Key Check", "Access Control", "Type Confusion", "Input Validation",
    "CPI Reentrancy", "Reentrancy", "Unchecked External Calls", "Unchecked Calls",
    "Integer Overflow", "Integer Underflow", "Integer Flow",
    "Bump Seed", "Bump Seed Canonicalization", "Denial of Service", "DoS",
    "Not Vulnerable", "No security risk", "No vulnerabilities", "None detected",
]


def parse_response_solana(response):
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
            for v in KNOWN_VULNERABILITIES_SOLANA:
                if v.lower() == part or v.lower() in part:
                    if v not in found:
                        found.append(v)
                    break
    return (True, found) if found else (False, [])


VULN_DISPLAY_SOLANA = {
    "missing_key_check": "Missing Key Check", "type_confusion": "Type Confusion",
    "cpi_reentrancy": "CPI Reentrancy", "unchecked_calls": "Unchecked External Calls",
    "integer_overflow": "Integer Overflow", "bump_seed": "Bump Seed",
    "dos": "Denial of Service", "not_vulnerable": "Not Vulnerable",
}
VULNERABILITY_ALIASES_SOLANA = {
    "Missing Key Check": ["missing key check", "access control"],
    "Type Confusion": ["type confusion", "input validation"],
    "CPI Reentrancy": ["cpi reentrancy", "cpi", "reentrancy"],
    "Unchecked External Calls": ["unchecked external calls", "unchecked calls"],
    "Integer Overflow": ["integer overflow", "integer underflow", "integer flow"],
    "Bump Seed": ["bump seed", "bump seed canonicalization"],
    "Denial of Service": ["denial of service", "dos"],
    "Not Vulnerable": ["not vulnerable", "no security risk", "no vulnerabilities", "none detected"],
}
GROUND_TRUTH_MAP_SOLANA = {"integer_flow": "Integer Overflow", "cpi": "CPI Reentrancy"}


def normalize_predictions_solana(predictions):
    canonical_set = set()
    for pred in predictions:
        pred_lower = pred.lower().strip()
        for canonical, aliases in VULNERABILITY_ALIASES_SOLANA.items():
            if pred_lower in aliases:
                canonical_set.add(canonical)
                break
    return canonical_set


def compute_metrics_solana(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        truth = r["ground_truth"]
        predictions = normalize_predictions_solana(r["predictions"])
        if truth in GROUND_TRUTH_MAP_SOLANA:
            truth = GROUND_TRUTH_MAP_SOLANA[truth]
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


# --- Model loading (Unsloth) ---

def load_model(model_name_or_path, max_seq_length=8192):
    from unsloth import FastLanguageModel
    print(f"Loading model: {model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path, max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True, device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_response(model, tokenizer, messages, seed, max_tokens=MAX_NEW_TOKENS):
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


def run_algorand(model, tokenizer, out_dir, seed_base, rag):
    test_dir = os.path.join(BASE_DIR, "test_contracts")
    test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".json")])
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
            if not rag:
                seed = seed_base + s * 10000 + idx
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_ALGORAND_NORAG},
                    {"role": "user", "content": USER_PROMPT_ALGORAND.format(code=code)},
                ]
                response = generate_response(model, tokenizer, messages, seed)
            else:
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
                    {"role": "system", "content": SYSTEM_PROMPT_ALGORAND_RAG},
                    {"role": "user", "content": USER_PROMPT_ALGORAND_RAG.format(checklist=checklist, code=code)},
                ]
                response = generate_response(model, tokenizer, messages, seed_audit)
            vulnerabilities = parse_response_algorand(response)
            results.append({
                "file": os.path.basename(test_file), "ground_truth": ground_truth,
                "predictions": vulnerabilities, "response": response,
            })
            print(f"  [algorand][sample {s}] [{idx+1}/{len(test_files)}] "
                  f"{os.path.basename(test_file)}: truth={ground_truth} | pred={vulnerabilities}")
        metrics = compute_metrics_algorand(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[algorand] Sample {s} done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


_solana_rag_state = None


def get_solana_rag_state():
    global _solana_rag_state
    if _solana_rag_state is not None:
        return _solana_rag_state

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
    with open(os.path.join(KB_DIR, "vulnerability_info.json"), "r", encoding="utf-8") as f:
        vuln_info = json.load(f)
    rag_contracts = []
    with open(os.path.join(KB_DIR, "rag_contracts.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            rag_contracts.append(json.loads(line))

    print("Loading Solana RAG embedding model...")
    embed_model = SentenceTransformer("hkunlp/instructor-xl")
    instruction = "Represent the following functional description of a Solana smart contract, written in Rust:"
    rag_embeddings = []
    for contract in rag_contracts:
        emb = embed_model.encode([instruction, contract["description"]])
        rag_embeddings.append(emb)
    rag_embeddings = np.array(rag_embeddings)
    embed_model = embed_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Solana RAG knowledge base ready: {len(rag_contracts)} contracts")

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

    _solana_rag_state = (search_similar_contracts, build_rag_checklist)
    return _solana_rag_state


def run_solana(model, tokenizer, out_dir, seed_base, rag):
    test_dir = os.path.join(REPO_DIR, "data", "test_set")
    test_files = sorted([
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.startswith("solana_") and f.endswith(".json")
    ])
    os.makedirs(out_dir, exist_ok=True)
    search_similar_contracts, build_rag_checklist = get_solana_rag_state() if rag else (None, None)
    all_metrics = []
    for s in range(N_SAMPLES):
        start = time.time()
        results = []
        for idx, test_file in enumerate(test_files):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            code = test_data["smart_contract"]
            truth_display = VULN_DISPLAY_SOLANA.get(test_data["vulnerability"], test_data["vulnerability"])
            if not rag:
                seed = seed_base + s * 10000 + idx
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SOLANA_NORAG},
                    {"role": "user", "content": USER_PROMPT_SOLANA_NORAG.format(code=code)},
                ]
                response = generate_response(model, tokenizer, messages, seed)
            else:
                seed_desc = seed_base + s * 10000 + idx * 4
                desc_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SOLANA_DESCRIPTION},
                    {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
                ]
                description = generate_response(model, tokenizer, desc_messages, seed_desc, max_tokens=512)
                similar = search_similar_contracts(description)
                checklist = build_rag_checklist(similar, num_detailed=NUM_DETAILED_SOLANA)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SOLANA_RAG},
                    {"role": "user", "content": USER_PROMPT_SOLANA_RAG.format(checklist=checklist, code=code)},
                ]
                response = generate_response(model, tokenizer, messages, seed_desc + 1)
            success, found_vulns = parse_response_solana(response)
            results.append({
                "file": os.path.basename(test_file), "ground_truth": truth_display,
                "predictions": found_vulns, "response": response,
            })
            print(f"  [solana][sample {s}] [{idx+1}/{len(test_files)}] "
                  f"{os.path.basename(test_file)}: truth={truth_display} | pred={found_vulns}")
        metrics = compute_metrics_solana(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[solana] Sample {s} done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


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


def already_done(out_dir):
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as f:
            return json.load(f).get("n_samples", 0) >= N_SAMPLES
    except Exception:
        return False


# (config_name, language, model_path, rag, seed_base, max_seq_length)
PLAN = [
    ("algorand_qwen3coder_norag", "algorand", BASE_MODEL_ID, False, 140000, 4096),
    ("algorand_qwen3coder_rag", "algorand", BASE_MODEL_ID, True, 141000, 4096),
    ("algorand_qwen3coder_ft_norag", "algorand", ALGORAND_FT_DIR, False, 142000, 4096),
    ("algorand_qwen3coder_ft_rag", "algorand", ALGORAND_FT_DIR, True, 143000, 4096),
    ("solana_qwen3coder_norag", "solana", BASE_MODEL_ID, False, 150000, 8192),
    ("solana_qwen3coder_rag", "solana", BASE_MODEL_ID, True, 151000, 8192),
    ("solana_qwen3coder_ft_norag", "solana", SOLANA_FT_DIR, False, 152000, 8192),
    ("solana_qwen3coder_ft_rag", "solana", SOLANA_FT_DIR, True, 153000, 8192),
]

only_config = sys.argv[1] if len(sys.argv) > 1 else None
if only_config:
    PLAN = [p for p in PLAN if p[0] == only_config]
    if not PLAN:
        raise SystemExit(f"Unknown config name: {only_config}")

loaded_path, model, tokenizer = None, None, None
for out_name, language, model_path, rag, seed_base, max_seq_length in PLAN:
    out_dir = os.path.join(RESULTS_ROOT, out_name)
    if already_done(out_dir):
        print(f"--- {out_name}: already complete, skipping ---")
        continue
    if loaded_path != model_path:
        if model is not None:
            unload_model(model, tokenizer)
        model, tokenizer = load_model(model_path, max_seq_length=max_seq_length)
        loaded_path = model_path
    print("=" * 70)
    print(f"CONFIG: {out_name} (language={language}, rag={rag}, model={model_path})")
    print("=" * 70)
    if language == "algorand":
        run_algorand(model, tokenizer, out_dir, seed_base, rag)
    else:
        run_solana(model, tokenizer, out_dir, seed_base, rag)

if model is not None:
    unload_model(model, tokenizer)

print("\nAll Qwen3-Coder-30B configs done.")

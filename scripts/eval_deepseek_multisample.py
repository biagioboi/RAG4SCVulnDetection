"""
SOTA generalist (DeepSeek-V3.2, via API) benchmark, no-RAG and RAG, on BOTH
Algorand (45 contracts) and Solana (97 contracts), 5 independent stochastic
samples per config -- same N=5 standard used for every open-weight model in
the paper.

This replaces the previous single-run Algorand-only DeepSeek numbers (which
left the SOTA row the only one in every table without a std), and adds the
Solana comparison that was previously missing entirely.

Reproducibility note: like every other API-based model in this repo
(scripts/eval_api_baseline.py), the "5 samples" are 5 independent stochastic
calls at temperature 0.6 (not individually seeded -- the DeepSeek API
offers no deterministic seed parameter), so these runs are a valid basis
for mean/std but not bit-reproducible run-to-run.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python scripts/eval_deepseek_multisample.py [config_name]

config_name (optional): one of
    algorand_deepseek_norag, algorand_deepseek_rag,
    solana_deepseek_norag, solana_deepseek_rag
"""

import os
import sys
import json
import re
import time
import gc
import statistics as stats
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.insert(0, BASE_DIR)

# This script is purely API-based (no local model inference); force the RAG
# embedding model (loaded by rag.py at import time, and by _load_solana_rag)
# onto CPU so it never competes with other jobs for GPU memory.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
RESULTS_ROOT = os.path.join(BASE_DIR, "results_multisample")
os.makedirs(RESULTS_ROOT, exist_ok=True)

N_SAMPLES = 5
MAX_RETRY = 3
MAX_WORKERS = 6

MODEL_ID = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
API_KEY = os.environ["DEEPSEEK_API_KEY"]  # raises KeyError if unset, on purpose
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def call_api(messages, max_tokens=4000):
    """reasoning_effort='none' is critical: without it, deepseek-v4-pro (and
    deepseek-flash) spend an unbounded, separately-billed hidden reasoning
    budget (completion_tokens_details.reasoning_tokens) before emitting any
    visible content -- observed consuming the entire max_tokens budget (even
    at 16000) while still returning empty/truncated content, which silently
    corrupts every downstream prediction as "no vulnerability found". With
    reasoning_effort='none', finish_reason is reliably 'stop' and the model's
    own <think> block (already requested in-prompt) carries the reasoning
    instead, at a fraction of the token cost.
    """
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID, messages=messages,
                max_tokens=max_tokens, temperature=0.6, top_p=0.95,
                extra_body={"reasoning_effort": "none"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            wait = 15 * (attempt + 1)
            print(f"  [api error, attempt {attempt+1}/{MAX_RETRY}] {e} -- retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {MAX_RETRY} attempts: {last_err}")


# ============================== ALGORAND ==============================

from rag import search_similarity, parse_response as parse_response_algorand, create_rag_checklist  # noqa: E402
from prompts import user_prompt as USER_PROMPT_NO_RAG_ALGORAND, dev_description_prompt, user_description_prompt  # noqa: E402

ALGORAND_TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
USER_PROMPT_NO_RAG_ALGORAND = USER_PROMPT_NO_RAG_ALGORAND.strip()

ALGORAND_SYSTEM_PROMPT_NO_RAG = """
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

ALGORAND_SYSTEM_PROMPT_RAG = """
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

USER_PROMPT_RAG_ALGORAND = """
Here is a **vulnerability checklist** (from RAG) with your priority guidance. Use it to steer your audit, but also be ready to discover new issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following PyTeal smart contract using the reasoning structure.

Contract Code:

```python
{code}
```
""".strip()

ALGORAND_MAP_TRUTH_TO_DISPLAY = {
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
ALGORAND_SAFE_LABELS = {"Not Vulnerable", "No security risk", "No vulnerabilities"}


def compute_metrics_algorand(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        truth = ALGORAND_MAP_TRUTH_TO_DISPLAY.get(r["ground_truth"], r["ground_truth"])
        preds = set(r["predictions"])
        preds_is_safe = len(preds) == 0 or preds.issubset(ALGORAND_SAFE_LABELS)
        if truth == "Not Vulnerable":
            if preds_is_safe:
                tn += 1
            else:
                real = preds - ALGORAND_SAFE_LABELS
                fp += len(real)
                for p in real:
                    per_vuln[p]["fp"] += 1
        else:
            if truth in preds:
                tp += 1
                per_vuln[truth]["tp"] += 1
                extras = preds - {truth} - ALGORAND_SAFE_LABELS
                fp += len(extras)
                for e in extras:
                    per_vuln[e]["fp"] += 1
            else:
                fn += 1
                per_vuln[truth]["fn"] += 1
                wrong = preds - ALGORAND_SAFE_LABELS
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


def _algorand_one_no_rag(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    code = content["smart_contract"]
    ground_truth = content["vulnerability"]
    messages = [
        {"role": "system", "content": ALGORAND_SYSTEM_PROMPT_NO_RAG},
        {"role": "user", "content": USER_PROMPT_NO_RAG_ALGORAND.format(code=code)},
    ]
    response = call_api(messages)
    vulnerabilities = parse_response_algorand(response)
    return {
        "file": os.path.basename(test_file), "ground_truth": ground_truth,
        "predictions": vulnerabilities, "response": response,
    }


def _algorand_one_rag(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    code = content["smart_contract"]
    ground_truth = content["vulnerability"]
    desc_messages = [
        {"role": "system", "content": dev_description_prompt},
        {"role": "user", "content": user_description_prompt.format(code=code)},
    ]
    description = call_api(desc_messages, max_tokens=512)
    rag_contracts = search_similarity(description)
    checklist = create_rag_checklist(rag_contracts, with_few_shot=True)
    messages = [
        {"role": "system", "content": ALGORAND_SYSTEM_PROMPT_RAG},
        {"role": "user", "content": USER_PROMPT_RAG_ALGORAND.format(checklist=checklist, code=code)},
    ]
    response = call_api(messages)
    vulnerabilities = parse_response_algorand(response)
    return {
        "file": os.path.basename(test_file), "ground_truth": ground_truth,
        "predictions": vulnerabilities, "response": response,
    }


def run_algorand(out_dir, rag):
    test_files = sorted([
        os.path.join(ALGORAND_TEST_DIR, f) for f in os.listdir(ALGORAND_TEST_DIR)
        if f.endswith(".json")
    ])
    print(f"[algorand] {len(test_files)} test contracts, rag={rag}")
    worker = _algorand_one_rag if rag else _algorand_one_no_rag
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = []
    for s in range(N_SAMPLES):
        start = time.time()
        results = [None] * len(test_files)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(worker, tf): i for i, tf in enumerate(test_files)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                r = results[idx]
                print(f"  [sample {s}] [{idx+1}/{len(test_files)}] {r['file']}: "
                      f"truth={r['ground_truth']} | pred={r['predictions']}")
        metrics = compute_metrics_algorand(results)
        with open(os.path.join(out_dir, f"results_sample{s}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[algorand][sample {s}] done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


# ============================== SOLANA ==============================

SOLANA_TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
SOLANA_KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
SOLANA_VULN_INFO_PATH = os.path.join(SOLANA_KB_DIR, "vulnerability_info.json")
SOLANA_RAG_CONTRACTS_PATH = os.path.join(SOLANA_KB_DIR, "rag_contracts.jsonl")

SOLANA_SYSTEM_PROMPT_NO_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
Your task is to analyze Rust code precisely and systematically to identify security vulnerabilities.

Required behavior:
- Perform a structured reasoning phase inside <think>...</think> tags.
- After </think>, provide the final judgment inside <final>...</final> tags with this structure:
  - A short summary sentence.
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it can be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

If no vulnerability is found, write "### Vulnerability: Not Vulnerable".
The <name> in "### Vulnerability:" MUST be exactly one of: Missing Key Check, Type Confusion, CPI Reentrancy, Unchecked External Calls, Integer Overflow, Bump Seed, Denial of Service, Not Vulnerable.
Do NOT include any extra commentary outside these tags.""".strip()

SOLANA_USER_PROMPT_NO_RAG = """Please perform a detailed security analysis of the following Solana smart contract.
Carefully examine its logic, identify any potential vulnerabilities:

Contract Code:

```rust
{code}
```""".strip()

SOLANA_SYSTEM_PROMPT_RAG = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
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
The <name> in "### Vulnerability:" MUST be exactly one of: Missing Key Check, Type Confusion, CPI Reentrancy, Unchecked External Calls, Integer Overflow, Bump Seed, Denial of Service, Not Vulnerable.
Do NOT include any extra commentary outside these tags.""".strip()

SOLANA_USER_PROMPT_RAG = """Here is a vulnerability checklist (from RAG) with priority guidance.
Use it to steer your audit, but also look for issues beyond this list.

{checklist}

Now perform a detailed security analysis of the following Solana smart contract:

Contract Code:

```rust
{code}
```""".strip()

SOLANA_SYSTEM_PROMPT_DESCRIPTION = """You are an expert smart contract security auditor specialized in Solana and Rust.
Describe the given Solana program with exactly these 3 sections:

### Contract type:
State whether this is a "Solana program instruction handler".

### Contract Purpose
One short paragraph stating the primary objective.

### Functional Description
Step-by-step description of the execution logic. Be explicit about which accounts are validated and how.
Do not include variable names or code syntax.""".strip()

SOLANA_KNOWN_VULNERABILITIES = [
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
            for v in SOLANA_KNOWN_VULNERABILITIES:
                if v.lower() == part or v.lower() in part:
                    if v not in found:
                        found.append(v)
                    break
    return found


SOLANA_VULN_DISPLAY = {
    "missing_key_check": "Missing Key Check", "type_confusion": "Type Confusion",
    "cpi_reentrancy": "CPI Reentrancy", "unchecked_calls": "Unchecked External Calls",
    "integer_overflow": "Integer Overflow", "bump_seed": "Bump Seed",
    "dos": "Denial of Service", "not_vulnerable": "Not Vulnerable",
}

SOLANA_VULNERABILITY_ALIASES = {
    "Missing Key Check": ["missing key check", "access control"],
    "Type Confusion": ["type confusion", "input validation"],
    "CPI Reentrancy": ["cpi reentrancy", "cpi", "reentrancy"],
    "Unchecked External Calls": ["unchecked external calls", "unchecked calls"],
    "Integer Overflow": ["integer overflow", "integer underflow", "integer flow"],
    "Bump Seed": ["bump seed", "bump seed canonicalization"],
    "Denial of Service": ["denial of service", "dos"],
    "Not Vulnerable": ["not vulnerable", "no security risk", "no vulnerabilities", "none detected"],
}
SOLANA_GROUND_TRUTH_MAP = {"integer_flow": "Integer Overflow", "cpi": "CPI Reentrancy"}


def normalize_predictions_solana(predictions):
    canonical_set = set()
    for pred in predictions:
        pred_lower = pred.lower().strip()
        for canonical, aliases in SOLANA_VULNERABILITY_ALIASES.items():
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
        if truth in SOLANA_GROUND_TRUTH_MAP:
            truth = SOLANA_GROUND_TRUTH_MAP[truth]
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


_solana_embed_model = None
_solana_rag_contracts = None
_solana_rag_embeddings = None
_solana_vuln_info = None
_SOLANA_INSTRUCTION = "Represent the following functional description of a Solana smart contract, written in Rust:"


def _load_solana_rag():
    global _solana_embed_model, _solana_rag_contracts, _solana_rag_embeddings, _solana_vuln_info
    if _solana_embed_model is not None:
        return
    from sentence_transformers import SentenceTransformer
    print("[solana] loading embedding model for RAG...")
    _solana_embed_model = SentenceTransformer("hkunlp/instructor-xl")
    with open(SOLANA_VULN_INFO_PATH, "r", encoding="utf-8") as f:
        _solana_vuln_info = json.load(f)
    _solana_rag_contracts = []
    with open(SOLANA_RAG_CONTRACTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            _solana_rag_contracts.append(json.loads(line))
    embeddings = []
    for contract in _solana_rag_contracts:
        emb = _solana_embed_model.encode([_SOLANA_INSTRUCTION, contract["description"]])
        embeddings.append(emb)
    _solana_rag_embeddings = np.array(embeddings)
    print(f"[solana] RAG knowledge base ready: {len(_solana_rag_contracts)} contracts")


def _solana_search_similar(query_description):
    from sklearn.metrics.pairwise import cosine_similarity
    query_emb = _solana_embed_model.encode([_SOLANA_INSTRUCTION, query_description]).reshape(1, -1)
    results = []
    for i, emb in enumerate(_solana_rag_embeddings):
        sim = cosine_similarity(emb.reshape(1, -1), query_emb)[0][0]
        results.append({
            "vulnerability": _solana_rag_contracts[i]["vulnerability"],
            "vulnerable_part": _solana_rag_contracts[i]["vulnerable_part"],
            "similarity": f"{sim:.4f}",
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    seen, unique_results = set(), []
    for r in results:
        if r["vulnerability"] not in seen:
            seen.add(r["vulnerability"])
            unique_results.append(r)
    return unique_results


def _solana_build_checklist(similar_contracts, num_detailed=2):
    checklist = ""
    for i, contract in enumerate(similar_contracts):
        vuln_key = contract["vulnerability"]
        if vuln_key not in _solana_vuln_info:
            continue
        info = _solana_vuln_info[vuln_key]
        score = contract["similarity"]
        if i < num_detailed:
            checklist += f"""\n    {i+1}. **{info['name']}** ({score})\n        - Description: {info['description']}\n        - Preconditions: {info['precondition']}\n        - Security check: {info['security_check']}\n        - Example vulnerable pattern:\n          ```rust\n    {contract['vulnerable_part'].strip()}\n          ```\n"""
        else:
            checklist += f"""\n    {i+1}. **{info['name']}** ({score})\n"""
    return checklist


def _solana_one_no_rag(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    code = test_data["smart_contract"]
    truth = SOLANA_VULN_DISPLAY.get(test_data["vulnerability"], test_data["vulnerability"])
    messages = [
        {"role": "system", "content": SOLANA_SYSTEM_PROMPT_NO_RAG},
        {"role": "user", "content": SOLANA_USER_PROMPT_NO_RAG.format(code=code)},
    ]
    response = call_api(messages)
    predictions = parse_response_solana(response)
    return {
        "file": os.path.basename(test_file), "ground_truth": truth,
        "predictions": predictions, "response": response,
    }


def _solana_one_rag(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    code = test_data["smart_contract"]
    truth = SOLANA_VULN_DISPLAY.get(test_data["vulnerability"], test_data["vulnerability"])
    desc_messages = [
        {"role": "system", "content": SOLANA_SYSTEM_PROMPT_DESCRIPTION},
        {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
    ]
    description = call_api(desc_messages, max_tokens=512)
    similar = _solana_search_similar(description)
    checklist = _solana_build_checklist(similar, num_detailed=2)
    messages = [
        {"role": "system", "content": SOLANA_SYSTEM_PROMPT_RAG},
        {"role": "user", "content": SOLANA_USER_PROMPT_RAG.format(checklist=checklist, code=code)},
    ]
    response = call_api(messages)
    predictions = parse_response_solana(response)
    return {
        "file": os.path.basename(test_file), "ground_truth": truth,
        "predictions": predictions, "response": response,
    }


def run_solana(out_dir, rag):
    test_files = sorted([
        os.path.join(SOLANA_TEST_DIR, f) for f in os.listdir(SOLANA_TEST_DIR)
        if f.startswith("solana_") and f.endswith(".json")
    ])
    print(f"[solana] {len(test_files)} test contracts, rag={rag}")
    if rag:
        _load_solana_rag()
    worker = _solana_one_rag if rag else _solana_one_no_rag
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = []
    for s in range(N_SAMPLES):
        sample_path = os.path.join(out_dir, f"results_sample{s}.json")
        if os.path.exists(sample_path):
            with open(sample_path, encoding="utf-8") as f:
                metrics = json.load(f)["metrics"]
            all_metrics.append(metrics)
            print(f"[solana][sample {s}] already done, skipping. F1={metrics['f1_score']}")
            continue
        start = time.time()
        results = [None] * len(test_files)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(worker, tf): i for i, tf in enumerate(test_files)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                r = results[idx]
                print(f"  [sample {s}] [{idx+1}/{len(test_files)}] {r['file']}: "
                      f"truth={r['ground_truth']} | pred={r['predictions']}")
        metrics = compute_metrics_solana(results)
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump({"sample": s, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
        all_metrics.append(metrics)
        print(f"[solana][sample {s}] done in {(time.time()-start)/60:.1f} min. F1={metrics['f1_score']}")
    save_summary(out_dir, all_metrics)


# ============================== SHARED ==============================

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


PLAN = [
    ("algorand_deepseek_norag", run_algorand, False),
    ("algorand_deepseek_rag", run_algorand, True),
    ("solana_deepseek_norag", run_solana, False),
    ("solana_deepseek_rag", run_solana, True),
]

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


if __name__ == "__main__":
    for name, fn, rag in PLAN:
        out_dir = os.path.join(RESULTS_ROOT, name)
        if already_done(out_dir):
            print(f"=== {name}: already complete, skipping ===")
            continue
        print(f"\n=== Running {name} (rag={rag}) ===")
        fn(out_dir, rag)
        gc.collect()
    print("\nAll configured DeepSeek-V3.2 benchmarks complete.")

"""
Zero-shot (no fine-tuning, no RAG) baseline for additional SOTA generalist
models accessed via API, on both Algorand and Solana test sets, mirroring
the DeepSeek-V3.2 no-RAG benchmark already reported in the paper.

Currently configured for:
  - GLM-4.6 (Zhipu/Z.ai)   -- env var ZAI_API_KEY
  - Kimi K2.6 (Moonshot)   -- env var MOONSHOT_API_KEY

Note on reproducibility: unlike the local open-weight models, these APIs do
not offer a reliable per-request deterministic seed across providers, so the
5 "samples" per config are 5 independent stochastic calls at temperature 0.6
(not individually seeded) -- still a valid basis for mean/std, just not
bit-reproducible run-to-run the way the local multi-sample scripts are.

Usage:
    export ZAI_API_KEY="..."
    export MOONSHOT_API_KEY="..."
    python scripts/eval_api_baseline.py [config_name]

config_name (optional): filter to one entry of PLAN, e.g. "algorand_glm46_norag".
"""

import os
import sys
import json
import re
import time
import statistics as stats
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
load_dotenv(os.path.join(BASE_DIR, ".env"))
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
RESULTS_ROOT = os.path.join(BASE_DIR, "results_multisample")
os.makedirs(RESULTS_ROOT, exist_ok=True)

N_SAMPLES = 5
MAX_RETRY = 3

# === API model definitions ===
# max_tokens is generous by default because reasoning models (glm-4.7-flash
# confirmed via smoke test) spend a large, variable share of the completion
# budget on hidden chain-of-thought (reasoning_content / reasoning_tokens)
# before ever emitting the <final> block; too small a budget truncates mid-
# reasoning and the response never reaches a parseable verdict.
MODELS = {
    "glm47flash": dict(
        display="GLM-4.7-Flash",
        model_id="glm-4.7-flash",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key_env="ZAI_API_KEY",
        max_tokens=6000,
    ),
    "glm46": dict(
        display="GLM-4.6",
        model_id="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key_env="ZAI_API_KEY",
        max_tokens=4000,
    ),
    "kimik26": dict(
        display="Kimi K2.6",
        model_id="kimi-k2.6",
        base_url="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        max_tokens=4000,
    ),
    "nemotronultra": dict(
        # qwen/qwen3-coder:free was deprecated (now paid-only) as of this
        # writing; nemotron-3-ultra-550b-a55b is the current free model
        # closest in scale to a SOTA-generalist comparator. Verified working
        # (cost=0) via a live smoke test before wiring this in.
        display="Nemotron-3-Ultra-550B-A55B (free via OpenRouter)",
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4000,
    ),
}


def get_client(model_key):
    cfg = MODELS[model_key]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"Missing env var {cfg['api_key_env']} for {cfg['display']}")
    return OpenAI(base_url=cfg["base_url"], api_key=api_key), cfg["model_id"], cfg["max_tokens"]


def call_api(client, model_id, messages, max_tokens):
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages,
                max_tokens=max_tokens, temperature=0.6, top_p=0.95,
            )
            content = resp.choices[0].message.content or ""
            if resp.choices[0].finish_reason == "length" and not content.strip():
                # Common with reasoning models on the free tier: the whole
                # budget was spent on hidden chain-of-thought before any
                # visible <final> block was emitted. Retrying won't help
                # (same budget), so surface it as an empty response instead
                # of silently returning nothing useful.
                print("  [warn] truncated with empty content (reasoning budget exhausted)")
            return content
        except Exception as e:
            last_err = e
            wait = 15 * (attempt + 1)
            print(f"  [api error, attempt {attempt+1}/{MAX_RETRY}] {e} -- retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {MAX_RETRY} attempts: {last_err}")


# ============================== ALGORAND ==============================

sys.path.insert(0, BASE_DIR)
from rag import parse_response as parse_response_algorand  # noqa: E402
from prompts import user_prompt as USER_PROMPT_ALGORAND  # noqa: E402

ALGORAND_TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
USER_PROMPT_ALGORAND = USER_PROMPT_ALGORAND.strip()

# Same corrected prompt used in eval_algorand_all_multisample.py (parser-compatible phrasing).
SYSTEM_PROMPT_ALGORAND = """
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


def run_algorand(client, model_id, max_tokens, out_dir):
    test_files = sorted([
        os.path.join(ALGORAND_TEST_DIR, f) for f in os.listdir(ALGORAND_TEST_DIR)
        if f.endswith(".json")
    ])
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
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_ALGORAND},
                {"role": "user", "content": USER_PROMPT_ALGORAND.format(code=code)},
            ]
            response = call_api(client, model_id, messages, max_tokens)
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


# ============================== SOLANA ==============================

SOLANA_TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")

SYSTEM_PROMPT_SOLANA = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
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

USER_PROMPT_SOLANA = """Please perform a detailed security analysis of the following Solana smart contract.
Carefully examine its logic, identify any potential vulnerabilities:

Contract Code:

```rust
{code}
```""".strip()

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


def run_solana(client, model_id, max_tokens, out_dir):
    test_files = sorted([
        os.path.join(SOLANA_TEST_DIR, f) for f in os.listdir(SOLANA_TEST_DIR)
        if f.startswith("solana_") and f.endswith(".json")
    ])
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = []
    for s in range(N_SAMPLES):
        start = time.time()
        results = []
        for idx, test_file in enumerate(test_files):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            code = test_data["smart_contract"]
            truth_display = VULN_DISPLAY_SOLANA.get(test_data["vulnerability"], test_data["vulnerability"])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_SOLANA},
                {"role": "user", "content": USER_PROMPT_SOLANA.format(code=code)},
            ]
            response = call_api(client, model_id, messages, max_tokens)
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


def already_done(out_dir):
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as f:
            return json.load(f).get("n_samples", 0) >= N_SAMPLES
    except Exception:
        return False


# glm46 requires paid credits (confirmed via smoke test: "Insufficient
# balance or no resource package"); glm47flash works on the free tier.
PLAN = [
    ("algorand_glm47flash_norag", "glm47flash", "algorand"),
    ("solana_glm47flash_norag", "glm47flash", "solana"),
    ("algorand_glm46_norag", "glm46", "algorand"),
    ("solana_glm46_norag", "glm46", "solana"),
    ("algorand_kimik26_norag", "kimik26", "algorand"),
    ("solana_kimik26_norag", "kimik26", "solana"),
    ("algorand_nemotronultra_norag", "nemotronultra", "algorand"),
    ("solana_nemotronultra_norag", "nemotronultra", "solana"),
]

only_config = sys.argv[1] if len(sys.argv) > 1 else None
if only_config:
    PLAN = [p for p in PLAN if p[0] == only_config]
    if not PLAN:
        raise SystemExit(f"Unknown config name: {only_config}")

for out_name, model_key, language in PLAN:
    out_dir = os.path.join(RESULTS_ROOT, out_name)
    if already_done(out_dir):
        print(f"--- {out_name}: already complete, skipping ---")
        continue
    client, model_id, max_tokens = get_client(model_key)
    print("=" * 70)
    print(f"CONFIG: {out_name} (model={MODELS[model_key]['display']}, language={language}, max_tokens={max_tokens})")
    print("=" * 70)
    if language == "algorand":
        run_algorand(client, model_id, max_tokens, out_dir)
    else:
        run_solana(client, model_id, max_tokens, out_dir)

print("\nAll API baseline configs done.")

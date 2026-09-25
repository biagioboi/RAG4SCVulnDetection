"""
Evaluation: Qwen3-32B for Solana Vulnerability Detection (4 configurations).

Re-run of notebooks/qwen3-32b/qwen3-evaluation.ipynb from LLM-Contract-Analyzer,
adapted to run as a standalone script on this machine.

Results are written to results/qwen3-32b-rerun/ (kept separate from the
original results/qwen3-32b/ so the two runs can be compared directly).
"""

import os
import json
import re
import gc
import time
import numpy as np
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v4")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "qwen3-32b-v4")
os.makedirs(RESULTS_DIR, exist_ok=True)

VULN_INFO_PATH = os.path.join(KB_DIR, "vulnerability_info.json")
RAG_CONTRACTS_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl")

for path, name in [
    (TEST_DIR, "Test set"),
    (VULN_INFO_PATH, "Vulnerability info"),
    (RAG_CONTRACTS_PATH, "RAG contracts"),
    (FT_MODEL_DIR, "Fine-tuned model"),
]:
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"{name}: {status} ({path})")

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.startswith("solana_") and f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

# === Prompts ===
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

# === Parser ===
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
        content = match.group(1).strip().lower()
    else:
        # Fallback: the model (esp. the non-fine-tuned base model on the
        # longer RAG prompt) often skips the <final> wrapper and writes the
        # verdict directly as "### Vulnerability: ...". Without this
        # fallback such responses were silently scored as "no vulnerability
        # found", inflating true negatives and killing recall.
        think_end = re.search(r"</think>", response, re.IGNORECASE)
        content = response[think_end.end():] if think_end else response
        content = content.strip().lower()
    found = [v for v in KNOWN_VULNERABILITIES if v.lower() in content]
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

# === RAG pipeline ===
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

print(f"RAG knowledge base: {len(rag_contracts)} contracts, {len(vuln_info)} vulnerability types")

instruction = "Represent the following functional description of a Solana smart contract, written in Rust:"
rag_embeddings = []
for contract in rag_contracts:
    emb = embed_model.encode([instruction, contract["description"]])
    rag_embeddings.append(emb)
rag_embeddings = np.array(rag_embeddings)
print(f"Embeddings computed: {rag_embeddings.shape}")


def search_similar_contracts(query_description, top_k=None):
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
    seen = set()
    unique_results = []
    for r in results:
        if r["vulnerability"] not in seen:
            seen.add(r["vulnerability"])
            unique_results.append(r)
    return unique_results


def build_rag_checklist(similar_contracts, num_detailed=3):
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


print("RAG pipeline ready.")

# Move embedding model to CPU to free GPU memory for the main model
embed_model = embed_model.to("cpu")
torch.cuda.empty_cache()
gc.collect()

# === Model loading ===
from unsloth import FastLanguageModel


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-32B",
        max_seq_length=6144,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FT_MODEL_DIR,
        max_seq_length=6144,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_response(model, tokenizer, messages, max_tokens=1024):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.95
    )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


print("Model loading functions ready.")

# === Evaluation & metrics ===
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


def evaluate_config(model, tokenizer, test_files, use_rag=False, config_name=""):
    results = []
    total = len(test_files)

    for idx, test_file in enumerate(test_files):
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        code = test_data["smart_contract"]
        ground_truth = test_data["vulnerability"]
        truth_display = VULN_DISPLAY.get(ground_truth, ground_truth)

        if use_rag:
            desc_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
                {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
            ]
            description = generate_response(model, tokenizer, desc_messages, max_tokens=512)
            similar = search_similar_contracts(description)
            checklist = build_rag_checklist(similar)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": USER_PROMPT_RAG.format(checklist=checklist, code=code)},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
                {"role": "user", "content": USER_PROMPT_NO_RAG.format(code=code)},
            ]

        found_vulns = []
        response = ""
        for attempt in range(3):
            response = generate_response(model, tokenizer, messages)
            success, found_vulns = parse_response(response)
            if success:
                break

        results.append({
            "file": os.path.basename(test_file),
            "ground_truth": truth_display,
            "predictions": found_vulns,
            "response": response,
        })

        pred_str = ", ".join(found_vulns) if found_vulns else "PARSE_FAIL"
        print(f"  [{idx+1}/{total}] {os.path.basename(test_file)}: "
              f"truth={truth_display} | pred={pred_str}")

    return results


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


def save_config_results(config_name, results, metrics):
    path = os.path.join(RESULTS_DIR, f"results_{config_name}.json")
    data = {"config": config_name, "metrics": metrics, "details": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


print("Evaluation functions ready.")

# === Configurations 1 & 2: Base model ===
print("Loading base model...")
model_base, tokenizer_base = load_base_model()
print("Base model loaded.\n")

print("=" * 60)
print("CONFIG 1: Qwen3-Base (no RAG)")
print("=" * 60)
start = time.time()
results_base_no_rag = evaluate_config(model_base, tokenizer_base, test_files, use_rag=False, config_name="Base-NoRAG")
metrics_base_no_rag = compute_metrics(results_base_no_rag)
save_config_results("base_no_rag", results_base_no_rag, metrics_base_no_rag)
print(f"\nConfig 1 done in {(time.time()-start)/60:.1f} min")
print(f"Results: {metrics_base_no_rag}\n")

print("=" * 60)
print("CONFIG 2: Qwen3-Base + RAG")
print("=" * 60)
start = time.time()
results_base_rag = evaluate_config(model_base, tokenizer_base, test_files, use_rag=True, config_name="Base-RAG")
metrics_base_rag = compute_metrics(results_base_rag)
save_config_results("base_rag", results_base_rag, metrics_base_rag)
print(f"\nConfig 2 done in {(time.time()-start)/60:.1f} min")
print(f"Results: {metrics_base_rag}\n")

unload_model(model_base, tokenizer_base)

# === Configurations 3 & 4: Fine-tuned model ===
print("Loading fine-tuned model...")
model_ft, tokenizer_ft = load_finetuned_model()
print("Fine-tuned model loaded.\n")

print("=" * 60)
print("CONFIG 3: Qwen3-FT (no RAG)")
print("=" * 60)
start = time.time()
results_ft_no_rag = evaluate_config(model_ft, tokenizer_ft, test_files, use_rag=False, config_name="FT-NoRAG")
metrics_ft_no_rag = compute_metrics(results_ft_no_rag)
save_config_results("ft_no_rag", results_ft_no_rag, metrics_ft_no_rag)
print(f"\nConfig 3 done in {(time.time()-start)/60:.1f} min")
print(f"Results: {metrics_ft_no_rag}\n")

print("=" * 60)
print("CONFIG 4: Qwen3-FT + RAG")
print("=" * 60)
start = time.time()
results_ft_rag = evaluate_config(model_ft, tokenizer_ft, test_files, use_rag=True, config_name="FT-RAG")
metrics_ft_rag = compute_metrics(results_ft_rag)
save_config_results("ft_rag", results_ft_rag, metrics_ft_rag)
print(f"\nConfig 4 done in {(time.time()-start)/60:.1f} min")
print(f"Results: {metrics_ft_rag}\n")

unload_model(model_ft, tokenizer_ft)

# === Summary ===
print("=" * 80)
print("QWEN3-32B EVALUATION RESULTS (RERUN)")
print("=" * 80)
for name, m in [
    ("Base (no RAG)", metrics_base_no_rag),
    ("Base + RAG", metrics_base_rag),
    ("FT (no RAG)", metrics_ft_no_rag),
    ("FT + RAG", metrics_ft_rag),
]:
    print(f"{name:20s} acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1_score']:.4f}")

detailed = {
    "model": "Qwen3-32B",
    "hardware": "NVIDIA RTX A6000 (48GB)",
    "configs": {
        "base_no_rag": {"metrics": metrics_base_no_rag},
        "base_rag": {"metrics": metrics_base_rag},
        "ft_no_rag": {"metrics": metrics_ft_no_rag},
        "ft_rag": {"metrics": metrics_ft_rag},
    },
}
with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
    json.dump(detailed, f, indent=2, ensure_ascii=False)

print("\nDone.")

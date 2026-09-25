"""
Re-evaluate the RAG configurations for Algorand/PyTeal with the SAME
redesigned static-checklist architecture applied to Solana (v5/v6): no
embedding-based top-3 retrieval, no description-generation step, no
similarity score -- just the full, static reference of all 8 vulnerability
categories, always shown, with the model deciding independently which (if
any) apply.

Only the 2 RAG configurations (Base+RAG, FT+RAG) are re-run: the no-RAG
configurations are untouched by this change, so the original official
results (results/no_rag/*) remain valid for those and are not repeated
here.

Model: same one used for the official results (Nikix999/Qwen3-32B-PyTeal-Audit,
available locally as models/Qwen3-32B-pyteal-desc/checkpoint-189).
"""

import os
import json
import re
import gc
import time
from collections import defaultdict

import torch

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
VULN_INFO_PATH = os.path.join(BASE_DIR, "vulnerability_info.json")
FT_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen3-32B-pyteal-desc", "checkpoint-189")
RESULTS_DIR = os.path.join(BASE_DIR, "results_static_rag")
os.makedirs(RESULTS_DIR, exist_ok=True)

for path, name in [(TEST_DIR, "Test set"), (VULN_INFO_PATH, "Vulnerability info"), (FT_MODEL_DIR, "FT model")]:
    print(f"{name}: {'OK' if os.path.exists(path) else 'MISSING'} ({path})")

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

with open(VULN_INFO_PATH, "r", encoding="utf-8") as f:
    VULN_INFO = json.load(f)


def build_static_checklist():
    parts = []
    for i, (key, info) in enumerate(VULN_INFO.items()):
        parts.append(
            f"\n{i+1}. **{info['name']}**\n"
            f"   - Description: {info['description']}\n"
            f"   - Applies when: {info['precondition']}\n"
            f"   - Security check to verify: {info['security_check']}\n"
        )
    return "".join(parts)


STATIC_CHECKLIST = build_static_checklist()
print(f"Static checklist built ({len(STATIC_CHECKLIST)} chars, all {len(VULN_INFO)} categories).")

# === Prompts: same structure/fields as the original Algorand pipeline
# (4-field final block including Mitigation), only the checklist source
# changes from top-3 similarity retrieval to a full static reference. ===
SYSTEM_PROMPT_RAG = """
You are an expert smart contract security auditor specialized in the Algorand blockchain and PyTeal.
Your task is to analyze PyTeal code precisely and systematically to identify security vulnerabilities,
leveraging the reference information below.

### Required behavior:
- Always perform a structured, explicit reasoning phase first and put it inside the <think> block.
  - In <think> you must:
    - Summarize the contract's purpose and high-level architecture.
    - Inspect the logic block-by-block.
    - For each reference vulnerability that seems plausibly relevant given your own reading of the code,
      check whether the code enforces the required security invariant; explain why it does or does not apply.
    - Do NOT assume a vulnerability applies just because it appears in the reference below -- most contracts
      are safe with respect to most categories. Decide independently from the code.
    - Also remain open to detecting other issues not present in the reference.

- After </think>, provide the final judgment in the <final>...</final> block using this exact structure:
  - A short summary sentence (one or two lines).
  - ### Vulnerability: <name or "None detected">
  - ### Explanation: <concise cause and how it could be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>
  - ### Mitigation: <for each identified issue, a concrete fix or recommendation>

### Constraints:
- Output must contain only the two blocks <think> and <final>, in that order.
- Do NOT include any extra commentary, greetings, or meta-text.
- If no vulnerability is found, explicitly write "### Vulnerability: Not Vulnerable" and "### Risk: No significant risk identified".
""".strip()

USER_PROMPT_RAG = """
### Vulnerability reference (documentation, not a prediction for this contract)
{checklist}

Now perform a detailed security analysis of the following PyTeal smart contract, deciding independently
which (if any) of the categories above actually apply.

Contract Code:

{code}
""".strip()

print("Prompts defined.")

# === Parser: identical to rag.py's original parse_response (not tag-based,
# searches for "### vulnerability" directly -- this is why Algorand's
# pipeline never hit the <final>-missing bug that affected Solana). ===
VULNS = ["Arbitrary delete", "Arbitrary Application Deletion", "Arbitrary Application update",
         "Arbitrary update", "Unchecked Asset Close To", "Unchecked Close Remainder To",
         "Unchecked Rekey to", "Unchecked Rekey operation", "Unchecked Transaction Fee",
         "Unchecked Asset Receiver", "Unchecked AssetReceiver", "Unchecked Payment Receiver",
         "Not Vulnerable", "No security risk", "No vulnerabilities"]


def parse_response(response: str):
    pattern = re.compile(r"###\s*vulnerability\s*:?\s*\n?\s*(.+)", re.IGNORECASE)
    match = pattern.search(response.lower())
    vulnerabilities = []
    if match:
        vulns = match.group(1)
        for v in VULNS:
            if v.lower() in vulns:
                vulnerabilities.append(v)
    return vulnerabilities


print("Parser ready.")

from unsloth import FastLanguageModel


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-32B", max_seq_length=8192, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FT_MODEL_DIR, max_seq_length=8192, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_response(model, tokenizer, messages, max_tokens=1536):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.95)
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


print("Model loading functions ready.")


def evaluate_rag(model, tokenizer, test_files, config_name=""):
    results = []
    for idx, test_file in enumerate(test_files):
        with open(test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        code = content["smart_contract"]
        ground_truth = content["vulnerability"]

        messages = [
            {"role": "developer", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": USER_PROMPT_RAG.format(checklist=STATIC_CHECKLIST, code=code)},
        ]
        response = generate_response(model, tokenizer, messages)
        vulnerabilities = parse_response(response)

        results.append({
            "file": os.path.basename(test_file),
            "ground_truth": ground_truth,
            "predictions": vulnerabilities,
            "response": response,
        })
        print(f"  [{idx+1}/{len(test_files)}] {os.path.basename(test_file)}: "
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
        truth_raw = r["ground_truth"]
        truth = MAP_TRUTH_TO_DISPLAY.get(truth_raw, truth_raw)
        preds = set(r["predictions"])
        preds_is_safe = len(preds) == 0 or preds.issubset(SAFE_LABELS)

        if truth == "Not Vulnerable":
            if preds_is_safe:
                tn += 1
            else:
                real_preds = preds - SAFE_LABELS
                fp += len(real_preds)
                for p in real_preds:
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


def save_results(config_name, results, metrics):
    path = os.path.join(RESULTS_DIR, f"results_{config_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": config_name, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


print("Loading base model...")
model_base, tok_base = load_base_model()
print("Base model loaded.\n")

print("=" * 60)
print("CONFIG: Qwen3-Base + RAG (static reference)")
print("=" * 60)
start = time.time()
results_base_rag = evaluate_rag(model_base, tok_base, test_files, "base_rag")
metrics_base_rag = compute_metrics(results_base_rag)
save_results("base_rag", results_base_rag, metrics_base_rag)
print(f"Done in {(time.time()-start)/60:.1f} min. Results: {metrics_base_rag}\n")

unload_model(model_base, tok_base)

print("Loading fine-tuned model (Qwen3-32B-PyTeal-Audit)...")
model_ft, tok_ft = load_finetuned_model()
print("FT model loaded.\n")

print("=" * 60)
print("CONFIG: Qwen3-FT + RAG (static reference)")
print("=" * 60)
start = time.time()
results_ft_rag = evaluate_rag(model_ft, tok_ft, test_files, "ft_rag")
metrics_ft_rag = compute_metrics(results_ft_rag)
save_results("ft_rag", results_ft_rag, metrics_ft_rag)
print(f"Done in {(time.time()-start)/60:.1f} min. Results: {metrics_ft_rag}\n")

unload_model(model_ft, tok_ft)

print("=" * 80)
print("ALGORAND -- STATIC RAG RE-EVALUATION SUMMARY")
print("=" * 80)
print(f"Base + RAG (static) acc={metrics_base_rag['accuracy']:.4f} f1={metrics_base_rag['f1_score']:.4f}")
print(f"FT + RAG (static)   acc={metrics_ft_rag['accuracy']:.4f} f1={metrics_ft_rag['f1_score']:.4f}")
print("\nOriginal official results (top-3 similarity retrieval), for comparison:")
print("Base + RAG: F1=0.7536 (P=0.8966 R=0.65 Acc=0.6458)")
print("FT + RAG:   F1=0.8800 (P=0.9429 R=0.825 Acc=0.8085)")
print("\nDone.")

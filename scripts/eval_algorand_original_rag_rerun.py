"""
Faithful re-run of the ORIGINAL Algorand RAG pipeline (3-item checklist via
embedding similarity retrieval, exact prompts from test.ipynb), to isolate
ONE variable at a time: same model, same checklist format/size the model
was actually trained on, just a fresh sampling run -- used as a sanity
check / reproducibility baseline before testing any redesign.

Runs both Base+RAG and FT+RAG, using the existing precomputed
contract_embeddings.json knowledge base and rag.py's own
search_similarity/create_rag_checklist functions.
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
RESULTS_DIR = os.path.join(BASE_DIR, "results_original_rag_rerun")
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.insert(0, BASE_DIR)
from rag import search_similarity, parse_response, create_rag_checklist  # noqa: E402
from prompts import dev_description_prompt, user_description_prompt  # noqa: E402

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

# === Exact original prompts from test.ipynb cell 4 ===
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


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-32B", max_seq_length=4096, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FT_MODEL_DIR, max_seq_length=4096, dtype=None, load_in_4bit=True,
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


def evaluate_rag(model, tokenizer, test_files):
    results = []
    for idx, test_file in enumerate(test_files):
        with open(test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        code = content["smart_contract"]
        ground_truth = content["vulnerability"]

        desc_messages = [
            {"role": "developer", "content": dev_description_prompt},
            {"role": "user", "content": user_description_prompt.format(code=code)},
        ]
        description = generate_response(model, tokenizer, desc_messages, max_tokens=512)

        rag_contracts = search_similarity(description)
        checklist = create_rag_checklist(rag_contracts, with_few_shot=True)

        messages = [
            {"role": "developer", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": USER_PROMPT_RAG.format(checklist=checklist, code=code)},
        ]
        response = generate_response(model, tokenizer, messages)
        vulnerabilities = parse_response(response)

        results.append({
            "file": os.path.basename(test_file),
            "ground_truth": ground_truth,
            "predictions": vulnerabilities,
            "checklist": checklist,
            "response": response,
        })
        print(f"  [{idx+1}/{len(test_files)}] {os.path.basename(test_file)}: "
              f"truth={ground_truth} | pred={vulnerabilities}")
        print(f"    --- DESCRIPTION ---\n{description}\n")
        print(f"    --- CHECKLIST ---\n{checklist}\n")
        print(f"    --- RAW RESPONSE ---\n{response}\n")
        print("    " + "=" * 70)
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


def save_results(config_name, results, metrics):
    path = os.path.join(RESULTS_DIR, f"results_{config_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": config_name, "metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


print("Loading base model...")
model_base, tok_base = load_base_model()
print("Base model loaded.\n")

print("=" * 60)
print("CONFIG: Qwen3-Base + RAG (original 3-item similarity retrieval)")
print("=" * 60)
start = time.time()
results_base_rag = evaluate_rag(model_base, tok_base, test_files)
metrics_base_rag = compute_metrics(results_base_rag)
save_results("base_rag", results_base_rag, metrics_base_rag)
print(f"Done in {(time.time()-start)/60:.1f} min. Results: {metrics_base_rag}\n")

unload_model(model_base, tok_base)

print("Loading fine-tuned model (Qwen3-32B-PyTeal-Audit)...")
model_ft, tok_ft = load_finetuned_model()
print("FT model loaded.\n")

print("=" * 60)
print("CONFIG: Qwen3-FT + RAG (original 3-item similarity retrieval)")
print("=" * 60)
start = time.time()
results_ft_rag = evaluate_rag(model_ft, tok_ft, test_files)
metrics_ft_rag = compute_metrics(results_ft_rag)
save_results("ft_rag", results_ft_rag, metrics_ft_rag)
print(f"Done in {(time.time()-start)/60:.1f} min. Results: {metrics_ft_rag}\n")

unload_model(model_ft, tok_ft)

print("=" * 80)
print("ALGORAND -- ORIGINAL 3-ITEM RAG RE-RUN (reproducibility check)")
print("=" * 80)
print(f"Base + RAG acc={metrics_base_rag['accuracy']:.4f} f1={metrics_base_rag['f1_score']:.4f}")
print(f"FT + RAG   acc={metrics_ft_rag['accuracy']:.4f} f1={metrics_ft_rag['f1_score']:.4f}")
print("\nOfficial results, for comparison:")
print("Base + RAG: F1=0.7536 (P=0.8966 R=0.65 Acc=0.6458)")
print("FT + RAG:   F1=0.8800 (P=0.9429 R=0.825 Acc=0.8085)")
print("\nDone.")

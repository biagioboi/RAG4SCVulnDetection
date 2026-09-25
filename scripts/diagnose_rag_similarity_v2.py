"""
Re-run of diagnose_rag_similarity.py, but measuring the thing that actually
matters for the pipeline: does the TOP-1 retrieved KB category match the
contract's true vulnerability category? (The original script only reported
raw similarity magnitudes/spread, not whether retrieval was actually
*correct*.)

Compares OLD (data/knowledge_base/rag_contracts.jsonl.bak, pre-fix generic
boilerplate descriptions) vs NEW (rag_contracts.jsonl, current, category-led
descriptions after scripts/fix_rag_contracts_descriptions.py) using the
SAME 81 generated contract descriptions (one GPU pass, FT v11 model), so the
only variable is the KB description text.
"""

import os
import json
import numpy as np

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
OLD_KB_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl.bak")
NEW_KB_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v11")
OUT_PATH = os.path.join(REPO_DIR, "results", "rag_similarity_diagnosis_v2.json")

GT_MAP = {"integer_flow": "integer_overflow", "cpi": "cpi_reentrancy"}

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.startswith("solana_") and f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

SYSTEM_PROMPT_DESCRIPTION = """You are an expert smart contract security auditor specialized in Solana and Rust.
Describe the given Solana program with exactly these 3 sections:

### Contract type:
State whether this is a "Solana program instruction handler".

### Contract Purpose
One short paragraph stating the primary objective.

### Functional Description
Step-by-step description of the execution logic. Be explicit about which accounts are validated and how.
Do not include variable names or code syntax.""".strip()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading embedding model...")
embed_model = SentenceTransformer("hkunlp/instructor-xl")
instruction = "Represent the following functional description of a Solana smart contract, written in Rust:"


def load_kb(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    embs = np.array([embed_model.encode([instruction, r["description"]]) for r in rows])
    return rows, embs


print("Embedding OLD knowledge base...")
old_rows, old_embs = load_kb(OLD_KB_PATH)
print("Embedding NEW knowledge base...")
new_rows, new_embs = load_kb(NEW_KB_PATH)

# Move embedding model to CPU to free GPU memory for the 32B model
embed_model = embed_model.to("cpu")
import torch, gc
torch.cuda.empty_cache()
gc.collect()

from unsloth import FastLanguageModel

print("Loading FT v11 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FT_MODEL_DIR, max_seq_length=8192, dtype=None, load_in_4bit=True,
    device_map={"": 0},
)
FastLanguageModel.for_inference(model)
print("Model loaded.")


def generate_response(messages, max_tokens=512):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.95
    )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


def top_unique(rows, embs, query_desc):
    query_emb = embed_model.encode([instruction, query_desc]).reshape(1, -1)
    results = []
    for i, emb in enumerate(embs):
        sim = float(cosine_similarity(emb.reshape(1, -1), query_emb)[0][0])
        results.append({"vulnerability": rows[i]["vulnerability"], "similarity": sim})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    seen = set()
    unique = []
    for r in results:
        if r["vulnerability"] not in seen:
            seen.add(r["vulnerability"])
            unique.append(r)
    return unique


records = []
old_top1_correct = 0
new_top1_correct = 0
non_safe_total = 0

for idx, test_file in enumerate(test_files):
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    code = test_data["smart_contract"]
    ground_truth = GT_MAP.get(test_data["vulnerability"], test_data["vulnerability"])

    if ground_truth == "not_vulnerable":
        continue  # KB only has VULNERABLE reference entries; only score on vulnerable contracts
    non_safe_total += 1

    desc_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
        {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
    ]
    description = generate_response(desc_messages)

    old_top = top_unique(old_rows, old_embs, description)
    new_top = top_unique(new_rows, new_embs, description)

    old_hit = old_top[0]["vulnerability"] == ground_truth
    new_hit = new_top[0]["vulnerability"] == ground_truth
    old_top1_correct += int(old_hit)
    new_top1_correct += int(new_hit)

    records.append({
        "file": os.path.basename(test_file),
        "ground_truth": ground_truth,
        "old_top1": old_top[0]["vulnerability"], "old_top1_sim": old_top[0]["similarity"], "old_hit": old_hit,
        "new_top1": new_top[0]["vulnerability"], "new_top1_sim": new_top[0]["similarity"], "new_hit": new_hit,
    })
    print(f"[{idx+1}/{len(test_files)}] {os.path.basename(test_file)} (truth={ground_truth}): "
          f"OLD top1={old_top[0]['vulnerability']}({old_top[0]['similarity']:.3f}){'OK' if old_hit else 'X'} | "
          f"NEW top1={new_top[0]['vulnerability']}({new_top[0]['similarity']:.3f}){'OK' if new_hit else 'X'}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "old_top1_accuracy": old_top1_correct / non_safe_total,
        "new_top1_accuracy": new_top1_correct / non_safe_total,
        "n": non_safe_total,
        "records": records,
    }, f, indent=2, ensure_ascii=False)

print("\n=== SUMMARY (top-1 retrieval accuracy on vulnerable contracts) ===")
print(f"n = {non_safe_total}")
print(f"OLD (boilerplate descriptions): {old_top1_correct}/{non_safe_total} = {old_top1_correct/non_safe_total:.1%}")
print(f"NEW (category-led descriptions): {new_top1_correct}/{non_safe_total} = {new_top1_correct/non_safe_total:.1%}")
print(f"Saved to {OUT_PATH}")

"""
Diagnostic: measure the actual RAG retrieval similarity scores for all 59
Solana test contracts, to check whether low-similarity ("not really
relevant") knowledge-base entries are still being injected into the
checklist unconditionally (no minimum-similarity threshold in the current
pipeline).

Uses the BASE Qwen3-32B model (not fine-tuned) to generate descriptions,
since that is what both the Base+RAG and FT+RAG configurations use for the
description step in the existing eval scripts.
"""

import os
import json
import numpy as np

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
VULN_INFO_PATH = os.path.join(KB_DIR, "vulnerability_info.json")
RAG_CONTRACTS_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl")
OUT_PATH = os.path.join(REPO_DIR, "results", "rag_similarity_diagnosis.json")

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
print(f"KB embeddings computed: {rag_embeddings.shape}")

import torch
from unsloth import FastLanguageModel

print("Loading base Qwen3-32B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-32B",
    max_seq_length=6144,
    dtype=None,
    load_in_4bit=True,
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


def search_similar(query_description):
    query_emb = embed_model.encode([instruction, query_description]).reshape(1, -1)
    results = []
    for i, emb in enumerate(rag_embeddings):
        sim = float(cosine_similarity(emb.reshape(1, -1), query_emb)[0][0])
        results.append({"vulnerability": rag_contracts[i]["vulnerability"], "similarity": sim})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    seen = set()
    unique = []
    for r in results:
        if r["vulnerability"] not in seen:
            seen.add(r["vulnerability"])
            unique.append(r)
    return unique


records = []
for idx, test_file in enumerate(test_files):
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    code = test_data["smart_contract"]
    ground_truth = test_data["vulnerability"]

    desc_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
        {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
    ]
    description = generate_response(desc_messages)
    similar = search_similar(description)
    top3 = similar[:3]

    record = {
        "file": os.path.basename(test_file),
        "ground_truth": ground_truth,
        "top3": top3,
    }
    records.append(record)
    top3_str = ", ".join(f"{r['vulnerability']}={r['similarity']:.3f}" for r in top3)
    print(f"[{idx+1}/{len(test_files)}] {os.path.basename(test_file)} (truth={ground_truth}): {top3_str}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT_PATH}")

# Summary stats
top1_sims = [r["top3"][0]["similarity"] for r in records]
top3_sims = [s["similarity"] for r in records for s in r["top3"]]
not_vuln_top1 = [r["top3"][0]["similarity"] for r in records if r["ground_truth"] == "not_vulnerable"]
vuln_top1 = [r["top3"][0]["similarity"] for r in records if r["ground_truth"] != "not_vulnerable"]

print("\n=== SUMMARY ===")
print(f"Top-1 similarity: min={min(top1_sims):.3f} max={max(top1_sims):.3f} mean={np.mean(top1_sims):.3f}")
print(f"All top-3 similarities: min={min(top3_sims):.3f} max={max(top3_sims):.3f} mean={np.mean(top3_sims):.3f}")
print(f"Top-1 similarity on 'not_vulnerable' contracts (n={len(not_vuln_top1)}): mean={np.mean(not_vuln_top1):.3f}")
print(f"Top-1 similarity on vulnerable contracts (n={len(vuln_top1)}): mean={np.mean(vuln_top1):.3f}")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    below = sum(1 for s in top1_sims if s < thresh)
    print(f"Contracts where top-1 similarity < {thresh}: {below}/{len(top1_sims)}")

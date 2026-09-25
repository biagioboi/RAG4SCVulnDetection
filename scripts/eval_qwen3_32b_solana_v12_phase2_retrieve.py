"""
v12 phase 2/3: retrieval + checklist construction using Qwen3-Embedding-4B.

Deliberately does NOT import unsloth (see phase1 docstring for why). Loads
the descriptions saved by phase 1, embeds the RAG knowledge base and every
description with Qwen3-Embedding-4B (same plain `.encode()` call validated
in scripts/diagnose_rag_embedding_models_clean.py -- no special query/
document prompt, since that symmetric setup is what was actually measured
at 37.5% top-1 accuracy), and builds the num_detailed=2 checklist for each
contract using the exact same build_rag_checklist logic as the existing
eval scripts (so the only pipeline change is the embedding model).
"""

import os
import json

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "qwen3-32b-v12")

VULN_INFO_PATH = os.path.join(KB_DIR, "vulnerability_info.json")
RAG_CONTRACTS_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl")
DESCRIPTIONS_PATH = os.path.join(RESULTS_DIR, "descriptions.json")
OUT_PATH = os.path.join(RESULTS_DIR, "checklists_k2.json")

NUM_DETAILED = 2  # k=2, the best config found this session

descriptions = json.load(open(DESCRIPTIONS_PATH, "r", encoding="utf-8"))
print(f"Loaded {len(descriptions)} descriptions from {DESCRIPTIONS_PATH}")

with open(VULN_INFO_PATH, "r", encoding="utf-8") as f:
    vuln_info = json.load(f)

rag_contracts = []
with open(RAG_CONTRACTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rag_contracts.append(json.loads(line))
print(f"RAG knowledge base: {len(rag_contracts)} contracts, {len(vuln_info)} vulnerability types")

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Qwen3-Embedding-4B...")
embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", trust_remote_code=True)
print("Embedding model loaded.")

rag_embeddings = np.array([embed_model.encode(c["description"]) for c in rag_contracts])
print(f"KB embeddings computed: {rag_embeddings.shape}")


def search_similar_contracts(query_description):
    query_emb = embed_model.encode(query_description).reshape(1, -1)
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


def build_rag_checklist(similar_contracts, num_detailed=NUM_DETAILED):
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


records = []
for idx, d in enumerate(descriptions):
    similar = search_similar_contracts(d["description"])
    checklist = build_rag_checklist(similar)
    records.append({
        "file": d["file"],
        "ground_truth": d["ground_truth"],
        "code": d["code"],
        "checklist": checklist,
        "top_categories": [s["vulnerability"] for s in similar[:3]],
    })
    print(f"  [{idx+1}/{len(descriptions)}] {d['file']} (truth={d['ground_truth']}): "
          f"top3={[s['vulnerability'] for s in similar[:3]]}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(records)} checklists to {OUT_PATH}")

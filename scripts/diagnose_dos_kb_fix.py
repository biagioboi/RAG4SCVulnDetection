"""
Quick targeted check (6 contracts only, cheap): does adding the
reinitialization-guard DoS reference entry (dos-VULNERABLE-05) improve the
RAG retrieval rank of "dos" specifically for the 6 DoS-truth contracts in
the test set? Compares OLD KB (rag_contracts.jsonl.bak2, 5 dos entries) vs
NEW KB (rag_contracts.jsonl, 6 dos entries) using the same freshly generated
v14 description per contract.
"""

import os
import json
import gc

import torch
import numpy as np

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
KB_DIR = os.path.join(REPO_DIR, "data", "knowledge_base")
OLD_KB_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl.bak2")
NEW_KB_PATH = os.path.join(KB_DIR, "rag_contracts.jsonl")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v14")

DOS_FILES = ["solana_04.json", "solana_34.json", "solana_39.json",
             "solana_46.json", "solana_70.json", "solana_84.json"]

SYSTEM_PROMPT_DESCRIPTION = """You are an expert smart contract security auditor specialized in Solana and Rust.
Describe the given Solana program with exactly these 3 sections:

### Contract type:
State whether this is a "Solana program instruction handler".

### Contract Purpose
One short paragraph stating the primary objective.

### Functional Description
Step-by-step description of the execution logic. Be explicit about which accounts are validated and how.
Do not include variable names or code syntax.""".strip()

from unsloth import FastLanguageModel

print("Loading FT v14 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FT_MODEL_DIR, max_seq_length=8192, dtype=None, load_in_4bit=True,
    device_map={"": 0},
)
FastLanguageModel.for_inference(model)


def generate_response(messages, max_tokens=512):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.95
    )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


descriptions = []
for fname in DOS_FILES:
    data = json.load(open(os.path.join(TEST_DIR, fname), "r", encoding="utf-8"))
    code = data["smart_contract"]
    desc_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
        {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
    ]
    description = generate_response(desc_messages)
    descriptions.append({"file": fname, "description": description})
    print(f"  {fname}: description generated")

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading instructor-xl...")
embed_model = SentenceTransformer("hkunlp/instructor-xl")
instruction = "Represent the following functional description of a Solana smart contract, written in Rust:"


def load_kb(path):
    rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    embs = np.array([embed_model.encode([instruction, r["description"]]) for r in rows])
    return rows, embs


def dos_rank(rows, embs, query_desc):
    query_emb = embed_model.encode([instruction, query_desc]).reshape(1, -1)
    sims = []
    for i, emb in enumerate(embs):
        sim = float(cosine_similarity(emb.reshape(1, -1), query_emb)[0][0])
        sims.append({"vulnerability": rows[i]["vulnerability"], "similarity": sim})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    seen, unique = set(), []
    for s in sims:
        if s["vulnerability"] not in seen:
            seen.add(s["vulnerability"])
            unique.append(s)
    for rank, s in enumerate(unique, start=1):
        if s["vulnerability"] == "dos":
            return rank, s["similarity"]
    return None, None


old_rows, old_embs = load_kb(OLD_KB_PATH)
new_rows, new_embs = load_kb(NEW_KB_PATH)

print("\n=== DoS rank comparison (1 = top, best) ===")
for d in descriptions:
    old_r, old_s = dos_rank(old_rows, old_embs, d["description"])
    new_r, new_s = dos_rank(new_rows, new_embs, d["description"])
    print(f"  {d['file']}: OLD dos_rank={old_r} (sim={old_s:.4f})  ->  NEW dos_rank={new_r} (sim={new_s:.4f})")

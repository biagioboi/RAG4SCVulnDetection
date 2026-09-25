"""
Compare embedding backends for RAG retrieval accuracy on the Solana KB,
holding everything else fixed: same 40 vulnerable test contracts, same
generated query descriptions (saved explicitly so this run is reproducible
and the raw data isn't lost), same (original, pre-"fix") rag_contracts.jsonl
KB descriptions. Only the embedding model changes.

Motivation: diagnose_rag_similarity_v2.py found that instructor-xl gives
near-uniform similarity scores and only 37.5% top-1 retrieval accuracy
(15/40) on the ORIGINAL KB descriptions, and rewriting the KB descriptions
to remove shared boilerplate did not help (30.0%, likely noise-level or
worse). instructor-xl is a 2022-era embedding model; this script tests
whether a newer, code-capable embedding model (Qwen3-Embedding-4B) actually
discriminates these security-vulnerability descriptions better.

All intermediate artifacts (descriptions, per-contract top-1 results for
each embedding backend) are saved to results/rag_embedding_model_comparison/
-- nothing here is discarded, per explicit instruction not to drop results
that might matter later.
"""

import os
import json
import gc

import torch
import numpy as np

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
KB_PATH = os.path.join(REPO_DIR, "data", "knowledge_base", "rag_contracts.jsonl")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v11")
OUT_DIR = os.path.join(REPO_DIR, "results", "rag_embedding_model_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

DESCRIPTIONS_PATH = os.path.join(OUT_DIR, "query_descriptions.json")
COMPARISON_PATH = os.path.join(OUT_DIR, "comparison_results.json")

GT_MAP = {"integer_flow": "integer_overflow", "cpi": "cpi_reentrancy"}

SYSTEM_PROMPT_DESCRIPTION = """You are an expert smart contract security auditor specialized in Solana and Rust.
Describe the given Solana program with exactly these 3 sections:

### Contract type:
State whether this is a "Solana program instruction handler".

### Contract Purpose
One short paragraph stating the primary objective.

### Functional Description
Step-by-step description of the execution logic. Be explicit about which accounts are validated and how.
Do not include variable names or code syntax.""".strip()


def get_vulnerable_test_files():
    files = sorted([
        os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
        if f.startswith("solana_") and f.endswith(".json")
    ])
    out = []
    for f in files:
        data = json.load(open(f, "r", encoding="utf-8"))
        truth = GT_MAP.get(data["vulnerability"], data["vulnerability"])
        if truth != "not_vulnerable":
            out.append((os.path.basename(f), truth, data["smart_contract"]))
    return out


def generate_descriptions():
    if os.path.exists(DESCRIPTIONS_PATH):
        print(f"Reusing existing descriptions from {DESCRIPTIONS_PATH}")
        return json.load(open(DESCRIPTIONS_PATH, "r", encoding="utf-8"))

    from unsloth import FastLanguageModel

    print("Loading FT v11 model to generate query descriptions...")
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

    records = []
    for idx, (fname, truth, code) in enumerate(get_vulnerable_test_files()):
        desc_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
            {"role": "user", "content": f"Describe this Solana contract:\n\n```rust\n{code}\n```"},
        ]
        description = generate_response(desc_messages)
        records.append({"file": fname, "ground_truth": truth, "description": description})
        print(f"  [{idx+1}] {fname} (truth={truth}) description generated ({len(description)} chars)")

    with open(DESCRIPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} descriptions to {DESCRIPTIONS_PATH}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return records


def evaluate_embedding_model(model_name, query_records, kb_rows):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"\nLoading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, trust_remote_code=True)

    is_instructor = "instructor" in model_name.lower()
    instruction = "Represent the following functional description of a Solana smart contract, written in Rust:"

    def encode(text):
        if is_instructor:
            return embed_model.encode([instruction, text])
        return embed_model.encode(text)

    kb_embs = np.array([encode(r["description"]) for r in kb_rows])

    results = []
    correct = 0
    for r in query_records:
        query_emb = encode(r["description"]).reshape(1, -1)
        sims = []
        for i, emb in enumerate(kb_embs):
            sim = float(cosine_similarity(emb.reshape(1, -1), query_emb)[0][0])
            sims.append({"vulnerability": kb_rows[i]["vulnerability"], "similarity": sim})
        sims.sort(key=lambda x: x["similarity"], reverse=True)
        seen = set()
        unique = []
        for s in sims:
            if s["vulnerability"] not in seen:
                seen.add(s["vulnerability"])
                unique.append(s)

        top1 = unique[0]
        hit = top1["vulnerability"] == r["ground_truth"]
        correct += int(hit)
        results.append({
            "file": r["file"], "ground_truth": r["ground_truth"],
            "top1": top1["vulnerability"], "top1_sim": top1["similarity"], "hit": hit,
            "ranked": unique,
        })

    accuracy = correct / len(query_records)
    print(f"{model_name}: top-1 accuracy = {correct}/{len(query_records)} = {accuracy:.1%}")

    del embed_model
    gc.collect()
    torch.cuda.empty_cache()

    return {"model": model_name, "accuracy": accuracy, "correct": correct, "n": len(query_records), "details": results}


def main():
    query_records = generate_descriptions()

    kb_rows = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            kb_rows.append(json.loads(line))
    print(f"KB entries: {len(kb_rows)}")

    all_results = {}
    for model_name in ["hkunlp/instructor-xl", "Qwen/Qwen3-Embedding-4B"]:
        try:
            all_results[model_name] = evaluate_embedding_model(model_name, query_records, kb_rows)
        except Exception as e:
            print(f"FAILED for {model_name}: {e}")
            all_results[model_name] = {"model": model_name, "error": str(e)}

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL COMPARISON ===")
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: ERROR - {res['error']}")
        else:
            print(f"{name}: {res['correct']}/{res['n']} = {res['accuracy']:.1%}")
    print(f"\nFull results saved to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()

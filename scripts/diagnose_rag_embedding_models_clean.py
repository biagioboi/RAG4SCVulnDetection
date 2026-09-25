"""
Retry of diagnose_rag_embedding_models.py's embedding-model comparison, in a
process that never imports unsloth. The combined run failed for
Qwen/Qwen3-Embedding-4B with `'Qwen3Attention' object has no attribute
'apply_qkv'` -- Unsloth monkey-patches the Qwen3Attention class globally
(to speed up our fine-tuned Qwen3-32B LLM) as a side effect of being
imported, and that patch is incompatible with the separate Qwen3-based
embedding model loaded in the same process.

Reuses the 40 query descriptions already generated and saved by
diagnose_rag_embedding_models.py (results/rag_embedding_model_comparison/
query_descriptions.json) -- no LLM generation needed here, so no reason to
import unsloth at all. Re-tests instructor-xl too, for a clean apples-to-
apples baseline in the same unsloth-free process.
"""

import os
import json
import gc

import torch
import numpy as np

REPO_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/LLM-Contract-Analyzer"
KB_PATH = os.path.join(REPO_DIR, "data", "knowledge_base", "rag_contracts.jsonl")
OUT_DIR = os.path.join(REPO_DIR, "results", "rag_embedding_model_comparison")
DESCRIPTIONS_PATH = os.path.join(OUT_DIR, "query_descriptions.json")
COMPARISON_PATH = os.path.join(OUT_DIR, "comparison_results_clean.json")


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
        print(f"  {r['file']} (truth={r['ground_truth']}): top1={top1['vulnerability']}({top1['similarity']:.3f}) "
              f"{'OK' if hit else 'X'}")

    accuracy = correct / len(query_records)
    print(f"{model_name}: top-1 accuracy = {correct}/{len(query_records)} = {accuracy:.1%}")

    del embed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"model": model_name, "accuracy": accuracy, "correct": correct, "n": len(query_records), "details": results}


def main():
    if not os.path.exists(DESCRIPTIONS_PATH):
        raise FileNotFoundError(
            f"{DESCRIPTIONS_PATH} not found -- run diagnose_rag_embedding_models.py first "
            "to generate and save the query descriptions."
        )
    query_records = json.load(open(DESCRIPTIONS_PATH, "r", encoding="utf-8"))
    print(f"Loaded {len(query_records)} saved query descriptions from {DESCRIPTIONS_PATH}")

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

    print("\n=== FINAL COMPARISON (unsloth-free process) ===")
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: ERROR - {res['error']}")
        else:
            print(f"{name}: {res['correct']}/{res['n']} = {res['accuracy']:.1%}")
    print(f"\nFull results saved to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()

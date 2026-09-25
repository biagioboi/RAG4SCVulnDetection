"""
v12 phase 1/3: generate the contract descriptions used for RAG retrieval.

Split into 3 phases (this file + phase2 + phase3) because Qwen3-Embedding-4B
(the replacement for instructor-xl, validated to give 37.5% vs 30.0% top-1
retrieval accuracy on identical data -- see
scripts/diagnose_rag_embedding_models_clean.py) crashes with
`'Qwen3Attention' object has no attribute 'apply_qkv'` when loaded in the
same process as Unsloth (Unsloth monkey-patches the Qwen3Attention class
globally for our fine-tuned Qwen3-32B LLM, and that patch is incompatible
with a second, differently-initialized Qwen3-based model in the same
process). The fix is to never have Unsloth and Qwen3-Embedding loaded in the
same Python process:

  phase 1 (this file, Unsloth): generate a description for every test
      contract with the FT model.
  phase 2 (no Unsloth import): embed the KB and the descriptions with
      Qwen3-Embedding-4B, retrieve, build the RAG checklists.
  phase 3 (Unsloth again): run the actual RAG-audit generation using the
      precomputed checklists, parse, score.

Descriptions for ALL 81 test contracts are generated here (not just the 40
vulnerable ones used in the embedding-model diagnostic), since the real
pipeline builds a checklist for every contract, safe or not.
"""

import os
import json
import time

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
TEST_DIR = os.path.join(REPO_DIR, "data", "test_set")
FT_MODEL_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v11")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "qwen3-32b-v12")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_PATH = os.path.join(RESULTS_DIR, "descriptions.json")

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


records = []
start = time.time()
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

    records.append({
        "file": os.path.basename(test_file),
        "ground_truth": ground_truth,
        "code": code,
        "description": description,
    })
    print(f"  [{idx+1}/{len(test_files)}] {os.path.basename(test_file)} (truth={ground_truth}) "
          f"description generated ({len(description)} chars)")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\nDone in {(time.time()-start)/60:.1f} min")
print(f"Saved {len(records)} descriptions to {OUT_PATH}")

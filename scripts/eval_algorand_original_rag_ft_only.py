"""
Re-run of ONLY the FT+RAG configuration (Base+RAG already completed and
saved). Forces explicit single-GPU device_map to avoid the "modules
dispatched to CPU/disk" error caused by another process (ollama) holding
~20GB of VRAM concurrently.
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
import rag  # noqa: E402
from rag import search_similarity, parse_response, create_rag_checklist  # noqa: E402
from prompts import dev_description_prompt, user_description_prompt  # noqa: E402

# Free GPU memory: another process (ollama) is concurrently using ~20GB,
# and the embedding model isn't needed on GPU (only used between generation
# calls, not during them).
rag.model = rag.model.to("cpu")
torch.cuda.empty_cache()
gc.collect()
print("Moved instructor-xl embedding model to CPU to free GPU memory.")

test_files = sorted([
    os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
    if f.endswith(".json")
])
print(f"Test contracts found: {len(test_files)}")

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


def load_finetuned_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FT_MODEL_DIR, max_seq_length=4096, dtype=None, load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_tokens=1536):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.95)
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


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
        print(f"    --- RAW RESPONSE ---\n{response}\n")
        print("    " + "=" * 70)
    return results


def save_results(config_name, results):
    path = os.path.join(RESULTS_DIR, f"results_{config_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": config_name, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


print("Loading fine-tuned model (Qwen3-32B-PyTeal-Audit)...")
model_ft, tok_ft = load_finetuned_model()
print("FT model loaded.\n")

print("=" * 60)
print("CONFIG: Qwen3-FT + RAG (original 3-item similarity retrieval)")
print("=" * 60)
start = time.time()
results_ft_rag = evaluate_rag(model_ft, tok_ft, test_files)
save_results("ft_rag", results_ft_rag)
print(f"Done in {(time.time()-start)/60:.1f} min.")
print("\nDone.")

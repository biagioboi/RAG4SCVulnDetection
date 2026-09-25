"""
Step 5d: SOTA generalist baseline benchmark WITH RAG (Solana).
Mirrors the DeepSeek-V3.2 + RAG configuration used in the Algorand study
(Tortora, 2025), to enable a like-for-like SOTA-vs-specialist comparison
across chains under retrieval augmentation.

No fine-tuning is applied to DeepSeek: only the retrieval/checklist
pipeline (Section "Retrieval-Augmented Generation Pipeline") is added on
top of the same in-context model used in run_benchmark_deepseek_no_rag.py.

Prerequisite: contract_embeddings.json (Solana knowledge base embeddings)
must already exist, as produced by src/rag/generate_embeddings.py.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    export DEEPSEEK_MODEL="deepseek-chat"   # match the Algorand benchmark's
                                             # exact model id/snapshot
    python run_benchmark_deepseek_rag.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.clients import get_client
from utils.test_logs import create_test_log_dir, create_log_file, write_log, write_log_summary
from utils.prompts import (
    system_prompt_description, user_prompt_description,
    system_prompt_rag, user_prompt_rag,
)
from utils.parser import parse_response
from rag.rag import search_similarity, create_rag_checklist

# === CONFIG ===
model_name = "DeepSeek-V3.2"

config_model = {
    "model_name": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    "type": "api",
    "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "api_key": os.environ["DEEPSEEK_API_KEY"],  # raises KeyError if unset, on purpose
}

client = get_client(config_model)

test_dir = "../../data/test_set"
test_logs_basedir = "../../results/test_logs/rag"

test_logs_dir = create_test_log_dir(test_logs_basedir, model_name)

for file in sorted(os.listdir(test_dir)):
    if not file.endswith(".json"):
        continue

    full_path = os.path.join(test_dir, file)

    with open(full_path, "r", encoding="utf-8") as f:
        content = json.load(f)
        code = content["smart_contract"]
        vulnerability = content["vulnerability"]

    log_file = create_log_file(test_logs_dir, file, vulnerability)
    write_log_summary(
        test_logs_dir,
        f"{'-' * 50}ANALYZING {file}-{vulnerability}{'-' * 50}",
    )

    # STEP 1: Generate a security-focused description of the code
    messages_description = [
        {"role": "system", "content": system_prompt_description},
        {"role": "user", "content": user_prompt_description.format(code=code)},
    ]

    description = client.generate(messages_description)
    write_log(log_file, description, type="Description")

    # STEP 2: Retrieve similar vulnerable contracts from the vector store
    rag_contracts = search_similarity(description)

    # STEP 3: Build a prioritized checklist from retrieval results
    checklist = create_rag_checklist(rag_contracts, with_few_shot=True)
    write_log(log_file, checklist, type="Checklist")

    # STEP 4: Perform the audit with RAG context
    messages = [
        {"role": "system", "content": system_prompt_rag},
        {"role": "user", "content": user_prompt_rag.format(checklist=checklist, code=code)},
    ]

    MAX_RETRY = 3
    retry = 0
    vulnerabilities = []
    response = ""

    while retry < MAX_RETRY:
        print(f"[{file}] Attempt {retry + 1}/{MAX_RETRY}...")
        response = client.generate(messages)

        success, vulnerabilities = parse_response(response)

        if success:
            break
        else:
            retry += 1

    write_log(log_file, response, type="Audit")
    write_log_summary(
        test_logs_dir,
        f"Vulnerabilities found in the audit: {vulnerabilities}",
    )

print(f"\nBenchmark complete. Logs saved to {test_logs_dir}")

"""
Step 5c: SOTA generalist baseline benchmark WITHOUT RAG (Solana).
Mirrors the DeepSeek-V3.2 baseline used in the Algorand study (Tortora, 2025),
to enable a like-for-like SOTA-vs-specialist comparison across chains.

No fine-tuning is applied to DeepSeek: it is evaluated purely via
in-context learning (API), exactly as in the Algorand benchmark.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    export DEEPSEEK_MODEL="deepseek-chat"   # set to the EXACT model id/snapshot
                                             # used for the Algorand DeepSeek-V3.2
                                             # benchmark, for a fair comparison
                                             # (see Threats to Validity: API model
                                             # versions can change silently).
    python run_benchmark_deepseek_no_rag.py
"""

import json
import os

from utils.clients import get_client
from utils.test_logs import create_test_log_dir, create_log_file, write_log, write_log_summary
from utils.prompts import system_prompt_no_rag, user_prompt_no_rag
from utils.parser import parse_response

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
test_logs_basedir = "../../results/test_logs/no_rag"

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

    messages = [
        {"role": "system", "content": system_prompt_no_rag},
        {"role": "user", "content": user_prompt_no_rag.format(code=code)},
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

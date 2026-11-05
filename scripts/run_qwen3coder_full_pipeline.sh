#!/bin/bash
set -x
cd /home/biasi/Documents/RAG4SCVulnDetection
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

wait_for_gpu_mem() {
    local need_mib="$1"
    local waited=0
    local max_wait=43200
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        echo "GPU free: ${free_mib} MiB (need >= ${need_mib} MiB), waited ${waited}s"
        if [ "$free_mib" -ge "$need_mib" ]; then
            break
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "Timed out waiting for GPU memory."
            break
        fi
        sleep 60
        waited=$((waited + 60))
    done
}

ALGORAND_CKPT="models/Qwen3-Coder-30B-pyteal-desc"
SOLANA_CKPT="LLM-Contract-Analyzer/models/Qwen3-Coder-30B-Solana-Audit"

if [ -d "$ALGORAND_CKPT" ]; then
    echo "=== Qwen3-Coder-30B-pyteal-desc checkpoint already exists, skipping Algorand training ==="
else
    wait_for_gpu_mem 26000
    python scripts/train_qwen3coder_30b_algorand.py
    train_exit=$?
    echo "=== QWEN3CODER_ALGORAND_TRAINING_DONE (exit=$train_exit) ==="
    if [ "$train_exit" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: Algorand training failed ==="
        exit 1
    fi
fi

if [ -d "$SOLANA_CKPT" ]; then
    echo "=== Qwen3-Coder-30B-Solana-Audit checkpoint already exists, skipping Solana training ==="
else
    wait_for_gpu_mem 26000
    python scripts/train_qwen3coder_30b_solana.py
    train_exit=$?
    echo "=== QWEN3CODER_SOLANA_TRAINING_DONE (exit=$train_exit) ==="
    if [ "$train_exit" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: Solana training failed ==="
        exit 1
    fi
fi

CONFIGS="algorand_qwen3coder_norag algorand_qwen3coder_rag algorand_qwen3coder_ft_norag algorand_qwen3coder_ft_rag solana_qwen3coder_norag solana_qwen3coder_rag solana_qwen3coder_ft_norag solana_qwen3coder_ft_rag"
for c in $CONFIGS; do
    wait_for_gpu_mem 26000
    python scripts/eval_qwen3coder_baseline.py "$c"
    exit_code=$?
    echo "=== QWEN3CODER_CONFIG_DONE: $c (exit=$exit_code) ==="
    if [ "$exit_code" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: config $c failed ==="
        exit 1
    fi
done

echo "=== QWEN3CODER_PIPELINE_COMPLETE ==="

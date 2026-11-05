#!/bin/bash
set -x
cd /home/biasi/Documents/RAG4SCVulnDetection
source .venv-gemma4/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

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

ALGORAND_CKPT="models/Gemma-4-31B-pyteal-desc"
SOLANA_CKPT="LLM-Contract-Analyzer/models/Gemma-4-31B-Solana-Audit"

if [ -d "$ALGORAND_CKPT" ]; then
    echo "=== Gemma-4-31B-pyteal-desc checkpoint already exists, skipping Algorand training ==="
else
    wait_for_gpu_mem 26000
    python scripts/train_gemma4_31b_algorand.py
    train_exit=$?
    echo "=== GEMMA4_ALGORAND_TRAINING_DONE (exit=$train_exit) ==="
    if [ "$train_exit" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: Algorand training failed ==="
        exit 1
    fi
fi

if [ -d "$SOLANA_CKPT" ]; then
    echo "=== Gemma-4-31B-Solana-Audit checkpoint already exists, skipping Solana training ==="
else
    wait_for_gpu_mem 26000
    python scripts/train_gemma4_31b_solana.py
    train_exit=$?
    echo "=== GEMMA4_SOLANA_TRAINING_DONE (exit=$train_exit) ==="
    if [ "$train_exit" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: Solana training failed ==="
        exit 1
    fi
fi

CONFIGS="algorand_gemma4_31b_norag algorand_gemma4_31b_rag algorand_gemma4_31b_ft_norag algorand_gemma4_31b_ft_rag solana_gemma4_31b_norag solana_gemma4_31b_rag solana_gemma4_31b_ft_norag solana_gemma4_31b_ft_rag"
for c in $CONFIGS; do
    wait_for_gpu_mem 26000
    python scripts/eval_gemma4_baseline.py "$c"
    exit_code=$?
    echo "=== GEMMA4_CONFIG_DONE: $c (exit=$exit_code) ==="
    if [ "$exit_code" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: config $c failed ==="
        exit 1
    fi
done

echo "=== GEMMA4_PIPELINE_COMPLETE ==="

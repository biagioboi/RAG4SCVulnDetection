#!/bin/bash
set -x
cd /home/biasi/Documents/RAG4SCVulnDetection
source .venv/bin/activate

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

CKPT_DIR="LLM-Contract-Analyzer/models/QwQ-32B-Solana-Audit"

if [ -d "$CKPT_DIR" ]; then
    echo "=== QwQ-32B-Solana-Audit checkpoint already exists, skipping training ==="
else
    wait_for_gpu_mem 26000
    python scripts/train_qwq_32b_solana.py
    train_exit=$?
    echo "=== QWQ_SOLANA_TRAINING_DONE (exit=$train_exit) ==="
    if [ "$train_exit" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: training failed ==="
        exit 1
    fi
fi

SOLANA_QWQ_CONFIGS="solana_qwq_base_norag solana_qwq_base_rag solana_qwq_ft_norag solana_qwq_ft_rag"
for c in $SOLANA_QWQ_CONFIGS; do
    wait_for_gpu_mem 26000
    python scripts/eval_solana_all_multisample.py "$c"
    exit_code=$?
    echo "=== SOLANA_CONFIG_DONE: $c (exit=$exit_code) ==="
    if [ "$exit_code" -ne 0 ]; then
        echo "=== PIPELINE_ABORTED: Solana config $c failed ==="
        exit 1
    fi
done

echo "=== QWQ_SOLANA_PIPELINE_COMPLETE ==="

#!/bin/bash
set -x
cd /home/biasi/Documents/RAG4SCVulnDetection
source .venv/bin/activate

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_for_gpu_mem() {
    # Wait until at least $1 MiB free on GPU 0, checking every 60s, up to 6h.
    local need_mib="$1"
    local waited=0
    local max_wait=21600
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

# One fresh python process per model/config, so CUDA memory is fully
# released by process exit between them (a single long-lived process
# accumulated ~47GB across sequential model loads and OOM'd).

ALGORAND_MODELS="qwen3_base qwen3_ft qwq_base qwq_ft"
for m in $ALGORAND_MODELS; do
    wait_for_gpu_mem 26000
    python scripts/eval_algorand_all_multisample.py "$m"
    exit_code=$?
    echo "=== ALGORAND_MODEL_DONE: $m (exit=$exit_code) ==="
    if [ "$exit_code" -ne 0 ]; then
        echo "=== CHAIN_ABORTED: Algorand model $m failed ==="
        exit 1
    fi
done
echo "=== ALGORAND_ALL_MULTISAMPLE_DONE ==="

SOLANA_CONFIGS="solana_ft_norag solana_base_rag solana_base_norag"
for c in $SOLANA_CONFIGS; do
    wait_for_gpu_mem 26000
    python scripts/eval_solana_all_multisample.py "$c"
    exit_code=$?
    echo "=== SOLANA_CONFIG_DONE: $c (exit=$exit_code) ==="
    if [ "$exit_code" -ne 0 ]; then
        echo "=== CHAIN_ABORTED: Solana config $c failed ==="
        exit 1
    fi
done
echo "=== SOLANA_ALL_MULTISAMPLE_DONE ==="

echo "=== CHAIN_COMPLETE ==="

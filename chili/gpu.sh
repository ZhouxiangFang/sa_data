#!/usr/bin/env bash
# Run a command on one free GPU (CUDA_VISIBLE_DEVICES), holding its lock until it exits.
#   chili/gpu.sh python calculate_diversity.py --seeds 43
# Waits until a GPU in GPUS (default: every GPU) is free; see common.sh.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
(( $# )) || { echo "Usage: chili/gpu.sh COMMAND [ARGS...]" >&2; exit 2; }

IFS=$', \n' read -d '' -ra gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)}"
while true; do
    for gpu in "${gpus[@]}"; do
        gpus_free "$gpu" && lock_gpus "$gpu" || continue
        echo "Running on GPU $gpu: $*"
        CUDA_VISIBLE_DEVICES=$gpu "$@"
        exit
    done
    sleep "$POLL_SECONDS"
done

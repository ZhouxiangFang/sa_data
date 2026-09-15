#!/usr/bin/env bash
set -euo pipefail

# Full-parameter instruction tuning on four local GPUs by default.
#
# Usage:
#   ./git_training.sh [MODEL] [DATASET] [CHECKPOINT_ROOT] [extra Python args...]
#
# Example:
#   ./git_training.sh qwen3 ../data/git_data_10k.csv /home/$USER/models
#   GPU_GROUP=4,5,6,7 ./git_training.sh qwen3
#
# GPU_GROUP may be auto (default), 0,1,2,3, or 4,5,6,7. Auto only selects a
# group when every card in that group uses less than MAX_USED_MB (default 2 GB).
#
# Supported base-model shortcuts:
#   llama3 llama3.1 qwen2.5 qwen3 olmo2 olmo3

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Temporary workaround for this host while its loaded NVIDIA driver and user-space
# CUDA libraries have different patch versions. It is a no-op on other hosts.
driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib
if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    if [[ ! -e "$driver_lib/libcuda.so.1" ]]; then
        echo "Missing temporary NVIDIA 595.84 libraries: $driver_lib" >&2
        exit 1
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$driver_lib:"*) ;;
        *) export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

model="${1:-qwen3}"
dataset="${2:-$script_dir/../data/git_data_10k.csv}"
checkpoint_root="${3:-/home/${USER}/models}"
if (( $# > 3 )); then
    extra_args=("${@:4}")
else
    extra_args=()
fi

gpu_group="${GPU_GROUP:-auto}"
max_used_mb="${MAX_USED_MB:-2000}"
master_port="${MASTER_PORT:-29500}"
ds_config="${DS_CONFIG:-$script_dir/ds_config.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "$gpu_group" == auto ]]; then
    if ! gpu_memory=$(nvidia-smi \
        --query-gpu=index,memory.used \
        --format=csv,noheader,nounits); then
        echo "Cannot query GPUs; set GPU_GROUP=0,1,2,3 or GPU_GROUP=4,5,6,7 explicitly." >&2
        exit 1
    fi

    declare -A memory_used=()
    while IFS=',' read -r index used; do
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        memory_used[$index]="$used"
    done <<< "$gpu_memory"

    first_total=0
    second_total=0
    first_idle=1
    second_idle=1
    for index in 0 1 2 3; do
        [[ -n "${memory_used[$index]:-}" ]] || first_idle=0
        (( ${memory_used[$index]:-$max_used_mb} < max_used_mb )) || first_idle=0
        first_total=$((first_total + ${memory_used[$index]:-0}))
    done
    for index in 4 5 6 7; do
        [[ -n "${memory_used[$index]:-}" ]] || second_idle=0
        (( ${memory_used[$index]:-$max_used_mb} < max_used_mb )) || second_idle=0
        second_total=$((second_total + ${memory_used[$index]:-0}))
    done

    if (( first_idle && (! second_idle || first_total <= second_total) )); then
        gpu_group=0,1,2,3
    elif (( second_idle )); then
        gpu_group=4,5,6,7
    else
        echo "Neither GPU group is idle enough (limit: ${max_used_mb} MB per GPU)." >&2
        echo "  0,1,2,3 total used: ${first_total} MB" >&2
        echo "  4,5,6,7 total used: ${second_total} MB" >&2
        exit 1
    fi
elif [[ "$gpu_group" != 0,1,2,3 && "$gpu_group" != 4,5,6,7 ]]; then
    echo "GPU_GROUP must be auto, 0,1,2,3, or 4,5,6,7; got: $gpu_group" >&2
    exit 2
fi

echo "MODEL=$model"
echo "DATASET=$dataset"
echo "CHECKPOINT_ROOT=$checkpoint_root"
echo "GPU_GROUP=$gpu_group"
echo "DS_CONFIG=$ds_config"

exec deepspeed --include="localhost:$gpu_group" --master_port="$master_port" \
    "$script_dir/git_training.py" \
    --model "$model" \
    --dataset "$dataset" \
    --output_dir "$checkpoint_root" \
    --deepspeed "$ds_config" \
    "${extra_args[@]}"

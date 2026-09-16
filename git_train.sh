#!/usr/bin/env bash
set -euo pipefail

# Train two models at a time, one on each four-GPU group.
# Logs: ../log/git_train_MODEL_PID.log

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage:
  ./git_train.sh [MODEL] [DATASET] [CHECKPOINT_ROOT] [extra Python args...]
  ./git_train.sh --models qwen2.5,qwen3 [DATASET] [CHECKPOINT_ROOT] [extra Python args...]

Models: llama3, llama3.1, qwen2.5, qwen3 (default), olmo2, mistral
GPU_GROUP: auto (default), 0,1,2,3, or 4,5,6,7
MAX_USED_MB: maximum memory used per GPU (default 2000, exclusive)
GPU_POLL_SECONDS: time between GPU checks (default 30)
EOF
}

# Parse the model list, then the optional dataset, output directory, and arguments.
case "${1:-}" in
    --models|--model)
        if (( $# < 2 )); then
            echo "$1 requires a model name or comma-separated model list." >&2
            exit 2
        fi
        model_list="$2"
        shift 2
        ;;
    --models=*)
        model_list="${1#*=}"
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        model_list="${1:-qwen3}"
        if (( $# > 0 )); then shift; fi
        ;;
esac

if [[ -z "$model_list" || "$model_list" == ,* || "$model_list" == *, || "$model_list" == *,,* ]]; then
    echo "Model list must contain nonempty names separated by commas." >&2
    exit 2
fi
IFS=',' read -r -a models <<< "$model_list"
for model in "${models[@]}"; do
    case "$model" in
        llama3|llama3.1|qwen2.5|qwen3|olmo2|mistral) ;;
        *)
            echo "Unsupported model: $model" >&2
            exit 2
            ;;
    esac
done

dataset="${1:-$script_dir/../data/git_data_10k.csv}"
checkpoint_root="${2:-/home/${USER}/models}"
extra_args=("${@:3}")

log_dir="$script_dir/../log"
mkdir -p -- "$log_dir"

# Make pip-installed NVIDIA headers visible when DeepSpeed compiles CUDA ops.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    for include_dir in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/include; do
        if [[ -d "$include_dir" && ":${CPATH:-}:" != *":$include_dir:"* ]]; then
            export CPATH="$include_dir${CPATH:+:$CPATH}"
        fi
    done
fi

# Temporary workaround for this host while its loaded NVIDIA driver and user-space
# CUDA libraries have different patch versions. It is a no-op on other hosts.
driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib
if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    if [[ ! -e "$driver_lib/libcuda.so.1" ]]; then
        echo "Missing temporary NVIDIA 595.84 libraries: $driver_lib" >&2
        exit 1
    fi
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$driver_lib:"* ]]; then
        export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

requested_gpu_group="${GPU_GROUP:-auto}"
max_used_mb="${MAX_USED_MB:-2000}"
poll_seconds="${GPU_POLL_SECONDS:-30}"
master_port="${MASTER_PORT:-29500}"
ds_config="${DS_CONFIG:-$script_dir/ds_config.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! "$max_used_mb" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_USED_MB must be a positive integer; got: $max_used_mb" >&2
    exit 2
fi

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_POLL_SECONDS must be a positive integer; got: $poll_seconds" >&2
    exit 2
fi

case "$requested_gpu_group" in
    auto) candidate_groups=(0,1,2,3 4,5,6,7) ;;
    0,1,2,3|4,5,6,7) candidate_groups=("$requested_gpu_group") ;;
    *)
        echo "GPU_GROUP must be auto, 0,1,2,3, or 4,5,6,7; got: $requested_gpu_group" >&2
        exit 2
        ;;
esac

for program in nvidia-smi deepspeed; do
    if ! command -v "$program" >/dev/null; then
        echo "$program is required; activate your training environment." >&2
        exit 1
    fi
done

# Return success when every GPU in a group is below the memory limit.
group_is_idle() {
    local group="$1" gpu_memory index used
    local -a indices
    local -A memory_used=()

    gpu_memory=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits) || return 1
    while IFS=',' read -r index used; do
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        if [[ "$index" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]]; then
            memory_used[$index]=$((10#$used))
        fi
    done <<< "$gpu_memory"

    IFS=',' read -r -a indices <<< "$group"
    for index in "${indices[@]}"; do
        used="${memory_used[$index]:-$max_used_mb}"
        (( used < max_used_mb )) || return 1
    done
}

wait_for_group() {
    local group="$1"
    until group_is_idle "$group"; do
        echo "Waiting for GPUs $group below $max_used_mb MB each; retrying in ${poll_seconds}s."
        sleep "$poll_seconds"
    done
}

run_model() {
    local model="$1" gpu_group="$2" port="$3" log_file="$4"
    echo "LOG_FILE=$log_file"
    echo "MODEL=$model"
    echo "DATASET=$dataset"
    echo "CHECKPOINT_ROOT=$checkpoint_root"
    echo "DS_CONFIG=$ds_config"
    wait_for_group "$gpu_group"
    echo "GPU_GROUP=$gpu_group"

    deepspeed --include="localhost:$gpu_group" --master_port="$port" \
        "$script_dir/git_train.py" \
        --model "$model" \
        --dataset "$dataset" \
        --output_dir "$checkpoint_root" \
        --deepspeed "$ds_config" \
        "${extra_args[@]}"
}

run_queue() {
    local worker="$1" gpu_group="$2" index model log_file
    local port=$((master_port + worker))

    for ((index=worker; index<${#models[@]}; index+=${#candidate_groups[@]})); do
        model="${models[$index]}"
        log_file="$log_dir/git_train_${model}_$$.log"
        if ! run_model "$model" "$gpu_group" "$port" "$log_file" 2>&1 | tee -a "$log_file"; then
            return 1
        fi
    done
}

pids=()
for worker in "${!candidate_groups[@]}"; do
    run_queue "$worker" "${candidate_groups[$worker]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"

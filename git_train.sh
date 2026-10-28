#!/usr/bin/env bash
set -euo pipefail

# Train up to two models at a time. Models share one queue; whichever
# four-GPU group is idle takes the next model.
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
JOB_ATTEMPTS: runs per model before it counts as failed (default 2)
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

dataset="${1:-$script_dir/../data/git_data_20k.csv}"
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
max_attempts="${JOB_ATTEMPTS:-2}"
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

if [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOB_ATTEMPTS must be a positive integer; got: $max_attempts" >&2
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

# Per-GPU locks shared with eval_run_helpers.sh, so a training job and an
# evaluation never start on the same GPUs.
gpu_lock_file() {
    printf '/tmp/align_eval_gpu_%s.lock' "$1"
}

group_is_unlocked() {
    local gpu
    local -a indices
    IFS=',' read -r -a indices <<< "$1"
    for gpu in "${indices[@]}"; do
        flock -n "$(gpu_lock_file "$gpu")" true 2>/dev/null || return 1
    done
}

# Hold the group's locks for the rest of the calling (sub)shell; exit status 75
# means an evaluation took one first.
lock_group() {
    local gpu fd
    local -a indices
    IFS=',' read -r -a indices <<< "$1"
    for gpu in "${indices[@]}"; do
        exec {fd}>"$(gpu_lock_file "$gpu")"
        flock -n "$fd" || return 75
    done
}

run_model() {
    local model="$1" gpu_group="$2" port="$3" log_file="$4"
    echo "LOG_FILE=$log_file"
    echo "MODEL=$model"
    echo "DATASET=$dataset"
    echo "CHECKPOINT_ROOT=$checkpoint_root"
    echo "DS_CONFIG=$ds_config"
    echo "GPU_GROUP=$gpu_group"

    deepspeed --include="localhost:$gpu_group" --master_port="$port" \
        "$script_dir/git_train.py" \
        --model "$model" \
        --dataset "$dataset" \
        --output_dir "$checkpoint_root" \
        --deepspeed "$ds_config" \
        "${extra_args[@]}"
}

run_logged() {
    local model="$1" gpu_group="$2" port="$3"
    local log_file="$log_dir/git_train_${model}_$$.log"
    run_model "$model" "$gpu_group" "$port" "$log_file" 2>&1 | tee -a "$log_file"
}

# A single dispatcher hands the next pending model to whichever group is idle,
# so a group held by other users never blocks models another group could train.
# A failed model goes back to the end of the queue until it uses its attempts.
pending=("${!models[@]}")
declare -A group_pid=() group_job=()
attempts=()
failed=()
last_wait_message=""
while (( ${#pending[@]} > 0 || ${#group_pid[@]} > 0 )); do
    for group in "${!group_pid[@]}"; do
        pid="${group_pid[$group]}"
        ! kill -0 "$pid" 2>/dev/null || continue
        index="${group_job[$group]}"
        unset 'group_pid[$group]' 'group_job[$group]'
        job_status=0
        wait "$pid" || job_status=$?
        if (( job_status == 0 )); then
            echo "Finished ${models[$index]} on GPUs $group"
            continue
        fi
        if (( job_status == 75 )); then
            echo "GPUs $group were reserved by an evaluation before ${models[$index]} started; requeued."
            pending=("$index" "${pending[@]}")
            continue
        fi
        attempts[index]=$(( ${attempts[index]:-0} + 1 ))
        if (( attempts[index] < max_attempts )); then
            echo "Failed ${models[$index]} on GPUs $group; requeued (attempt ${attempts[index]}/$max_attempts)." >&2
            pending+=("$index")
        else
            echo "Failed ${models[$index]} on GPUs $group after $max_attempts attempt(s)." >&2
            failed+=("${models[$index]}")
        fi
    done

    busy_groups=()
    for worker in "${!candidate_groups[@]}"; do
        group="${candidate_groups[$worker]}"
        (( ${#pending[@]} > 0 )) || break
        [[ -z "${group_pid[$group]:-}" ]] || continue
        if ! group_is_idle "$group" || ! group_is_unlocked "$group"; then
            busy_groups+=("$group")
            continue
        fi
        index="${pending[0]}"
        pending=("${pending[@]:1}")
        (
            lock_group "$group" || exit $?
            run_logged "${models[$index]}" "$group" "$((master_port + worker))"
        ) &
        group_pid[$group]=$!
        group_job[$group]="$index"
    done

    wait_message=""
    if (( ${#pending[@]} > 0 && ${#busy_groups[@]} > 0 )); then
        wait_message="${#pending[@]} model(s) pending; GPUs ${busy_groups[*]} in use by other processes (limit $max_used_mb MB each)."
    fi
    if [[ -n "$wait_message" && "$wait_message" != "$last_wait_message" ]]; then
        echo "$(date '+%F %T') Waiting: $wait_message"
    fi
    last_wait_message="$wait_message"
    (( ${#pending[@]} == 0 && ${#group_pid[@]} == 0 )) || sleep "$poll_seconds"
done

if (( ${#failed[@]} > 0 )); then
    echo "${#failed[@]} model(s) failed: ${failed[*]}" >&2
    exit 1
fi

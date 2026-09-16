#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./eval_safe_run.sh --models MODEL,MODEL,... [eval_safe.py options]
  ./eval_safe_run.sh --folder DIR [eval_safe.py options]

Examples:
  ./eval_safe_run.sh --models qwen2.5-ins,llama3-ins
  ./eval_safe_run.sh --folder /models/checkpoints
  ./eval_safe_run.sh --models /models/ckpt-100,/models/ckpt-200 --permit
  ./eval_safe_run.sh --gpus 0,2 --models model-a,model-b,model-c

--folder scans one level down and includes directories containing config.json.
GPU selection, in order:
  --gpus, GPUS, CUDA_VISIBLE_DEVICES, or every GPU reported by nvidia-smi.
Extra models wait in a queue and use the next GPU that finishes.
MAX_USED_MB sets the availability threshold (default: 2000 MB).
GPU_POLL_SECONDS sets the retry interval (default: 10 seconds).
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin="${PYTHON_BIN:-python}"
log_dir="${LOG_DIR:-$script_dir/../log}"
run_timestamp=$(date +%Y%m%d_%H%M%S)

models=()
folders=()
gpus=()
eval_args=()

while (( $# > 0 )); do
    case "$1" in
        --models)
            [[ $# -ge 2 ]] || { echo "--models needs a value" >&2; exit 2; }
            IFS=',' read -r -a values <<< "$2"
            models+=("${values[@]}")
            shift 2
            ;;
        --models=*)
            IFS=',' read -r -a values <<< "${1#*=}"
            models+=("${values[@]}")
            shift
            ;;
        --folder)
            [[ $# -ge 2 ]] || { echo "--folder needs a value" >&2; exit 2; }
            folders+=("$2")
            shift 2
            ;;
        --folder=*)
            folders+=("${1#*=}")
            shift
            ;;
        --gpus)
            [[ $# -ge 2 ]] || { echo "--gpus needs a value" >&2; exit 2; }
            IFS=',' read -r -a gpus <<< "$2"
            shift 2
            ;;
        --gpus=*)
            IFS=',' read -r -a gpus <<< "${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            eval_args+=("$1")
            shift
            ;;
    esac
done

# Add model directories located directly inside each folder.
for folder in "${folders[@]}"; do
    if [[ ! -d "$folder" ]]; then
        echo "Folder not found: $folder" >&2
        exit 2
    fi

    found=0
    for model_dir in "$folder"/*; do
        if [[ -d "$model_dir" && -f "$model_dir/config.json" ]]; then
            models+=("$model_dir")
            ((found += 1))
        fi
    done
    echo "Found $found model(s) in $folder"
done

if (( ${#models[@]} == 0 )); then
    echo "No models found. Use --models or --folder." >&2
    exit 2
fi

# Avoid evaluating the same model twice when sources overlap.
unique_models=()
declare -A seen_models=()
for model in "${models[@]}"; do
    [[ -n "$model" ]] || { echo "Model names must not be empty." >&2; exit 2; }
    if [[ -z "${seen_models[$model]:-}" ]]; then
        unique_models+=("$model")
        seen_models[$model]=1
    fi
done
models=("${unique_models[@]}")

# If --gpus was omitted, use an environment setting or discover all GPUs.
if (( ${#gpus[@]} == 0 )); then
    gpu_list="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
    if [[ -n "$gpu_list" ]]; then
        IFS=',' read -r -a gpus <<< "$gpu_list"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t gpus < <(
            nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        )
    fi
fi

if (( ${#gpus[@]} == 0 )); then
    echo "No GPUs found. Set GPUS or use --gpus 0,1." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to check GPU availability." >&2
    exit 1
fi

# Temporary workaround for this machine's NVIDIA driver/library mismatch.
driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib
if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    [[ -e "$driver_lib/libcuda.so.1" ]] || {
        echo "Missing NVIDIA compatibility libraries: $driver_lib" >&2
        exit 1
    }
    export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

max_used_mb="${MAX_USED_MB:-2000}"
poll_seconds="${GPU_POLL_SECONDS:-10}"
if [[ ! "$max_used_mb" =~ ^[0-9]+$ || ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_USED_MB must be nonnegative and GPU_POLL_SECONDS must be positive." >&2
    exit 2
fi

gpu_memory_used() {
    nvidia-smi --id="$1" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

# Validate every requested GPU before entering the queue.
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
    if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
        echo "GPU $gpu was supplied more than once." >&2
        exit 2
    fi
    seen_gpus[$gpu]=1
    used=$(gpu_memory_used "$gpu") || used=""
    if [[ ! "$used" =~ ^[0-9]+$ ]]; then
        echo "Unable to query GPU $gpu with nvidia-smi." >&2
        exit 1
    fi
done

mkdir -p -- "$log_dir"

declare -A job_gpu=() job_model=() job_log=()
declare -A active_gpu=() reported_busy=()
failures=()
next_model=0

launch() {
    local gpu="$1" index="$2" model name log_file pid
    model="${models[$index]}"
    name="${model%/}"
    name="${name##*/}"
    name="${name//[^A-Za-z0-9._-]/_}"
    log_file="$log_dir/eval_safe_${name}_${run_timestamp}.log"

    echo "Starting on GPU $gpu: $model"
    echo "  log: $log_file"

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$python_bin" "$script_dir/eval_safe.py" \
        --model "$model" "${eval_args[@]}" >"$log_file" 2>&1 &

    pid=$!
    job_gpu[$pid]="$gpu"
    job_model[$pid]="$model"
    job_log[$pid]="$log_file"
    active_gpu[$gpu]=1
}

stop_jobs() {
    echo "Stopping evaluations..." >&2
    kill "${!job_gpu[@]}" 2>/dev/null || true
    exit 130
}
trap stop_jobs INT TERM

# Poll for completed jobs and idle GPUs until the queue is empty.
while (( next_model < ${#models[@]} || ${#job_gpu[@]} > 0 )); do
    made_progress=0

    for pid in "${!job_gpu[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu="${job_gpu[$pid]}"
            model="${job_model[$pid]}"
            if wait "$pid"; then
                echo "Finished on GPU $gpu: $model"
            else
                echo "Failed on GPU $gpu: $model" >&2
                echo "  log: ${job_log[$pid]}" >&2
                failures+=("$model")
            fi
            unset 'job_gpu[$pid]' 'job_model[$pid]' 'job_log[$pid]'
            unset 'active_gpu[$gpu]'
            made_progress=1
        fi
    done

    for gpu in "${gpus[@]}"; do
        (( next_model < ${#models[@]} )) || break
        [[ -z "${active_gpu[$gpu]:-}" ]] || continue

        used=$(gpu_memory_used "$gpu") || used=""
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used < max_used_mb )); then
            unset 'reported_busy[$gpu]'
            launch "$gpu" "$next_model"
            ((next_model += 1))
            made_progress=1
        elif [[ -z "${reported_busy[$gpu]:-}" ]]; then
            echo "GPU $gpu is busy (${used:-unknown} MB used); waiting."
            reported_busy[$gpu]=1
        fi
    done

    if (( next_model < ${#models[@]} || ${#job_gpu[@]} > 0 )) && \
       (( made_progress == 0 )); then
        sleep "$poll_seconds"
    fi
done

trap - INT TERM
if (( ${#failures[@]} > 0 )); then
    printf '%d evaluation(s) failed:\n' "${#failures[@]}" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "All ${#models[@]} evaluations completed."

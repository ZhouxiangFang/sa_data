#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_safe_eval.sh --models MODEL,MODEL,... [safe_eval.py options]
  ./run_safe_eval.sh --folder DIR [safe_eval.py options]

Examples:
  ./run_safe_eval.sh --models qwen2.5-ins,llama3-ins
  ./run_safe_eval.sh --folder /models/checkpoints
  ./run_safe_eval.sh --models /models/ckpt-100,/models/ckpt-200 --permit
  ./run_safe_eval.sh --gpus 0,2 --models model-a,model-b,model-c

--folder scans one level down and includes directories containing config.json.
GPU selection, in order:
  --gpus, GPUS, CUDA_VISIBLE_DEVICES, or every GPU reported by nvidia-smi.
Extra models wait in a queue and use the next GPU that finishes.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin="${PYTHON_BIN:-python}"
log_dir="${LOG_DIR:-$script_dir/../log}"

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

# Temporary workaround for this machine's NVIDIA driver/library mismatch.
driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib
if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    [[ -e "$driver_lib/libcuda.so.1" ]] || {
        echo "Missing NVIDIA compatibility libraries: $driver_lib" >&2
        exit 1
    }
    export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p -- "$log_dir"

declare -A job_gpu job_model job_log
failures=()
next_model=0

launch() {
    local gpu="$1" index="$2" model name log_file pid
    model="${models[$index]}"
    name="${model%/}"
    name="${name##*/}"
    name="${name//[^A-Za-z0-9._-]/_}"
    log_file="$log_dir/safe_eval_${index}_gpu${gpu}_${name}_$$.log"

    echo "Starting on GPU $gpu: $model"
    echo "  log: $log_file"

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$python_bin" "$script_dir/safe_eval.py" \
        --model "$model" "${eval_args[@]}" >"$log_file" 2>&1 &

    pid=$!
    job_gpu[$pid]="$gpu"
    job_model[$pid]="$model"
    job_log[$pid]="$log_file"
}

stop_jobs() {
    echo "Stopping evaluations..." >&2
    kill "${!job_gpu[@]}" 2>/dev/null || true
    exit 130
}
trap stop_jobs INT TERM

# Initially fill each GPU slot.
for gpu in "${gpus[@]}"; do
    (( next_model < ${#models[@]} )) || break
    launch "$gpu" "$next_model"
    ((next_model += 1))
done

# Whenever a job finishes, start the next queued model on that GPU.
while (( ${#job_gpu[@]} > 0 )); do
    finished_pid=""
    if wait -n -p finished_pid; then
        job_status=0
    else
        job_status=$?
    fi

    gpu="${job_gpu[$finished_pid]}"
    model="${job_model[$finished_pid]}"

    if (( job_status == 0 )); then
        echo "Finished on GPU $gpu: $model"
    else
        echo "Failed on GPU $gpu: $model" >&2
        echo "  log: ${job_log[$finished_pid]}" >&2
        failures+=("$model")
    fi

    unset 'job_gpu[$finished_pid]' 'job_model[$finished_pid]' 'job_log[$finished_pid]'

    if (( next_model < ${#models[@]} )); then
        launch "$gpu" "$next_model"
        ((next_model += 1))
    fi
done

trap - INT TERM
if (( ${#failures[@]} > 0 )); then
    printf '%d evaluation(s) failed:\n' "${#failures[@]}" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "All ${#models[@]} evaluations completed."

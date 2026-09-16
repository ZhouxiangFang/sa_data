#!/usr/bin/env bash
set -euo pipefail

# Temporary workaround: the running kernel still uses NVIDIA 595.84 while the
# system CUDA/NVML libraries have been upgraded to 595.91.07.
driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib

if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    if [[ ! -e "$driver_lib/libcuda.so.1" ]]; then
        echo "Missing temporary NVIDIA 595.84 libraries: $driver_lib" >&2
        echo "Ask to recreate the workaround, or reboot to load driver 595.91.07." >&2
        exit 1
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$driver_lib:"*) ;;
        *) export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

python_bin="${PYTHON_BIN:-python}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
log_dir="${LOG_DIR:-$script_dir/../log}"
run_timestamp=$(date +%Y%m%d_%H%M%S)

usage() {
    printf '%s\n' \
        "Usage:" \
        "  ./eval_if_run.sh --models MODEL [MODEL ...] [evaluation options]" \
        "  ./eval_if_run.sh --model MODEL --model MODEL [evaluation options]" \
        "  ./eval_if_run.sh --folder DIR [evaluation options]" \
        "" \
        "Examples:" \
        "  ./eval_if_run.sh --models qwen2.5-ins llama3-ins" \
        "  ./eval_if_run.sh --models /path/ckpt-100 /path/ckpt-200 --limit 10" \
        "  ./eval_if_run.sh --folder /path/to/models --limit 10" \
        "  ./eval_if_run.sh --gpus 0,2 --models qwen2.5-ins llama3-ins" \
        "" \
        "--folder scans one level down for directories containing config.json." \
        "GPU selection: --gpus, GPUS, CUDA_VISIBLE_DEVICES, then nvidia-smi." \
        "MAX_USED_MB defaults to 2000; GPU_POLL_SECONDS defaults to 10." \
        "" \
        "Shared evaluation options:" \
        "  --output_dir DIR" \
        "  --max_tokens N" \
        "  --tensor_parallel_size N" \
        "  --limit N"
}

models=()
folders=()
gpus=()
eval_args=()
tensor_parallel_size=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                models+=("$1")
                shift
            done
            ;;
        --model)
            if [[ $# -lt 2 ]]; then
                echo "--model requires a value" >&2
                exit 2
            fi
            models+=("$2")
            shift 2
            ;;
        --model=*)
            models+=("${1#--model=}")
            shift
            ;;
        --folder)
            if [[ $# -lt 2 ]]; then
                echo "--folder requires a value" >&2
                exit 2
            fi
            folders+=("$2")
            shift 2
            ;;
        --folder=*)
            folders+=("${1#--folder=}")
            shift
            ;;
        --gpus)
            if [[ $# -lt 2 ]]; then
                echo "--gpus requires a value" >&2
                exit 2
            fi
            IFS=',' read -r -a gpus <<< "$2"
            shift 2
            ;;
        --gpus=*)
            IFS=',' read -r -a gpus <<< "${1#--gpus=}"
            shift
            ;;
        --tensor_parallel_size)
            if [[ $# -lt 2 ]]; then
                echo "--tensor_parallel_size requires a value" >&2
                exit 2
            fi
            tensor_parallel_size="$2"
            eval_args+=("$1" "$2")
            shift 2
            ;;
        --tensor_parallel_size=*)
            tensor_parallel_size="${1#--tensor_parallel_size=}"
            eval_args+=("$1")
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

if [[ ${#models[@]} -eq 0 ]]; then
    echo "No models found. Use --models, --model, or --folder." >&2
    usage >&2
    exit 2
fi

# Avoid evaluating a model twice when inputs overlap.
unique_models=()
declare -A seen_models=()
for model in "${models[@]}"; do
    if [[ -z "$model" ]]; then
        echo "Model names must not be empty." >&2
        exit 2
    fi
    if [[ -z "${seen_models[$model]:-}" ]]; then
        unique_models+=("$model")
        seen_models[$model]=1
    fi
done
models=("${unique_models[@]}")

# Select the candidate GPU pool.
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
if [[ ! "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]] || \
   (( tensor_parallel_size > ${#gpus[@]} )); then
    echo "tensor_parallel_size must be between 1 and ${#gpus[@]}." >&2
    exit 2
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

wait_for_available_gpus() {
    local gpu used announced=0
    local -a selected

    while true; do
        selected=()
        for gpu in "${gpus[@]}"; do
            used=$(gpu_memory_used "$gpu") || used=""
            if [[ "$used" =~ ^[0-9]+$ ]] && (( used < max_used_mb )); then
                selected+=("$gpu")
                (( ${#selected[@]} == tensor_parallel_size )) && break
            fi
        done

        if (( ${#selected[@]} == tensor_parallel_size )); then
            available_gpu_group=$(IFS=,; echo "${selected[*]}")
            return
        fi

        if (( announced == 0 )); then
            echo "Waiting for $tensor_parallel_size GPU(s) below $max_used_mb MB used..."
            announced=1
        fi
        sleep "$poll_seconds"
    done
}

mkdir -p -- "$log_dir"
failed_models=()

for model in "${models[@]}"; do
    wait_for_available_gpus
    name="${model%/}"
    name="${name##*/}"
    name="${name//[^A-Za-z0-9._-]/_}"
    log_file="$log_dir/eval_if_${name}_${run_timestamp}.log"

    printf '\n%s\n' "============================================================"
    printf 'Evaluating model: %s\n' "$model"
    printf 'Using GPU(s): %s\n' "$available_gpu_group"
    printf 'Log: %s\n' "$log_file"
    printf '%s\n\n' "============================================================"

    if ! CUDA_VISIBLE_DEVICES="$available_gpu_group" "$python_bin" "$script_dir/eval_if.py" \
        --model "$model" "${eval_args[@]}" >"$log_file" 2>&1; then
        failed_models+=("$model")
        printf 'Evaluation failed: %s\n' "$model" >&2
        printf '  log: %s\n' "$log_file" >&2
    fi
done

if [[ ${#failed_models[@]} -gt 0 ]]; then
    printf '\nFailed models:\n' >&2
    printf '  %s\n' "${failed_models[@]}" >&2
    exit 1
fi

printf '\nAll %d model evaluation(s) completed successfully.\n' "${#models[@]}"

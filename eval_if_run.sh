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
        "" \
        "--folder scans one level down for directories containing config.json." \
        "" \
        "Shared evaluation options:" \
        "  --output_dir DIR" \
        "  --max_tokens N" \
        "  --tensor_parallel_size N" \
        "  --limit N"
}

models=()
folders=()
eval_args=()

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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
failed_models=()

for model in "${models[@]}"; do
    printf '\n%s\n' "============================================================"
    printf 'Evaluating model: %s\n' "$model"
    printf '%s\n\n' "============================================================"

    if ! python "$script_dir/eval_if.py" \
        --model "$model" "${eval_args[@]}"; then
        failed_models+=("$model")
        printf 'Evaluation failed: %s\n' "$model" >&2
    fi
done

if [[ ${#failed_models[@]} -gt 0 ]]; then
    printf '\nFailed models:\n' >&2
    printf '  %s\n' "${failed_models[@]}" >&2
    exit 1
fi

printf '\nAll %d model evaluation(s) completed successfully.\n' "${#models[@]}"

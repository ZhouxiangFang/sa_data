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
        "  ./run_if_eval.sh --models MODEL [MODEL ...] [evaluation options]" \
        "  ./run_if_eval.sh --model MODEL --model MODEL [evaluation options]" \
        "" \
        "Examples:" \
        "  ./run_if_eval.sh --models qwen2.5-ins llama3-ins" \
        "  ./run_if_eval.sh --models /path/ckpt-100 /path/ckpt-200 --limit 10" \
        "" \
        "Shared evaluation options:" \
        "  --output_dir DIR" \
        "  --max_tokens N" \
        "  --tensor_parallel_size N" \
        "  --limit N"
}

models=()
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

if [[ ${#models[@]} -eq 0 ]]; then
    echo "No models supplied." >&2
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
failed_models=()

for model in "${models[@]}"; do
    printf '\n%s\n' "============================================================"
    printf 'Evaluating model: %s\n' "$model"
    printf '%s\n\n' "============================================================"

    if ! python "$script_dir/if_eval.py" \
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

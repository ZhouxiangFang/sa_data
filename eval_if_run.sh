#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/eval_run_helpers.sh"

usage() {
    cat <<'EOF'
Usage:
  ./eval_if_run.sh --models MODEL [MODEL ...] [eval_if.py options]
  ./eval_if_run.sh --model MODEL --model MODEL [eval_if.py options]
  ./eval_if_run.sh --folder DIR [eval_if.py options]

Examples:
  ./eval_if_run.sh --models qwen2.5-ins llama3-ins
  ./eval_if_run.sh --folder /models/checkpoints --limit 10
  ./eval_if_run.sh --gpus 0,2 --models qwen2.5-ins llama3-ins

Results default to ../results/if_eval relative to this script. Override with --output_dir DIR.
Aligned checkpoints save to <model>_<num_train>_<dataset_name>/ with
<dataset>_<subcategory>_if_{summary,responses}.csv filenames. Baseline files
save directly in the result root as <model>_if_{summary,responses}.csv.

GPU priority: --gpus, GPUS, CUDA_VISIBLE_DEVICES, then nvidia-smi.
MAX_USED_MB defaults to 500; GPU_POLL_SECONDS defaults to 10.
EOF
}

models=()
folders=()
gpus=()
eval_args=(--output_dir "$script_dir/../results/if_eval")
tensor_parallel_size=1

while (( $# > 0 )); do
    case "$1" in
        --models)
            shift
            while (( $# > 0 )) && [[ "$1" != --* ]]; do models+=("$1"); shift; done
            ;;
        --model|--folder|--gpus|--tensor_parallel_size)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            case "$1" in
                --model) models+=("$2") ;;
                --folder) folders+=("$2") ;;
                --gpus) IFS=',' read -r -a gpus <<< "$2" ;;
                --tensor_parallel_size)
                    tensor_parallel_size="$2"
                    eval_args+=("$1" "$2")
                    ;;
            esac
            shift 2
            ;;
        --model=*) models+=("${1#*=}"); shift ;;
        --folder=*) folders+=("${1#*=}"); shift ;;
        --gpus=*) IFS=',' read -r -a gpus <<< "${1#*=}"; shift ;;
        --tensor_parallel_size=*)
            tensor_parallel_size="${1#*=}"
            eval_args+=("$1")
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) eval_args+=("$1"); shift ;;
    esac
done

scan_model_folders
dedupe_models
(( ${#models[@]} > 0 )) || { echo "No models found." >&2; exit 2; }
setup_gpu_pool

[[ "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]] && \
    (( tensor_parallel_size <= ${#gpus[@]} )) || {
    echo "tensor_parallel_size must be between 1 and ${#gpus[@]}." >&2
    exit 2
}

python_bin="${PYTHON_BIN:-python}"
log_dir="${LOG_DIR:-$script_dir/../log}"
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p -- "$log_dir"
failures=()

for model in "${models[@]}"; do
    reserve_gpu_group "$tensor_parallel_size"
    log_file="$log_dir/eval_if_$(model_tag "$model")_${timestamp}.log"

    echo "Evaluating on GPU(s) $available_gpu_group: $model"
    echo "  log: $log_file"

    if ! CUDA_VISIBLE_DEVICES="$available_gpu_group" PYTHONUNBUFFERED=1 \
        "$python_bin" "$script_dir/eval_if.py" \
        --model "$model" "${eval_args[@]}" >"$log_file" 2>&1; then
        echo "Evaluation failed: $model" >&2
        echo "  log: $log_file" >&2
        failures+=("$model")
    fi
    release_gpu_group
done

if (( ${#failures[@]} > 0 )); then
    printf '%d evaluation(s) failed:\n' "${#failures[@]}" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi
echo "All ${#models[@]} evaluations completed."

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
  ./eval_if_run.sh --gpus 0,1,2,3 --tensor_parallel_size 2 --models qwen2.5-ins llama3-ins

Results default to ../results/if_eval relative to this script. Override with --output_dir DIR.
Aligned checkpoints save to <model>_<num_train>_<dataset_name>/ with
<dataset>_<subcategory>_if_{summary,responses}.csv filenames. Baseline files
save directly in the result root as <model>_if_{summary,responses}.csv.

GPU priority: --gpus, GPUS, CUDA_VISIBLE_DEVICES, then nvidia-smi.
Models run concurrently, using one GPU each by default. --tensor_parallel_size N
reserves N GPUs per model; queued models start when enough GPUs become available.
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
declare -A job_gpus=() job_model=() job_log=() active_gpu=()
next_model=0

launch() {
    local model="${models[$1]}" log_file pid gpu
    log_file="$log_dir/eval_if_$(model_tag "$model")_${timestamp}.log"

    echo "Starting on GPU(s) $available_gpu_group: $model"
    echo "  log: $log_file"

    CUDA_VISIBLE_DEVICES="$available_gpu_group" PYTHONUNBUFFERED=1 \
        "$python_bin" "$script_dir/eval_if.py" \
        --model "$model" "${eval_args[@]}" >"$log_file" 2>&1 &
    pid=$!
    job_gpus[$pid]="$available_gpu_group"
    job_model[$pid]="$model"
    job_log[$pid]="$log_file"
    for gpu in "${selected[@]}"; do
        active_gpu[$gpu]=1
    done
    # The child inherits the reservation FDs and holds them until it exits.
    # Close the parent's copies before launching any other evaluation.
    release_gpu_group
}

stop_jobs() {
    trap '' INT TERM
    echo "Stopping evaluations..." >&2
    if (( ${#job_gpus[@]} > 0 )); then
        kill "${!job_gpus[@]}" 2>/dev/null || true
        wait "${!job_gpus[@]}" 2>/dev/null || true
    fi
    release_gpu_group
    exit 130
}
trap stop_jobs INT TERM

# Reap completed jobs, then fill every available group of GPUs.
while (( next_model < ${#models[@]} || ${#job_gpus[@]} > 0 )); do
    progressed=0
    for pid in "${!job_gpus[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        model="${job_model[$pid]}"
        if wait "$pid"; then
            echo "Finished on GPU(s) ${job_gpus[$pid]}: $model"
        else
            echo "Failed on GPU(s) ${job_gpus[$pid]}: $model" >&2
            echo "  log: ${job_log[$pid]}" >&2
            failures+=("$model")
        fi
        IFS=',' read -r -a finished_gpus <<< "${job_gpus[$pid]}"
        for gpu in "${finished_gpus[@]}"; do
            unset 'active_gpu[$gpu]'
        done
        unset 'job_gpus[$pid]' 'job_model[$pid]' 'job_log[$pid]'
        progressed=1
    done

    while (( next_model < ${#models[@]} )); do
        selected=()
        for gpu in "${gpus[@]}"; do
            [[ -z "${active_gpu[$gpu]:-}" ]] || continue
            if gpu_is_ready "$gpu"; then
                selected+=("$gpu")
                (( ${#selected[@]} == tensor_parallel_size )) && break
            fi
        done
        (( ${#selected[@]} == tensor_parallel_size )) || break
        try_reserve_gpu_group "${selected[@]}" || break
        launch "$next_model"
        ((next_model += 1))
        progressed=1
    done

    if (( next_model < ${#models[@]} || ${#job_gpus[@]} > 0 )) && (( progressed == 0 )); then
        if (( next_model < ${#models[@]} && ${#job_gpus[@]} == 0 )); then
            echo "Waiting for $tensor_parallel_size empty GPU(s)..."
        fi
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

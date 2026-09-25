#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/eval_run_helpers.sh"

usage() {
    cat <<'EOF'
Usage:
  ./eval_safe_run.sh --models MODEL,MODEL,... [eval_safe.py options]
  ./eval_safe_run.sh --folder DIR [eval_safe.py options]

Examples:
  ./eval_safe_run.sh --models qwen2.5-ins,llama3-ins
  ./eval_safe_run.sh --folder /models/checkpoints
  ./eval_safe_run.sh --gpus 0,2 --models model-a,model-b,model-c

Results default to ../results relative to this script. Override with --output_dir DIR.
Aligned checkpoints save to <model>_<num_train>_<dataset_name>/<dataset>_<subcategory>/.
Baseline models save to <model>/; filenames retain the model or dataset/subcategory prefix.

GPU priority: --gpus, GPUS, CUDA_VISIBLE_DEVICES, then nvidia-smi.
MAX_USED_MB defaults to 500; GPU_POLL_SECONDS defaults to 10.
EOF
}

models=()
folders=()
gpus=()
eval_args=(--output_dir "$script_dir/../results")

while (( $# > 0 )); do
    case "$1" in
        --models|--gpus)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            IFS=',' read -r -a values <<< "$2"
            if [[ "$1" == --models ]]; then models+=("${values[@]}"); else gpus=("${values[@]}"); fi
            shift 2
            ;;
        --models=*|--gpus=*)
            option="${1%%=*}"
            IFS=',' read -r -a values <<< "${1#*=}"
            if [[ "$option" == --models ]]; then models+=("${values[@]}"); else gpus=("${values[@]}"); fi
            shift
            ;;
        --folder)
            (( $# >= 2 )) || { echo "--folder needs a value" >&2; exit 2; }
            folders+=("$2")
            shift 2
            ;;
        --folder=*) folders+=("${1#*=}"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) eval_args+=("$1"); shift ;;
    esac
done

scan_model_folders || exit $?
dedupe_models || exit $?
(( ${#models[@]} > 0 )) || { echo "No models found." >&2; exit 2; }
setup_gpu_pool || exit $?

python_bin="${PYTHON_BIN:-python}"
log_dir="${LOG_DIR:-$script_dir/../log}"
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p -- "$log_dir"

declare -A job_gpu=() job_model=() job_log=() active_gpu=() reported_busy=()
failures=()
next_model=0

launch() {
    local gpu="$1" model="${models[$2]}" log_file pid
    log_file="$log_dir/eval_safe_$(model_tag "$model")_${timestamp}.log"

    echo "Starting on GPU $gpu: $model"
    echo "  log: $log_file"
    run_on_gpu "$gpu" "$python_bin" "$script_dir/eval_safe.py" \
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

# Reap completed jobs, then give queued models to every empty GPU.
while (( next_model < ${#models[@]} || ${#job_gpu[@]} > 0 )); do
    progressed=0

    for pid in "${!job_gpu[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        gpu="${job_gpu[$pid]}"
        model="${job_model[$pid]}"
        if wait "$pid"; then
            echo "Finished on GPU $gpu: $model"
        else
            echo "Failed on GPU $gpu: $model" >&2
            echo "  log: ${job_log[$pid]}" >&2
            failures+=("$model")
        fi
        unset 'job_gpu[$pid]' 'job_model[$pid]' 'job_log[$pid]' 'active_gpu[$gpu]'
        progressed=1
    done

    for gpu in "${gpus[@]}"; do
        (( next_model < ${#models[@]} )) || break
        [[ -z "${active_gpu[$gpu]:-}" ]] || continue

        if gpu_is_ready "$gpu"; then
            unset 'reported_busy[$gpu]'
            launch "$gpu" "$next_model"
            ((next_model += 1))
            progressed=1
        elif [[ -z "${reported_busy[$gpu]:-}" ]]; then
            echo "GPU $gpu unavailable ($(gpu_status "$gpu")); waiting."
            reported_busy[$gpu]=1
        fi
    done

    if (( next_model < ${#models[@]} || ${#job_gpu[@]} > 0 )) && (( progressed == 0 )); then
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

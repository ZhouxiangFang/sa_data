#!/usr/bin/env bash
set -uo pipefail

# Concrete eight-A6000 launcher. Each worker owns one four-GPU group and runs
# its assigned model/dataset/subcategory jobs sequentially.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage:
  ./align_run.sh [MODELS] [DATASETS] [NUM_TRAIN] [HARMFUL_RATE] [extra align.py args...]

Arguments:
  MODELS        Comma-separated models (default: qwen2.5-ins)
  DATASETS      Comma-separated datasets (default: wildguardmix,aegis)
  NUM_TRAIN     Total examples per job (default: 800)
  HARMFUL_RATE  Dataset-subcategory fraction (default: 0.5)

Examples:
  ./align_run.sh
  ./align_run.sh qwen2.5-ins,llama3.1-ins wildguardmix,aegis 800 0.5
  ./align_run.sh qwen2.5-ins wildguardmix 400 0.5 --epochs 2 --lr 1e-5

GPU_GROUP=auto uses both 0,1,2,3 and 4,5,6,7. It can instead be set to one
of those groups. MAX_USED_MB defaults to 2000 and GPU_POLL_SECONDS to 30.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi

# ---- concrete experiment configuration ----
model_list="${1:-qwen2.5-ins}"
dataset_list="${2:-wildguardmix,aegis}"
num_train="${3:-800}"
harmful_rate="${4:-0.5}"
extra_args=("${@:5}")

IFS=',' read -r -a models <<< "$model_list"
IFS=',' read -r -a datasets <<< "$dataset_list"

for model in "${models[@]}"; do
    [[ -n "$model" ]] || { echo "Model names must not be empty." >&2; exit 2; }
done
for dataset in "${datasets[@]}"; do
    case "$dataset" in
        wildguardmix|aegis) ;;
        *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
    esac
done

# Match align.py's validation and rounding for the number of examples drawn
# from the selected dataset subcategory.
count_values=$(python3 -c '
import math
import sys

try:
    total = int(sys.argv[1])
    rate = float(sys.argv[2])
except ValueError as error:
    raise SystemExit(f"Invalid NUM_TRAIN/HARMFUL_RATE: {error}")
if total <= 0:
    raise SystemExit("NUM_TRAIN must be positive")
if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
    raise SystemExit("HARMFUL_RATE must be between 0 and 1")
print(f"{total}\t{rate:g}\t{int(total * rate + 0.5)}")
' "$num_train" "$harmful_rate") || exit $?
IFS=$'\t' read -r num_train harmful_rate category_required <<< "$count_values"

# ---- conda environment ----
if [[ "${CONDA_DEFAULT_ENV:-}" != nlp ]]; then
    conda_script=/home/zf28/miniconda3/etc/profile.d/conda.sh
    [[ -f "$conda_script" ]] || { echo "Missing $conda_script" >&2; exit 1; }
    set +u
    # shellcheck source=/dev/null
    source "$conda_script"
    conda activate nlp
    conda_status=$?
    set -u
    (( conda_status == 0 )) || exit "$conda_status"
fi

for program in python nvidia-smi deepspeed; do
    command -v "$program" >/dev/null 2>&1 || {
        echo "$program is required in the nlp environment." >&2
        exit 1
    }
done

# Same CUDA setup used by git_train.sh.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    for include_dir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/include; do
        if [[ -d "$include_dir" && ":${CPATH:-}:" != *":$include_dir:"* ]]; then
            export CPATH="$include_dir${CPATH:+:$CPATH}"
        fi
    done
fi

driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib
if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    [[ -e "$driver_lib/libcuda.so.1" ]] || {
        echo "Missing temporary NVIDIA 595.84 libraries: $driver_lib" >&2
        exit 1
    }
    export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---- select subcategories with enough training examples ----
stats_file="$script_dir/subcategory_stats.json"
job_models=()
job_datasets=()
job_abbrs=()
skipped=()

for dataset in "${datasets[@]}"; do
    category_lines=$(python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    categories = json.load(handle)[sys.argv[2]]
for category in categories.values():
    print(category["abbr"], category["train"]["safe_response"], sep="\t")
' "$stats_file" "$dataset") || exit $?

    eligible=()
    while IFS=$'\t' read -r abbr available; do
        [[ -n "$abbr" ]] || continue
        if (( available >= category_required )); then
            eligible+=("$abbr")
        else
            skipped+=("$dataset/$abbr: $available available, $category_required required")
        fi
    done <<< "$category_lines"

    echo "$dataset: selected ${#eligible[@]} subcategory(s)."
    for model in "${models[@]}"; do
        for abbr in "${eligible[@]}"; do
            job_models+=("$model")
            job_datasets+=("$dataset")
            job_abbrs+=("$abbr")
        done
    done
done

if (( ${#skipped[@]} > 0 )); then
    echo "Skipping ${#skipped[@]} subcategory(s) without enough examples:"
    printf '  %s\n' "${skipped[@]}"
fi
(( ${#job_models[@]} > 0 )) || { echo "No eligible jobs remain." >&2; exit 2; }

# ---- two fixed four-GPU groups, matching git_train.sh ----
case "${GPU_GROUP:-auto}" in
    auto) gpu_groups=(0,1,2,3 4,5,6,7) ;;
    0,1,2,3|4,5,6,7) gpu_groups=("$GPU_GROUP") ;;
    *)
        echo "GPU_GROUP must be auto, 0,1,2,3, or 4,5,6,7." >&2
        exit 2
        ;;
esac

max_used_mb="${MAX_USED_MB:-2000}"
poll_seconds="${GPU_POLL_SECONDS:-30}"
master_port="${MASTER_PORT:-29500}"
[[ "$max_used_mb" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_USED_MB must be a positive integer." >&2
    exit 2
}
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "GPU_POLL_SECONDS must be a positive integer." >&2
    exit 2
}
[[ "$master_port" =~ ^[1-9][0-9]*$ ]] || {
    echo "MASTER_PORT must be a positive integer." >&2
    exit 2
}

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

model_tag() {
    local name="${1%/}"
    name="${name##*/}"
    printf '%s' "${name//[^A-Za-z0-9._-]/_}"
}

log_dir="${LOG_DIR:-$script_dir/../log}"
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p -- "$log_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_job() {
    local index="$1" gpu_group="$2" port="$3"
    local model="${job_models[$index]}"
    local dataset="${job_datasets[$index]}"
    local abbr="${job_abbrs[$index]}"
    local log_file="$log_dir/align_$(model_tag "$model")_${dataset}_${abbr}_${timestamp}_$((index + 1)).log"

    echo "Starting [$((index + 1))/${#job_models[@]}] on GPUs $gpu_group: $model | $dataset | $abbr"
    echo "LOG_FILE=$log_file"
    wait_for_group "$gpu_group"

    env -u CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1 \
        deepspeed --include="localhost:$gpu_group" --master_port="$port" \
        "$script_dir/align.py" \
        "${extra_args[@]}" \
        --model "$model" \
        --alignment_dataset "$dataset" \
        --abbr "$abbr" \
        --num_train "$num_train" \
        --harmful_rate "$harmful_rate" 2>&1 | tee -a "$log_file"
}

run_queue() {
    local worker="$1" gpu_group="$2" index
    local port=$((master_port + worker))
    local queue_status=0

    for ((index=worker; index<${#job_models[@]}; index+=${#gpu_groups[@]})); do
        run_job "$index" "$gpu_group" "$port" || queue_status=1
    done
    return "$queue_status"
}

echo "Configuration: models=$model_list datasets=$dataset_list num_train=$num_train harmful_rate=$harmful_rate"
echo "Required subcategory examples: $category_required"
echo "Queued ${#job_models[@]} job(s) on GPU groups: ${gpu_groups[*]}"

pids=()
for worker in "${!gpu_groups[@]}"; do
    run_queue "$worker" "${gpu_groups[$worker]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if (( status == 0 )); then
    echo "All ${#job_models[@]} alignment jobs completed."
else
    echo "One or more alignment jobs failed; check the log files." >&2
fi
exit "$status"

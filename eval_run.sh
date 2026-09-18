#!/usr/bin/env bash
set -euo pipefail

# Evaluate completed checkpoints with both evaluation launchers. Checkpoints are
# deleted only after both full evaluation passes succeed.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage:
  ./eval_run.sh --folder CHECKPOINT_DIR [options]

Options:
  --folder DIR                 Scan immediate children containing config.json;
                               may be repeated
  --gpus IDS                   Comma-separated GPU IDs passed to both launchers
  --limit N                    Limit IFBench and IFEval to N examples
  --max_tokens N               Maximum IF evaluation response tokens
  --tensor_parallel_size N     GPUs used by one IF evaluation (default: 1)
  --guard MODEL                Guard used by the safety evaluation
  --permit                     Enable the safety evaluation permit prompt
  --if-output-dir DIR          IF evaluation result directory
  --safe-output-dir DIR        Safety evaluation result directory
  --keep                       Keep checkpoints after successful evaluation
  -h, --help                   Show this help

Examples:
  ./align_run.sh --models qwen2.5-ins --datasets wildguardmix && \
      ./eval_run.sh --folder /scratch/zf28/ckpts

  ./eval_run.sh --folder /scratch/zf28/ckpts --gpus 0,1,2,3,4,5,6,7

By default, every discovered checkpoint is permanently deleted only when both
eval_if_run.sh and eval_safe_run.sh finish successfully. Results and logs are
stored outside the checkpoint directories by the underlying launchers.
EOF
}

folders=()
gpu_list=""
limit=""
max_tokens=""
tensor_parallel_size=1
guard=""
permit=0
if_output_dir=""
safe_output_dir=""
keep=0

while (( $# > 0 )); do
    case "$1" in
        --folder|--gpus|--limit|--max_tokens|--tensor_parallel_size|--guard|--if-output-dir|--safe-output-dir)
            (( $# >= 2 )) || { echo "$1 needs a value." >&2; exit 2; }
            case "$1" in
                --folder) folders+=("$2") ;;
                --gpus) gpu_list="$2" ;;
                --limit) limit="$2" ;;
                --max_tokens) max_tokens="$2" ;;
                --tensor_parallel_size) tensor_parallel_size="$2" ;;
                --guard) guard="$2" ;;
                --if-output-dir) if_output_dir="$2" ;;
                --safe-output-dir) safe_output_dir="$2" ;;
            esac
            shift 2
            ;;
        --folder=*) folders+=("${1#*=}"); shift ;;
        --gpus=*) gpu_list="${1#*=}"; shift ;;
        --limit=*) limit="${1#*=}"; shift ;;
        --max_tokens=*) max_tokens="${1#*=}"; shift ;;
        --tensor_parallel_size=*) tensor_parallel_size="${1#*=}"; shift ;;
        --guard=*) guard="${1#*=}"; shift ;;
        --if-output-dir=*) if_output_dir="${1#*=}"; shift ;;
        --safe-output-dir=*) safe_output_dir="${1#*=}"; shift ;;
        --permit) permit=1; shift ;;
        --keep) keep=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

(( ${#folders[@]} > 0 )) || {
    echo "At least one --folder is required." >&2
    exit 2
}
[[ "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]] || {
    echo "--tensor_parallel_size must be a positive integer." >&2
    exit 2
}
if [[ -n "$limit" && ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer." >&2
    exit 2
fi
if [[ -n "$max_tokens" && ! "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max_tokens must be a positive integer." >&2
    exit 2
fi

# eval_if_run.sh and eval_safe_run.sh expect the nlp environment to be active.
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

models=()
roots=()
for raw_folder in "${folders[@]}"; do
    folder="${raw_folder%/}"
    [[ -n "$folder" ]] || folder=/
    [[ -d "$folder" ]] || { echo "Folder not found: $folder" >&2; exit 2; }
    folder=$(realpath -e -- "$folder")
    roots+=("$folder")

    found=0
    for model_dir in "$folder"/*; do
        # Ignore symlinks so deletion can never follow a checkpoint outside the
        # explicitly supplied root.
        if [[ -d "$model_dir" && ! -L "$model_dir" && -f "$model_dir/config.json" ]]; then
            models+=("$(realpath -e -- "$model_dir")")
            ((found += 1))
        fi
    done
    echo "Found $found completed checkpoint(s) in $folder"
done

# Preserve the first occurrence if supplied folders overlap.
unique_models=()
declare -A seen_models=()
for model in "${models[@]}"; do
    if [[ -z "${seen_models[$model]:-}" ]]; then
        unique_models+=("$model")
        seen_models[$model]=1
    fi
done
models=("${unique_models[@]}")
(( ${#models[@]} > 0 )) || { echo "No completed checkpoints found." >&2; exit 2; }

if_args=(--models "${models[@]}" --tensor_parallel_size "$tensor_parallel_size")
safe_model_list=$(IFS=,; echo "${models[*]}")
safe_args=(--models "$safe_model_list")

if [[ -n "$gpu_list" ]]; then
    if_args+=(--gpus "$gpu_list")
    safe_args+=(--gpus "$gpu_list")
fi
[[ -n "$limit" ]] && if_args+=(--limit "$limit")
[[ -n "$max_tokens" ]] && if_args+=(--max_tokens "$max_tokens")
[[ -n "$if_output_dir" ]] && if_args+=(--output_dir "$if_output_dir")
[[ -n "$guard" ]] && safe_args+=(--guard "$guard")
(( permit == 0 )) || safe_args+=(--permit)
[[ -n "$safe_output_dir" ]] && safe_args+=(--output_dir "$safe_output_dir")

echo "Evaluating ${#models[@]} checkpoint(s) with instruction-following benchmarks."
if ! "$script_dir/eval_if_run.sh" "${if_args[@]}"; then
    echo "Instruction-following evaluation failed; checkpoints were kept." >&2
    exit 1
fi

echo "Evaluating ${#models[@]} checkpoint(s) with safety benchmarks."
if ! "$script_dir/eval_safe_run.sh" "${safe_args[@]}"; then
    echo "Safety evaluation failed; checkpoints were kept." >&2
    exit 1
fi

if (( keep == 1 )); then
    echo "Both evaluations succeeded; --keep was set, so checkpoints were retained."
    exit 0
fi

# Revalidate every destructive target immediately before deletion.
for model in "${models[@]}"; do
    valid_target=0
    for root in "${roots[@]}"; do
        if [[ "$(dirname -- "$model")" == "$root" ]]; then
            valid_target=1
            break
        fi
    done
    (( valid_target == 1 )) || {
        echo "Refusing to delete checkpoint outside the supplied folders: $model" >&2
        exit 1
    }
    [[ -d "$model" && ! -L "$model" && -f "$model/config.json" ]] || {
        echo "Refusing to delete an invalid checkpoint target: $model" >&2
        exit 1
    }
done

for model in "${models[@]}"; do
    rm -rf -- "$model"
    echo "Deleted checkpoint: $model"
done
echo "Both evaluations succeeded and ${#models[@]} checkpoint(s) were permanently deleted."

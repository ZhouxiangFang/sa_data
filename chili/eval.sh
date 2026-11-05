#!/usr/bin/env bash
# Evaluate models with eval.py (IF + safety, one model load each), one free GPU per model.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/eval.sh MODEL... [--delete] [-- eval.py options]

  MODEL...   Checkpoint paths, model shortcuts (qwen2.5-ins, llama3.1_git20k) or HF repo IDs
  --delete   Delete each trained checkpoint (a local folder named ..._seed<seed>) once
             its evaluation succeeds

eval.py options, e.g. --output_dir ../ablations/<setting> (default: results/), --limit 5.
Each model's log is log/eval_<name>_<time>.log. GPUS (default: every GPU) lists the
GPUs to use; a GPU takes the next model when it is free.
EOF
}

models=() eval_args=() delete=0
while (( $# )); do
    case $1 in
        --delete) delete=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; eval_args=("$@"); break ;;
        --*) echo "Unknown option: $1 (eval.py options go after --)" >&2; exit 2 ;;
        *) models+=("$1"); shift ;;
    esac
done
(( ${#models[@]} )) || { usage >&2; exit 2; }

job_label() { printf '%s' "$(basename "${models[$1]}")"; }

run_job() {
    local model=${models[$1]} log="$LOG_DIR/eval_$(basename "${models[$1]}")_$STAMP.log"
    echo "  log: $log"
    CUDA_VISIBLE_DEVICES=$2 python "$SA_DIR/eval.py" --model "$model" "${eval_args[@]}" \
        >> "$log" 2>&1 || return
    if (( delete )) && [[ -d $model && ! -L $model && -f $model/config.json
                          && $(basename "$model") =~ _seed[0-9]+$ ]]; then
        rm -rf -- "$model" && echo "Deleted $model"
    fi
}

IFS=$', \n' read -d '' -ra gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)}"
run_queue "${#models[@]}" "${gpus[@]}"

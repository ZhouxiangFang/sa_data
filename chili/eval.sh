#!/usr/bin/env bash
# Evaluate models with eval.py (IF + safety, one model load each) on chili, one free GPU per model.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/eval.sh MODEL... [--delete] [-- eval.py options]

  MODEL...   Checkpoint paths, model shortcuts (qwen2.5-ins, llama3.1_git20k) or HF repo IDs
  --delete   Delete each trained checkpoint (a local folder named ..._seed<seed>) once
             its evaluation succeeds

eval.py options, e.g. --output_dir ../results/correlations/<setting> (default:
results/baselines/, for models that are not trained checkpoints), --limit 5. Responses go to the
same tree under RESPONSES_ROOT (default /scratch/zf28/results; eval.py --responses_root, which
an option after -- overrides). Logs: log/chili/eval_<model>_<time>.log. GPUS (default: every
GPU, e.g. GPUS=0,1,2,3) lists the GPUs to use; a GPU takes the next model when it is free
(chili/common.sh).
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

job_label() { basename "${models[$1]}"; }

run_job() {
    local model=${models[$1]} log status
    log="$LOG_DIR/eval_$(basename "$model")_$STAMP.log"
    echo "  log: $log"
    {
        echo "$(date '+%F %T') eval $model on $(hostname), GPU $2"
        versions vllm torch transformers
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$2"
    } >> "$log"
    CUDA_VISIBLE_DEVICES=$2 python "$SA_DIR/eval.py" --model "$model" \
        --responses_root "$RESPONSES_ROOT" "${eval_args[@]}" >> "$log" 2>&1
    status=$?
    echo "$(date '+%F %T') exit $status" >> "$log"
    (( status == 0 )) || return "$status"
    if (( delete )) && [[ -d $model && ! -L $model && -f $model/config.json
                          && $(basename "$model") =~ _seed[0-9]+$ ]]; then
        rm -rf -- "$model" && echo "Deleted $model"
    fi
}

mapfile -t gpus < <(all_gpus)
echo "$(date '+%F %T') Evaluating ${#models[@]} model(s) on GPU(s) ${gpus[*]}" \
     "(responses: $RESPONSES_ROOT)"
run_queue "${#models[@]}" "${gpus[@]}"

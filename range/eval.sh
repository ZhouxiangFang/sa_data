#!/usr/bin/env bash
# Submit eval.py (IF + safety, one model load each) for some models as a Slurm array on range,
# one GPU (EVAL_GPU, default hopper: an H200 or H100) per model.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: range/eval.sh MODEL... [--delete] [-- eval.py options]

  MODEL...   Checkpoint paths, model shortcuts (qwen2.5-ins, llama3.1_git20k) or HF repo IDs
  --delete   Delete each trained checkpoint (a local folder named ..._seed<seed>) once
             its evaluation succeeds

eval.py options, e.g. --output_dir ../ablations/<setting> (default: results/), --limit 5.
Logs: log/slurm/eval_<job>_<task>.log; the model list is log/slurm/models_<time>.txt.
EVAL_GPU, EVAL_TIME, MAX_EVAL etc.: see range/common.sh. DEPENDENCY (e.g. afterok:<job>)
holds the array until another job has finished.
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

models_file=$LOG_DIR/models_$STAMP.txt
printf '%s\n' "${models[@]}" > "$models_file"
n=${#models[@]}
read -ra gpu <<< "$(gpu_opts 1)"
id=$(submit --job-name=eval --array="0-$((n - 1))%$MAX_EVAL" "${gpu[@]}" \
    --cpus-per-task="$EVAL_CPUS" --mem="$EVAL_MEM" --time="$EVAL_TIME" \
    --output="$LOG_DIR/eval_%A_%a.log" ${DEPENDENCY:+--dependency="$DEPENDENCY" --kill-on-invalid-dep=yes} \
    "$RANGE_DIR/eval.sbatch" "$models_file" "$delete" "${eval_args[@]}") || exit
echo "Evaluation: array $id, $n model(s) on 1 x $EVAL_GPU (list: $models_file)"

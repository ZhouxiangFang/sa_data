#!/usr/bin/env bash
#SBATCH --job-name=align
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One align.py job of a training array (range/train.sh submits it):
#   train.sbatch JOBS_TSV OUTPUT_DIR [align.py options]
# Task i trains line i+1 of JOBS_TSV (train_jobs.py output: model, dataset, group,
# checkpoint, status) with DeepSpeed on every GPU of the allocation.
set -uo pipefail
source "${RANGE_DIR:?submit with range/train.sh}/common.sh"
# TRAIN_VENV: train in a venv layered on the nlp env with another DeepSpeed (e.g.
# /scratch/zf28/envs/ds0196). Its deepspeed launcher, and so its python, comes first on PATH,
# and its ops are JIT-built into a folder of their own.
if [[ -n ${TRAIN_VENV:-} ]]; then
    source "$TRAIN_VENV/bin/activate" || exit 1
    export TORCH_EXTENSIONS_DIR=$TRAIN_VENV/torch_extensions
fi
python -c 'import deepspeed; print("DeepSpeed", deepspeed.__version__, "from", deepspeed.__file__)' 2>&1 | grep '^DeepSpeed'

jobs_tsv=$1 output_dir=$2
shift 2
IFS=$'\t' read -r model dataset abbr checkpoint _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$jobs_tsv")
[[ -n ${checkpoint:-} ]] || { echo "No line $((SLURM_ARRAY_TASK_ID + 1)) in $jobs_tsv" >&2; exit 2; }
echo "$(date '+%F %T') $(basename "$checkpoint") on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# With no --include/--num_gpus, the launcher uses exactly the GPUs Slurm assigned
# (CUDA_VISIBLE_DEVICES). The port depends on the job, as jobs share nodes, and stays below
# the ephemeral range (32768+): 29500 + job % 10000 reached it, and a client socket already
# on the port failed task 112 of array 383741 (EADDRINUSE, 2026-11-06). Ports in use are skipped.
# A failed attempt may have left part of the checkpoint; the planner said it is incomplete.
master_port=$((29500 + SLURM_JOB_ID % 2000))
while ss -Htan "sport = :$master_port" | grep -q .; do
    master_port=$((29500 + (master_port - 29500 + 1) % 3000))
done
mem_log=$(mktemp)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 15 \
    > "$mem_log" 2>/dev/null &
sampler=$!
deepspeed --master_port="$master_port" "$SA_DIR/align.py" \
    --model "$model" --alignment_dataset "$dataset" --abbr "$abbr" \
    --output_dir "$output_dir" --overwrite_output_dir "$@"
status=$?
kill "$sampler" 2>/dev/null
awk -F', ' '$2 > peak[$1] { peak[$1] = $2; total[$1] = $3 }
            END { for (g in peak) printf "Peak memory on GPU %s: %d of %d MiB\n", g, peak[g], total[g] }' "$mem_log"
rm -f -- "$mem_log"
# With train.sh --eval, eval task i waits on this task (aftercorr), but on 2026-11-06 evals 19+
# of array 383824 kept aftercorr:383741_*(unfulfilled) hours after their training succeeded
# (Slurm had probably purged the finished tasks' records; MinJobAge is 300 s). So the training
# task clears its eval's dependency itself, or cancels the eval if training failed.
if [[ -s $jobs_tsv.eval_id ]]; then
    eval_task=$(<"$jobs_tsv.eval_id")_$SLURM_ARRAY_TASK_ID
    if (( status == 0 )); then
        scontrol update JobId="$eval_task" Dependency= && echo "Released eval $eval_task"
    else
        scancel "$eval_task" && echo "Cancelled eval $eval_task"
    fi
fi
echo "$(date '+%F %T') exit $status"
exit "$status"

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

jobs_tsv=$1 output_dir=$2
shift 2
IFS=$'\t' read -r model dataset abbr checkpoint _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$jobs_tsv")
[[ -n ${checkpoint:-} ]] || { echo "No line $((SLURM_ARRAY_TASK_ID + 1)) in $jobs_tsv" >&2; exit 2; }
echo "$(date '+%F %T') $(basename "$checkpoint") on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# With no --include/--num_gpus, the launcher uses exactly the GPUs Slurm assigned
# (CUDA_VISIBLE_DEVICES). The port depends on the job, as jobs share nodes.
# A failed attempt may have left part of the checkpoint; the planner said it is incomplete.
deepspeed --master_port=$((29500 + SLURM_JOB_ID % 10000)) "$SA_DIR/align.py" \
    --model "$model" --alignment_dataset "$dataset" --abbr "$abbr" \
    --output_dir "$output_dir" --overwrite_output_dir "$@"
status=$?
echo "$(date '+%F %T') exit $status"
exit "$status"

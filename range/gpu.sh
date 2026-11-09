#!/usr/bin/env bash
# Submit one command as a one-GPU job (EVAL_GPU, default hopper: an H200 or H100), run from sa_data/:
#   range/gpu.sh python calculate_diversity.py --seeds 43
# GPUS=0 submits it without a GPU. Log: log/slurm/<name>_<job>.log (NAME, default: job).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
(( $# )) || { echo "Usage: range/gpu.sh COMMAND [ARGS...]" >&2; exit 2; }

name=${NAME:-job}
opts=(--job-name="$name" --cpus-per-task="$EVAL_CPUS" --mem="$EVAL_MEM" --time="$EVAL_TIME"
      --output="$LOG_DIR/${name}_%j.log")
(( ${GPUS:-1} )) && { read -ra gpu <<< "$(gpu_opts "${GPUS:-1}")"; opts+=("${gpu[@]}"); }
id=$(submit "${opts[@]}" ${DEPENDENCY:+--dependency="$DEPENDENCY"} "$RANGE_DIR/job.sbatch" "$@") || exit
echo "Submitted job $id: $*"

#!/usr/bin/env bash
# Submit one command as a job on GPUS A100s of one node (default 1: QOS shared; 4: a whole node,
# QOS regular; QOS overrides), run from sa_data/:
#   nersc/gpu.sh python calculate_diversity.py --seeds 43
# GPUS=0 submits it as a CPU job (CPU_ACCOUNT, CPUS=8, MEM=32G). TIME (default EVAL_TIME) is its
# wall time. Log: log/slurm/<name>_<job>.log (NAME, default: job).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
(( $# )) || { echo "Usage: nersc/gpu.sh COMMAND [ARGS...]" >&2; exit 2; }

name=${NAME:-job}
opts=(--job-name="$name" --time="${TIME:-$EVAL_TIME}" --output="$LOG_DIR/${name}_%j.log")
if (( ${GPUS:-1} )); then
    read -ra resources <<< "$(gpu_opts "${GPUS:-1}" "${QOS:-}")"
else
    read -ra resources <<< "$(cpu_opts "${CPUS:-8}" "${MEM:-32G}")"
fi
opts+=("${resources[@]}")
# A dependency that can no longer be met (e.g. afterok on a failed job) cancels the job.
id=$(submit "${opts[@]}" ${DEPENDENCY:+--dependency="$DEPENDENCY" --kill-on-invalid-dep=yes} \
    "$NERSC_DIR/job.sbatch" "$@") || exit
echo "Submitted job $id: $*"

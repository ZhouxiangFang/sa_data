#!/usr/bin/env bash
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Runs one command in the nlp environment, from sa_data/, with the resources it was
# submitted with (nersc/gpu.sh, nersc/run.sh):  job.sbatch COMMAND [ARGS...]
set -uo pipefail
source "${NERSC_DIR:?submit with nersc/gpu.sh}/common.sh"
cd -- "$SA_DIR" || exit
echo "$(date '+%F %T') on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-none}: $*"
"$@"
status=$?
echo "$(date '+%F %T') exit $status"
exit "$status"

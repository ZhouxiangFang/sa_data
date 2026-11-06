#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One eval.py job of an evaluation array (range/eval.sh or range/train.sh --eval submits it):
#   eval.sbatch MODELS_FILE DELETE [eval.py options]
# Task i evaluates the model on line i+1 of MODELS_FILE (a path, shortcut or repo ID; for a
# train_jobs.py TSV, the checkpoint column). DELETE=1 deletes a trained checkpoint (a local
# folder named ..._seed<seed>) once its evaluation succeeds.
set -uo pipefail
source "${RANGE_DIR:?submit with range/eval.sh}/common.sh"

models_file=$1 delete=$2
shift 2
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$models_file")
model=$(awk -F'\t' '{ print (NF >= 4 ? $4 : $1) }' <<< "$line")
[[ -n $model ]] || { echo "No line $((SLURM_ARRAY_TASK_ID + 1)) in $models_file" >&2; exit 2; }
echo "$(date '+%F %T') eval $(basename "$model") on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python "$SA_DIR/eval.py" --model "$model" "$@"
status=$?
if (( status == 0 && delete )) && [[ -d $model && ! -L $model && -f $model/config.json
                                     && $(basename "$model") =~ _seed[0-9]+$ ]]; then
    rm -rf -- "$model" && echo "Deleted $model"
fi
echo "$(date '+%F %T') exit $status"
exit "$status"

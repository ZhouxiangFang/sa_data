#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One eval.py job of an evaluation array (nersc/eval.sh or nersc/train.sh --eval submits it):
#   eval.sbatch MODELS_FILE DELETE [eval.py options]
# Task i evaluates the model on line i+1 of MODELS_FILE (a path, shortcut or repo ID; for a
# train_jobs.py TSV, the checkpoint column, the third). DELETE=1 deletes a trained checkpoint (a local
# folder named ..._seed<seed>) once its evaluation succeeds.
set -uo pipefail
source "${NERSC_DIR:?submit with nersc/eval.sh}/common.sh"
python -c 'import vllm, transformers, torch; print("vLLM", vllm.__version__, "transformers",
    transformers.__version__, "torch", torch.__version__, "from", vllm.__file__)' 2>&1 | grep '^vLLM'

models_file=$1 delete=$2
shift 2
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$models_file")
model=$(awk -F'\t' '{ print (NF >= 3 ? $3 : $1) }' <<< "$line")
[[ -n $model ]] || { echo "No line $((SLURM_ARRAY_TASK_ID + 1)) in $models_file" >&2; exit 2; }
echo "$(date '+%F %T') eval $(basename "$model") on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
# eval.py reads data/AILuminate.csv only after the IF benchmarks, so a missing copy (RANGE's
# ~/align/data, copied by hand) fails here instead of an hour in.
needs_ailuminate=1
[[ " $* " == *" --skip_safety "* ]] && needs_ailuminate=0
[[ " $* " == *" --datasets "* && " $* " != *" ailuminate "* ]] && needs_ailuminate=0
if (( needs_ailuminate )) && [[ ! -f $ROOT/data/AILuminate.csv ]]; then
    echo "$ROOT/data/AILuminate.csv is missing: copy RANGE's ~/align/data first" >&2
    exit 1
fi

python "$SA_DIR/eval.py" --model "$model" --responses_root "$RESPONSES_ROOT" "$@"
status=$?
if (( status == 0 && delete )) && [[ -d $model && ! -L $model && -f $model/config.json
                                     && $(basename "$model") =~ _seed[0-9]+$ ]]; then
    rm -rf -- "$model" && echo "Deleted $model"
fi
echo "$(date '+%F %T') exit $status"
exit "$status"

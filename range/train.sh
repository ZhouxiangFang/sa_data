#!/usr/bin/env bash
# Submit one run's alignment jobs (align.py, DeepSpeed) as a Slurm array on range, and with
# --eval an evaluation array whose task i starts when training task i succeeds.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: range/train.sh [options] [-- align.py options]

  --models MODEL...      Starting models (default: llama3.1_git20k mistral_git20k
                         olmo2_git20k qwen2.5_git20k, i.e. FocusV857/<model> on the HF Hub)
  --datasets DATASET...  Default: aegis wildguardmix gretel (gretel with --group_by tactic)
  --abbrs ABBR...        Only these groups (default: every non-benign subcategory with
                         enough examples, plus the random and evenly baselines)
  --group_by tactic      Group Gretel's harmful prompts by jailbreak tactic
  --num_train N          Examples per job (default: 1800)
  --harmful_rate R       Harmful fraction (default: 1/3)
  --general_rate R       General-instruction fraction (default: 1/3); benign gets the rest
  --seed SEED            Run seed (default: 43)
  --output_dir DIR       Checkpoint folder (default: /scratch/zf28/ckpts/seed<seed>)
  --results_dir DIR      Also skip jobs already evaluated into DIR (eval.py --output_dir)
  --eval                 Also evaluate each checkpoint (eval.py --output_dir RESULTS_DIR,
                         default results/) once it is trained, then delete it; checkpoints
                         saved by an earlier run are evaluated right away
  --eval_only            Only evaluate the saved checkpoints (implies --eval)
  --dry_run              Print the plan and sbatch commands without submitting

Jobs whose checkpoint is saved (or evaluated) are skipped, so rerunning resumes; rerun
after failures. Each training job uses TRAIN_NGPUS (2) x TRAIN_GPU (h200) with
EFFECTIVE_BATCH (32) examples per update, MICRO_BATCH (4) per GPU, LEARNING_RATE (1e-5) and
ZeRO-2 (DS_CONFIG); see range/common.sh for the other settings.
Logs: log/slurm/align_<job>_<task>.log and eval_<job>_<task>.log; the plan of a
submission is log/slurm/jobs_<time>.tsv (line i+1 = task i).
DEPENDENCY (e.g. afterany:<job>) delays the training array until that job ends.
EOF
}

models=() datasets=() abbrs=() align_args=()
group_by=subcategory num_train=1800 harmful_rate=0.3333333333333333
general_rate=0.3333333333333333 seed=43 output_dir="" results_dir="" eval=0 eval_only=0 dry_run=0
while (( $# )); do
    case $1 in
        --models|--datasets|--abbrs)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        --group_by|--num_train|--harmful_rate|--general_rate|--seed|--output_dir|--results_dir)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"
            shift 2
            ;;
        --eval) eval=1; shift ;;
        --eval_only) eval=1 eval_only=1; shift ;;
        --dry_run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; align_args=("$@"); break ;;
        *) echo "Unknown option: $1 (align.py options go after --)" >&2; exit 2 ;;
    esac
done
output_dir=${output_dir:-/scratch/zf28/ckpts/seed$seed}
(( eval )) && results_dir=${results_dir:-$ROOT/results}
settings=(--group_by "$group_by" --num_train "$num_train" --harmful_rate "$harmful_rate"
          --general_rate "$general_rate" --seed "$seed")

if (( EFFECTIVE_BATCH % (TRAIN_NGPUS * MICRO_BATCH) )); then
    echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH is not a multiple of" \
         "TRAIN_NGPUS x MICRO_BATCH = $((TRAIN_NGPUS * MICRO_BATCH))" >&2
    exit 2
fi
# Put first, so settings given after -- (e.g. --effective_batch_size) still win.
[[ -f $DS_CONFIG ]] || { echo "DeepSpeed config not found: $DS_CONFIG" >&2; exit 2; }
# ZeRO-2 with gradient accumulation computes wrong gradients on DeepSpeed 0.17.6 to 0.19.5
# (issues #7718, #8224): on 0.18.0, 8B checkpoints came out bit-identical to the base; on 0.18.3,
# gradients were 1/accumulation^2 of ZeRO-3's. 0.19.6 matches ZeRO-3. Check the DeepSpeed that
# training will import (TRAIN_VENV's if set).
if [[ ${ALLOW_ZERO2:-0} != 1 ]]; then
    bad_zero2=$("${TRAIN_VENV:+$TRAIN_VENV/bin/}python" - "$DS_CONFIG" <<'EOF'
import json, sys
from importlib.metadata import version
from packaging.version import Version
stage = json.load(open(sys.argv[1])).get("zero_optimization", {}).get("stage")
deepspeed = Version(version("deepspeed"))
if stage == 2 and Version("0.17.6") <= deepspeed < Version("0.19.6"):
    print(deepspeed)
EOF
    ) || exit 2
    if [[ -n $bad_zero2 ]]; then
        echo "$DS_CONFIG is ZeRO-2, and DeepSpeed $bad_zero2 computes wrong gradients for ZeRO-2 with" \
             "gradient accumulation (0.17.6 to 0.19.5; issues #7718, #8224). Use DeepSpeed >= 0.19.6, the" \
             "ZeRO-3 config sa_data/ds_config.json, or set ALLOW_ZERO2=1 to override." >&2
        exit 2
    fi
fi
# align.py derives the accumulation steps: effective batch / (GPUs x micro-batch).
align_args=(--deepspeed "$(realpath "$DS_CONFIG")" --per_device_train_batch_size "$MICRO_BATCH"
            --effective_batch_size "$EFFECTIVE_BATCH" --learning_rate "$LEARNING_RATE"
            "${settings[@]}" "${align_args[@]}")

plan=(--ckpt_dir "$output_dir" "${settings[@]}")
(( ${#models[@]} )) && plan+=(--models "${models[@]}")
(( ${#datasets[@]} )) && plan+=(--datasets "${datasets[@]}")
(( ${#abbrs[@]} )) && plan+=(--abbrs "${abbrs[@]}")
[[ -n $results_dir ]] && plan+=(--results_dir "$results_dir")
jobs_tsv=$(python "$SA_DIR/train_jobs.py" "${plan[@]}" 2> >(grep -v NVML >&2)) || exit $?

train_tsv=$LOG_DIR/jobs_${STAMP}.tsv saved_tsv=$LOG_DIR/jobs_${STAMP}_saved.tsv
declare -A count=([train]=0 [saved]=0 [done]=0)
train_lines=() saved_lines=()
while IFS= read -r line; do
    [[ -n $line ]] || continue
    status=${line##*$'\t'}
    (( ++count[$status] ))
    [[ $status == train ]] && train_lines+=("$line")
    [[ $status == saved ]] && saved_lines+=("$line")
done <<< "$jobs_tsv"
echo "Seed $seed, $group_by groups, $num_train examples: ${count[train]} to train," \
     "${count[saved]} saved, ${count[done]} evaluated. Checkpoints: $output_dir"
(( eval_only )) && train_lines=()

# sbatch_or_echo SBATCH_ARGS...: submits, or with --dry_run prints the command.
sbatch_or_echo() {
    if (( dry_run )); then
        printf 'sbatch' >&2; printf ' %q' "$@" >&2; echo >&2
        echo DRYRUN
    else
        submit "$@"
    fi
}

mkdir -p -- "$output_dir"
read -ra eval_opts <<< "$(gpu_opts 1)"
eval_opts+=(--cpus-per-task="$EVAL_CPUS" --mem="$EVAL_MEM"
           --time="$EVAL_TIME" --kill-on-invalid-dep=yes)
if (( ${#train_lines[@]} )); then
    printf '%s\n' "${train_lines[@]}" > "$train_tsv"
    n=${#train_lines[@]}
    train_id=$(sbatch_or_echo --job-name="align_s$seed" --array="0-$((n - 1))%$MAX_TRAIN" \
        ${DEPENDENCY:+--dependency="$DEPENDENCY" --kill-on-invalid-dep=yes} \
        --gres="gpu:$TRAIN_GPU:$TRAIN_NGPUS" --cpus-per-task="$TRAIN_CPUS" \
        --mem="$TRAIN_MEM" --time="$TRAIN_TIME" --output="$LOG_DIR/align_%A_%a.log" \
        "$RANGE_DIR/train.sbatch" "$train_tsv" "$output_dir" "${align_args[@]}") || exit
    echo "Training: array $train_id, $n job(s) on $TRAIN_NGPUS x $TRAIN_GPU, $MAX_TRAIN at a time" \
         "(plan: $train_tsv)"
    if (( eval )); then
        eval_id=$(sbatch_or_echo --job-name="eval_s$seed" --array="0-$((n - 1))%$MAX_EVAL" \
            --dependency="aftercorr:$train_id" "${eval_opts[@]}" \
            --output="$LOG_DIR/eval_%A_%a.log" \
            "$RANGE_DIR/eval.sbatch" "$train_tsv" 1 --output_dir "$results_dir") || exit
        # Training task i releases (or, if it fails, cancels) eval task i itself; see train.sbatch.
        (( dry_run )) || echo "$eval_id" > "$train_tsv.eval_id"
        echo "Evaluation: array $eval_id on 1 x $EVAL_GPU, each after its training job"
    fi
fi
if (( eval && ${#saved_lines[@]} )); then
    printf '%s\n' "${saved_lines[@]}" > "$saved_tsv"
    n=${#saved_lines[@]}
    eval_id=$(sbatch_or_echo --job-name="eval_s$seed" --array="0-$((n - 1))%$MAX_EVAL" \
        "${eval_opts[@]}" --output="$LOG_DIR/eval_%A_%a.log" \
        "$RANGE_DIR/eval.sbatch" "$saved_tsv" 1 --output_dir "$results_dir") || exit
    echo "Evaluation of saved checkpoints: array $eval_id, $n job(s) on 1 x $EVAL_GPU"
fi

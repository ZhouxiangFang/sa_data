#!/usr/bin/env bash
# Submit one seed of stage 1 or stage 2 on range as dependent Slurm jobs. Returns once queued.
set -uo pipefail
RANGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RANGE/common.sh"

usage() {
    cat <<'EOF'
Usage: range/run.sh [--seed SEED] [--selected DATASET [--mixed_diversity]] [--models MODEL...]
                    [--abbrs GROUP...] [--num_train N] [--response_label_filter safe|refusal]
                    [--skip STAGE...] [--dry_run]

Stage 1 (default): train the git20k models on each WildGuardMix subcategory and the random /
evenly baselines, evaluate them, score the diversity of their training samples and correlate it
with the harmful score. Stage 2 (--selected DATASET: wildguardmix, wildjailbreak, alert_dpo or
nemotron): select subsets from DATASET with every select_data.py method, then train and
evaluate them; --mixed_diversity runs the ablation that scores each metric on both strata
together (metrics only).
  select     stage 2: select_data.py on 1 x EVAL_GPU (SELECT_CPUS=32, SELECT_MEM=192G,
             SELECT_TIME=1-00:00:00), then a CPU job that reruns this command with
             --skip select once the selection succeeds (training is planned from the subsets)
  base       eval.py on the git20k starting models and their official instruct models, into
             results/baselines/ (only those not evaluated yet; the figures' reference bars)
  train      training array, TRAIN_NGPUS x TRAIN_GPU (2 x h200) per job, and an eval array,
             1 x EVAL_GPU per job: task i starts when training task i succeeds (checkpoints are
             kept; DELETE_CKPTS=1 deletes each once evaluated)
  diversity  stage 1: calculate_diversity.py for this seed on 1 x EVAL_GPU (it reads or draws
             the same sample files as training)
  post       one CPU job once the jobs above have ended (range/post.sh): plot_bars_overall.py
             for this seed and, in stage 1, correlate_diversity.py over every seed scored so far
--response_label_filter refusal keeps only harmful examples whose response is a refusal
(WildGuardMix); --num_train sets the examples per run (default 1800, a third harmful). Each
setting has its own folders: results/correlations/<setting>/ (stage 1) or
results/selection/<dataset>.../ (stage 2) for the scores, the same tree under
/scratch/zf28/results for the responses and under figs/ for the figures, and
CKPT_ROOT/seed<seed>/<setting>/ (setting_setup in range/common.sh). Rerunning resumes: training
skips jobs whose checkpoint or scores exist, and selection keeps the subsets already saved.
Watch: squeue --me
EOF
}

args=("$@")
seed=43 models=() abbrs=() skip=() dry_run=0 response_label_filter=safe num_train="" selected=""
mixed=0
while (( $# )); do
    case $1 in
        --seed|--response_label_filter|--num_train|--selected)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        --models|--abbrs|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        --dry_run) dry_run=1; shift ;;
        --mixed_diversity) mixed=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
skipped() { [[ " ${skip[*]} " == *" $1 "* ]]; }
setting_setup || exit 2
(( ${#models[@]} )) || models=(llama3.1_git20k mistral_git20k olmo2_git20k qwen2.5_git20k)
[[ -n $selected ]] && skip+=(diversity) || skip+=(select)

export JOB_IDS_FILE=$LOG_DIR/run_${STAMP}_jobs.txt
: > "$JOB_IDS_FILE"
echo "===== $(date '+%F %T') seed $seed, $setting setting"
# submit_or_echo NAME COMMAND...: a one-GPU job (GPUS=0: CPU), or with --dry_run the command.
submit_or_echo() {
    local name=$1
    shift
    if (( dry_run )); then
        echo "would submit $name${DEPENDENCY:+ ($DEPENDENCY)}: $*"
    else
        NAME=$name "$RANGE/gpu.sh" "$@" || echo "Submitting $name failed" >&2
    fi
}
job_ids() { echo "Job IDs: $(paste -sd' ' "$JOB_IDS_FILE")  (cancel all: scancel \$(cat $JOB_IDS_FILE))"; }

if ! skipped select; then
    select_opts=(--dataset "$selected" --seed "$seed" --size "$harmful" --response_label_filter
                 "$response_label_filter" --threads "${SELECT_CPUS:-32}" --num_proc 16)
    (( mixed )) && select_opts+=(--mixed_diversity)
    EVAL_CPUS=${SELECT_CPUS:-32} EVAL_MEM=${SELECT_MEM:-192G} EVAL_TIME=${SELECT_TIME:-1-00:00:00} \
        submit_or_echo "select_${selected}_s$seed" python select_data.py "${select_opts[@]}"
    deps=$(paste -sd: "$JOB_IDS_FILE")
    # sbatch exports the environment, so the follow-up must not inherit these.
    DEPENDENCY=${deps:+afterok:$deps} GPUS=0 submit_or_echo "run_s$seed" \
        env -u DEPENDENCY -u GPUS -u NAME bash "$RANGE/run.sh" "${args[@]}" --skip select
    job_ids
    exit
fi

if ! skipped base; then
    missing=()
    for model in "${models[@]}" "${models[@]/%_git20k/-ins}"; do
        evaluated "$ROOT/results/baselines/$model.csv" || missing+=("$model")
    done
    if (( ${#missing[@]} == 0 )); then
        echo "Base models already evaluated"
    elif (( dry_run )); then
        echo "would submit: range/eval.sh ${missing[*]}"
    else
        "$RANGE/eval.sh" "${missing[@]}" || echo "Submitting the base evaluations failed" >&2
    fi
fi

if ! skipped train; then
    train_opts=(--seed "$seed" --num_train "$num_train" --response_label_filter
                "$response_label_filter" --models "${models[@]}" --eval)
    [[ -n $selected ]] && train_opts+=(--selected "$selected")
    (( mixed )) && train_opts+=(--mixed_diversity)
    (( ${#abbrs[@]} )) && train_opts+=(--abbrs "${abbrs[@]}")
    (( dry_run )) && train_opts+=(--dry_run)
    "$RANGE/train.sh" "${train_opts[@]}" || echo "Submitting train/eval failed" >&2
fi

if ! skipped diversity; then
    submit_or_echo "diversity_s$seed" python calculate_diversity.py --seeds "$seed" \
        --num_train "$harmful" --response_label_filter "$response_label_filter" \
        --results_dir "$results"
fi

if ! skipped post; then
    post_opts=(--seed "$seed" --num_train "$num_train" --response_label_filter
               "$response_label_filter")
    [[ -n $selected ]] && post_opts+=(--selected "$selected")
    (( mixed )) && post_opts+=(--mixed_diversity)
    deps=$(paste -sd: "$JOB_IDS_FILE")
    DEPENDENCY=${deps:+afterany:$deps} GPUS=0 submit_or_echo "post_s$seed" bash \
        "$RANGE/post.sh" "${post_opts[@]}"
fi
job_ids

#!/usr/bin/env bash
# Submit one seed of stage 1 on range as dependent Slurm jobs. Returns once everything is queued.
set -uo pipefail
RANGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RANGE/common.sh"

usage() {
    cat <<'EOF'
Usage: range/run.sh [--seed SEED] [--models MODEL...] [--abbrs GROUP...] [--num_train N]
                    [--response_label_filter safe|refusal] [--skip STAGE...] [--dry_run]

Stage 1 for one seed (default 43): train the git20k models on each WildGuardMix subcategory and
the random / evenly baselines, evaluate them, score the diversity of their training samples and
correlate it with the harmful score.
  base       eval.py on the git20k starting models and their official instruct models, into
             results/ (only those not evaluated yet; the figures' reference bars)
  train      training array, TRAIN_NGPUS x TRAIN_GPU (2 x h200) per job, and an eval array,
             1 x EVAL_GPU per job: task i starts when training task i succeeds, then deletes
             its checkpoint
  diversity  calculate_diversity.py for this seed on 1 x EVAL_GPU (it reads or draws the same
             sample files as training)
  post       one CPU job once the jobs above have ended (range/post.sh): plot_bars_overall.py
             for this seed and correlate_diversity.py over every seed scored so far
--response_label_filter refusal keeps only harmful examples whose response is a refusal;
--num_train sets the examples per run (default 1800). Both make their own setting: results in
ablations/<setting>/, diversity in diversity/<setting>/ (setting_setup in range/common.sh).
Rerunning resumes: training skips jobs whose checkpoint or results exist. Watch: squeue --me
EOF
}

seed=43 models=() abbrs=() skip=() dry_run=0 response_label_filter=safe num_train=""
while (( $# )); do
    case $1 in
        --seed|--response_label_filter|--num_train)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        --models|--abbrs|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        --dry_run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
skipped() { [[ " ${skip[*]} " == *" $1 "* ]]; }
setting_setup || exit 2
(( ${#models[@]} )) || models=(llama3.1_git20k mistral_git20k olmo2_git20k qwen2.5_git20k)

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

if ! skipped base; then
    missing=()
    for model in "${models[@]}" "${models[@]/%_git20k/-ins}"; do
        [[ -f $ROOT/results/$model/${model}_overall.csv &&
           -f $ROOT/results/if_eval/${model}_if_summary.csv ]] || missing+=("$model")
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
    (( ${#abbrs[@]} )) && train_opts+=(--abbrs "${abbrs[@]}")
    (( dry_run )) && train_opts+=(--dry_run)
    "$RANGE/train.sh" "${train_opts[@]}" || echo "Submitting train/eval failed" >&2
fi

if ! skipped diversity; then
    submit_or_echo "diversity_s$seed" python calculate_diversity.py --seeds "$seed" \
        --num_train "$harmful" --response_label_filter "$response_label_filter" \
        --output_dir "$diversity"
fi

if ! skipped post; then
    deps=$(paste -sd: "$JOB_IDS_FILE")
    DEPENDENCY=${deps:+afterany:$deps} GPUS=0 submit_or_echo "post_s$seed" bash \
        "$RANGE/post.sh" --seed "$seed" --num_train "$num_train" \
        --response_label_filter "$response_label_filter"
fi
echo "Job IDs: $(paste -sd' ' "$JOB_IDS_FILE")  (cancel all: scancel \$(cat $JOB_IDS_FILE))"

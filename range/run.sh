#!/usr/bin/env bash
# Submit one run of the experiment on range as dependent Slurm jobs; every stage uses the
# same seed. Returns once everything is queued.
set -uo pipefail
RANGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RANGE/common.sh"

usage() {
    cat <<'EOF'
Usage: range/run.sh [--seed SEED] [--tactic] [--models MODEL...] [--datasets DATASET...]
                    [--skip STAGE...] [--dry_run]

The run of chili/run.sh (seed 43 by default; --tactic for the Gretel tactic setting),
submitted as Slurm jobs:
  train      training array, TRAIN_NGPUS x TRAIN_GPU (2 x h200) per job
  eval       evaluation array, 1 x EVAL_GPU (an H200 or H100) per job; task i starts when training
             task i succeeds, then deletes its checkpoint
  diversity  calculate_diversity.py on 1 x EVAL_GPU, right away (it draws the same
             samples as training)
  plot, correlate
             one CPU job (range/post.sh) once all the jobs above have ended
--skip leaves stages out. Rerunning resumes: training skips jobs whose checkpoint or
results exist, and saved checkpoints are evaluated. Watch with: squeue --me
EOF
}

seed=43 tactic=0 models=() datasets=() skip=() dry_run=0
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --tactic) tactic=1; shift ;;
        --dry_run) dry_run=1; shift ;;
        --models|--datasets|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
skipped() { [[ " ${skip[*]} " == *" $1 "* ]]; }

if (( tactic )); then
    setting=tactic results=$ROOT/ablations/gretel_tactic datasets=(gretel)
    run_opts=(--group_by tactic --num_train 900)
    diversity_opts=(--group_by tactic --num_train 300 --n_clusters 6)
else
    setting=default results=$ROOT/results
    (( ${#datasets[@]} )) || datasets=(aegis wildguardmix gretel)
    run_opts=() diversity_opts=()
fi
run_opts+=(--seed "$seed" --datasets "${datasets[@]}"
           --output_dir "/scratch/zf28/ckpts/seed$seed/$setting" --results_dir "$results")
(( ${#models[@]} )) && run_opts+=(--models "${models[@]}")
(( dry_run )) && run_opts+=(--dry_run)

export JOB_IDS_FILE=$LOG_DIR/run_${STAMP}_jobs.txt
: > "$JOB_IDS_FILE"
echo "===== $(date '+%F %T') seed $seed, $setting setting"

if ! skipped train || ! skipped eval; then
    train_opts=()
    skipped eval || train_opts+=(--eval)
    skipped train && train_opts+=(--eval_only)
    "$RANGE/train.sh" "${run_opts[@]}" "${train_opts[@]}" || echo "Submitting train/eval failed" >&2
fi

if ! skipped diversity; then
    if (( dry_run )); then
        echo "would submit: calculate_diversity.py --datasets ${datasets[*]} --seeds $seed ${diversity_opts[*]}"
    else
        NAME=diversity_s$seed "$RANGE/gpu.sh" python calculate_diversity.py \
            --datasets "${datasets[@]}" --seeds "$seed" "${diversity_opts[@]}" ||
            echo "Submitting diversity failed" >&2
    fi
fi

if ! skipped plot || ! skipped correlate; then
    post_opts=(--seed "$seed" --datasets "${datasets[@]}")
    (( tactic )) && post_opts+=(--tactic)
    skipped plot && post_opts+=(--skip plot)
    skipped correlate && post_opts+=(--skip correlate)
    deps=$(paste -sd: "$JOB_IDS_FILE")
    if (( dry_run )); then
        echo "would submit: range/post.sh ${post_opts[*]} after jobs: ${deps:-none}"
    else
        DEPENDENCY=${deps:+afterany:$deps} GPUS=0 NAME=post_s$seed \
            "$RANGE/gpu.sh" bash "$RANGE/post.sh" "${post_opts[@]}" ||
            echo "Submitting plot/correlate failed" >&2
    fi
fi
echo "Job IDs: $(paste -sd' ' "$JOB_IDS_FILE")  (cancel all: scancel \$(cat $JOB_IDS_FILE))"

#!/usr/bin/env bash
# Submit one run of the experiment on range as dependent Slurm jobs; every stage uses the
# same seed. Returns once everything is queued.
set -uo pipefail
RANGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RANGE/common.sh"

usage() {
    cat <<'EOF'
Usage: range/run.sh [--seed SEED] [--tactic | --selected] [--models MODEL...]
                    [--datasets DATASET...] [--abbrs GROUP...] [--skip STAGE...]
                    [--response_filter safe|refusal] [--dedup] [--num_train N] [--dry_run]

The run of chili/run.sh (seed 43 by default; --tactic for the Gretel tactic setting,
--selected for the selected-data setting), submitted as Slurm jobs:
  select     --selected only: one select_data.py job per dataset (range/select_data.sh),
             then a CPU job that reruns this command with --skip select once they all
             succeed, since training is planned from the subsets on disk (it submits jobs
             itself; pass --skip select when the subsets exist)
  train      training array, TRAIN_NGPUS x TRAIN_GPU (2 x h200) per job
  eval       evaluation array, 1 x EVAL_GPU (an H200 or H100) per job; task i starts when training
             task i succeeds, then deletes its checkpoint
  diversity  calculate_diversity.py on 1 x EVAL_GPU, right away (it draws the same
             samples as training); not with --selected
  plot, correlate
             one CPU job (range/post.sh) once all the jobs above have ended; not with
             --selected, and plot not with --tactic or pool options
--abbrs trains only the named groups, e.g. --selected --abbrs lexical_hdd random.
Pool options, as in chili/run.sh: --response_filter refusal keeps only harmful examples whose
response is labeled a refusal (wildguardmix only, then the default dataset), --dedup keeps
one row per prompt. They make their own setting named with their tag (refusal, dedup or
refusal_dedup): results in ablations/<tag>/ (ablations/selected_<tag>/,
ablations/gretel_tactic_<tag>/), diversity in diversity/<tag>/, subsets in
data/seed<seed>/<dataset>_<tag>/.
--num_train N sets the examples per run (default 1800, or 900 with --tactic; a third harmful,
which is also the diversity and subset size). Another size makes its own setting, named with
the harmful count: e.g. --response_filter refusal --num_train 1500 is refusal_500, and
--num_train 1500 alone default_500 (see setting_setup in range/common.sh).
--skip leaves stages out. Rerunning resumes: training skips jobs whose checkpoint or
results exist, and saved checkpoints are evaluated. Watch with: squeue --me
EOF
}

args=("$@")
seed=43 tactic=0 selected=0 models=() datasets=() abbrs=() skip=() dry_run=0
response_filter=safe dedup=0 num_train=""
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --tactic) tactic=1; shift ;;
        --selected) selected=1; shift ;;
        --response_filter|--num_train)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        --dedup) dedup=1; shift ;;
        --dry_run) dry_run=1; shift ;;
        --models|--datasets|--abbrs|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
skipped() { [[ " ${skip[*]} " == *" $1 "* ]]; }

(( tactic && selected )) && { echo "--tactic and --selected are different settings" >&2; exit 2; }
setting_setup || exit 2
# Selected subsets have no diversity CSV; only they are selected.
(( selected )) && skip+=(diversity plot correlate) || skip+=(select)
run_opts=(--group_by "$group_by" --num_train "$num_train" --seed "$seed"
          --datasets "${datasets[@]}" "${pool_opts[@]}"
          --output_dir "$CKPT_ROOT/seed$seed/$setting" --results_dir "$results")
(( ${#models[@]} )) && run_opts+=(--models "${models[@]}")
(( ${#abbrs[@]} )) && run_opts+=(--abbrs "${abbrs[@]}")
(( dry_run )) && run_opts+=(--dry_run)
diversity_opts=(--num_train "$harmful" --output_dir "$diversity_root" "${pool_opts[@]}")
(( tactic )) && diversity_opts+=(--group_by tactic)

export JOB_IDS_FILE=$LOG_DIR/run_${STAMP}_jobs.txt
: > "$JOB_IDS_FILE"
echo "===== $(date '+%F %T') seed $seed, $setting setting"
job_ids() { echo "Job IDs: $(paste -sd' ' "$JOB_IDS_FILE")  (cancel all: scancel \$(cat $JOB_IDS_FILE))"; }

if ! skipped select; then
    select_opts=(--seed "$seed" --datasets "${datasets[@]}")
    (( dry_run )) && select_opts+=(--dry_run)
    "$RANGE/select_data.sh" "${select_opts[@]}" -- --size "$harmful" "${pool_opts[@]}" ||
        { echo "Submitting selection failed" >&2; exit 1; }
    if (( dry_run )); then
        echo "would submit, after the selection jobs: range/run.sh ${args[*]} --skip select"
    else
        # sbatch exports the environment, so the job must not inherit these.
        DEPENDENCY=afterok:$(paste -sd: "$JOB_IDS_FILE") GPUS=0 NAME=run_s$seed \
            "$RANGE/gpu.sh" env -u DEPENDENCY -u GPUS -u NAME \
            bash "$RANGE/run.sh" "${args[@]}" --skip select ||
            echo "Submitting the training step failed; rerun with --skip select" >&2
    fi
    job_ids
    exit
fi

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
    post_opts=(--seed "$seed" --datasets "${datasets[@]}" --num_train "$num_train" "${pool_opts[@]}")
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
job_ids

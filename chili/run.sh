#!/usr/bin/env bash
# One run of the experiment on chili; every stage uses the same seed.
set -uo pipefail
CHILI=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$CHILI/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/run.sh [--seed SEED] [--tactic | --selected] [--models MODEL...]
                    [--datasets DATASET...] [--abbrs GROUP...] [--skip STAGE...]
                    [--response_filter safe|refusal] [--dedup] [--num_train N]

Default run (seed 43): the git20k models (FocusV857/<model>_git20k), each trained on
1,800 examples, harmful : WildJailbreak benign : general instructions = 1:1:1, with the
harmful part from one subcategory of aegis, wildguardmix or gretel (every non-benign
subcategory with 600 examples, plus the random and evenly baselines).
  results    results/seed<seed>/, IF in results/seed<seed>/if_eval/
  diversity  diversity/seed<seed>/<embedder>/ (means over seeds: diversity/<embedder>/)
  figures    figs/seed<seed>/git20k_1800_<dataset>/
--tactic runs the Gretel tactic setting instead: 900 examples (300 each), the harmful
part from one jailbreak tactic; results in ablations/gretel_tactic/seed<seed>/.
--selected runs the selected-data setting instead: the default mixture, with the harmful
part a 600-example subset chosen by select_data.py (one run per diversity metric and
baseline strategy; data/seed<seed>/<dataset>/); results in ablations/selected/seed<seed>/.
--abbrs trains only the named groups, e.g. --selected --abbrs semantic_vendi_qwen3-embedding-0.6b lexical_hdd.
Pool options apply to every stage alike (training, diversity, selection):
--response_filter refusal keeps only harmful examples whose response is labeled a refusal
(wildguardmix only, its default dataset then; default safe: any response labeled safe,
which includes harmless compliances), and --dedup keeps one row per prompt (default: every
row). They make their own setting, named with the tag refusal, dedup or refusal_dedup:
results in ablations/<tag>/ (ablations/selected_<tag>/, ablations/gretel_tactic_<tag>/),
checkpoints in .../seed<seed>/<setting>/, diversity in diversity/<tag>/, selected subsets
in data/seed<seed>/<dataset>_<tag>/; the plot stage is skipped.
--num_train N sets the examples per run (default 1800, or 900 with --tactic; a third harmful,
which is also the diversity and subset size). Another size makes its own setting, named with
the harmful count: e.g. --response_filter refusal --num_train 1500 is refusal_500 (results in
ablations/refusal_500/, diversity in diversity/refusal_500/), --num_train 1500 alone default_500.

Stages: select (--selected only), train, eval, diversity, plot (default setting only),
correlate (not with --selected: its subsets have no diversity CSV); --skip leaves
some out. A failed stage is reported and the next one still runs. Rerunning resumes:
training skips jobs whose checkpoint or results exist, and evaluation deletes each
checkpoint ($CKPT_ROOT/seed<seed>/<setting>/, CKPT_ROOT=/scratch/zf28/ckpts) once it is evaluated.
EOF
}

seed=43 tactic=0 selected=0 models=() datasets=() abbrs=() skip=() response_filter=safe dedup=0
num_train=""
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --tactic) tactic=1; shift ;;
        --selected) selected=1; shift ;;
        --response_filter|--num_train)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        --dedup) dedup=1; shift ;;
        --models|--datasets|--abbrs|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

(( tactic && selected )) && { echo "--tactic and --selected are different settings" >&2; exit 2; }
# The setting's folders and sizes (sa_data/settings.sh); pool options apply to every stage.
setting_setup || exit 2
ckpt_dir=$CKPT_ROOT/seed$seed/$setting
run_opts=(--group_by "$group_by" --num_train "$num_train" --seed "$seed"
          --datasets "${datasets[@]}" "${pool_opts[@]}")
(( ${#models[@]} )) && run_opts+=(--models "${models[@]}")
(( ${#abbrs[@]} )) && run_opts+=(--abbrs "${abbrs[@]}")
diversity_opts=(--num_train "$harmful" --output_dir "$diversity_root" "${pool_opts[@]}")
(( tactic )) && diversity_opts+=(--group_by tactic)

select_data() {
    "$CHILI/select_data.sh" --seed "$seed" --datasets "${datasets[@]}" -- --size "$harmful" \
        "${pool_opts[@]}"
}

train() {
    "$CHILI/train.sh" "${run_opts[@]}" --output_dir "$ckpt_dir" --results_dir "$results"
}

evaluate() {
    local plan
    plan=$(python "$SA_DIR/train_jobs.py" "${run_opts[@]}" --ckpt_dir "$ckpt_dir" \
           --results_dir "$results" 2>/dev/null) || return
    local -a saved
    mapfile -t saved < <(awk -F'\t' '$5 == "saved" { print $4 }' <<< "$plan")
    (( ${#saved[@]} )) || { echo "No saved checkpoints to evaluate."; return; }
    "$CHILI/eval.sh" "${saved[@]}" --delete -- --output_dir "$results"
}

diversity() {
    "$CHILI/gpu.sh" python "$SA_DIR/calculate_diversity.py" --datasets "${datasets[@]}" \
        --seeds "$seed" "${diversity_opts[@]}"
}

failed=()
stage() {
    local name=$1 status
    shift
    if [[ " ${skip[*]} " == *" $name "* ]] ||
       { [[ $name == plot && $setting != default ]]; } ||
       { [[ $name == select ]] && (( ! selected )); } ||
       { [[ $name == diversity || $name == plot || $name == correlate ]] && (( selected )); }; then
        echo "===== $(date '+%F %T') SKIPPED: $name"
        return
    fi
    echo "===== $(date '+%F %T') START: $name (seed $seed, $setting setting)"
    "$@"
    status=$?
    if (( status == 0 )); then
        echo "===== $(date '+%F %T') DONE: $name"
    else
        echo "===== $(date '+%F %T') FAILED: $name (exit $status)" >&2
        failed+=("$name")
    fi
}

stage select select_data
stage train train
stage eval evaluate
stage diversity diversity
stage plot plot_stage
stage correlate correlate_stage
if (( ${#failed[@]} )); then
    echo "Run finished with failed stage(s): ${failed[*]}" >&2
    exit 1
fi
echo "Run finished; all stages succeeded."

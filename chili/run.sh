#!/usr/bin/env bash
# One run of the experiment on chili; every stage uses the same seed.
set -uo pipefail
CHILI=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$CHILI/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/run.sh [--seed SEED] [--tactic | --selected] [--models MODEL...]
                    [--datasets DATASET...] [--abbrs GROUP...] [--skip STAGE...]

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
--abbrs trains only the named groups, e.g. --selected --abbrs semantic_vendi_bge-m3 lexical_hdd.

Stages: select (--selected only), train, eval, diversity, plot (default setting only),
correlate (not with --selected: its subsets have no diversity CSV); --skip leaves
some out. A failed stage is reported and the next one still runs. Rerunning resumes:
training skips jobs whose checkpoint or results exist, and evaluation deletes each
checkpoint (/scratch/zf28/ckpts/seed<seed>/{default,tactic,selected}/) once it is evaluated.
EOF
}

seed=43 tactic=0 selected=0 models=() datasets=() abbrs=() skip=()
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --tactic) tactic=1; shift ;;
        --selected) selected=1; shift ;;
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
embedders=(bge-m3 qwen3-embedding-0.6b)
if (( selected )); then
    setting=selected results=$ROOT/ablations/selected
    (( ${#datasets[@]} )) || datasets=(aegis wildguardmix gretel)
    run_opts=(--group_by selection) diversity_opts=()
elif (( tactic )); then
    setting=tactic results=$ROOT/ablations/gretel_tactic datasets=(gretel)
    run_opts=(--group_by tactic --num_train 900)
    diversity_opts=(--group_by tactic --num_train 300 --n_clusters 6)
else
    setting=default results=$ROOT/results
    (( ${#datasets[@]} )) || datasets=(aegis wildguardmix gretel)
    run_opts=() diversity_opts=()
fi
ckpt_dir=/scratch/zf28/ckpts/seed$seed/$setting
run_opts+=(--seed "$seed" --datasets "${datasets[@]}")
(( ${#models[@]} )) && run_opts+=(--models "${models[@]}")
(( ${#abbrs[@]} )) && run_opts+=(--abbrs "${abbrs[@]}")

select_data() {
    "$CHILI/select.sh" --seed "$seed" --datasets "${datasets[@]}"
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

plot() {
    local dataset status=0
    for dataset in "${datasets[@]}"; do
        python "$SA_DIR/plot_bars_overall.py" --seed "$seed" --suffix "1800_$dataset" || status=1
    done
    return "$status"
}

# Diversity vs harmful score over every seed scored so far, then the summary tables.
correlate() {
    local embedder dataset status=0
    for embedder in "${embedders[@]}"; do
        if (( tactic )); then
            python "$SA_DIR/plot_bars_diversity.py" \
                "$ROOT/diversity/$embedder/csv/gretel_train_tactic_diversity_300_1.0.csv" \
                --results-dir "$results" --suffix 900_gretel || status=1
            continue
        fi
        for dataset in "${datasets[@]}"; do
            python "$SA_DIR/plot_bars_diversity.py" \
                "$ROOT/diversity/$embedder/csv/${dataset}_train_diversity_600_1.0.csv" \
                --suffix "1800_$dataset" || status=1
        done
    done
    if (( tactic )); then
        python "$SA_DIR/summarize_diversity.py" --datasets gretel_tactic || status=1
    else
        python "$SA_DIR/summarize_diversity.py" --datasets "${datasets[@]}" || status=1
    fi
    return "$status"
}

failed=()
stage() {
    local name=$1 status
    shift
    if [[ " ${skip[*]} " == *" $name "* ]] || { [[ $name == plot ]] && (( tactic )); } ||
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
stage plot plot
stage correlate correlate
if (( ${#failed[@]} )); then
    echo "Run finished with failed stage(s): ${failed[*]}" >&2
    exit 1
fi
echo "Run finished; all stages succeeded."

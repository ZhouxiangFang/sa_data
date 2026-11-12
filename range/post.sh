#!/usr/bin/env bash
# The CPU stages of a run (figures, diversity vs harmful score, summary tables), as in
# chili/run.sh; range/run.sh submits it after the GPU jobs. Runs anywhere:
#   range/post.sh --seed SEED [--tactic] [--datasets DATASET...] [--skip plot|correlate]
#                 [--response_filter safe|refusal] [--dedup]
# With pool options, the results are in ablations/<tag>/ (ablations/gretel_tactic_<tag>/),
# the diversity scores in diversity/<tag>/, and plot is skipped (as in chili/run.sh).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

seed=43 tactic=0 datasets=() skip=() response_filter=safe dedup=0
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --tactic) tactic=1; shift ;;
        --response_filter)
            (( $# >= 2 )) || { echo "--response_filter needs a value" >&2; exit 2; }
            response_filter=$2; shift 2 ;;
        --dedup) dedup=1; shift ;;
        --datasets|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
embedders=(bge-m3 qwen3-embedding-0.6b)
(( tactic )) && datasets=(gretel)
pool_setup || exit 2
if (( tactic )); then
    results=$ROOT/ablations/gretel_tactic${tag:+_$tag}
elif [[ -n $tag ]]; then
    results=$ROOT/ablations/$tag
else
    results=$ROOT/results
fi
(( ${#datasets[@]} )) || datasets=(aegis wildguardmix gretel)
diversity_root=$ROOT/diversity${tag:+/$tag}
cd -- "$SA_DIR" || exit

plot() {
    local dataset status=0
    for dataset in "${datasets[@]}"; do
        python plot_bars_overall.py --seed "$seed" --suffix "1800_$dataset" || status=1
    done
    return "$status"
}

# Diversity vs harmful score over every seed scored so far, then the summary tables.
correlate() {
    local embedder dataset status=0
    for embedder in "${embedders[@]}"; do
        if (( tactic )); then
            python plot_bars_diversity.py \
                "$diversity_root/$embedder/csv/gretel_train_tactic_diversity_300_1.0.csv" \
                --results-dir "$results" --suffix 900_gretel || status=1
            continue
        fi
        for dataset in "${datasets[@]}"; do
            python plot_bars_diversity.py \
                "$diversity_root/$embedder/csv/${dataset}_train_diversity_600_1.0.csv" \
                --results-dir "$results" --suffix "1800_$dataset" || status=1
        done
    done
    if (( tactic )); then
        python summarize_diversity.py --diversity_dir "$diversity_root" \
            --datasets gretel_tactic || status=1
    else
        python summarize_diversity.py --diversity_dir "$diversity_root" \
            --datasets "${datasets[@]}" || status=1
    fi
    return "$status"
}

failed=()
for name in plot correlate; do
    if [[ " ${skip[*]} " == *" $name "* ]] ||
       { [[ $name == plot ]] && { (( tactic )) || [[ -n $tag ]]; }; }; then
        echo "===== $(date '+%F %T') SKIPPED: $name"
        continue
    fi
    echo "===== $(date '+%F %T') START: $name (seed $seed)"
    if "$name"; then
        echo "===== $(date '+%F %T') DONE: $name"
    else
        echo "===== $(date '+%F %T') FAILED: $name" >&2
        failed+=("$name")
    fi
done
(( ${#failed[@]} == 0 )) || { echo "Failed: ${failed[*]}" >&2; exit 1; }

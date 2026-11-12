# Run settings shared by the chili and range launchers (chili/common.sh and range/common.sh
# source this; SA_DIR and ROOT must be set): option helpers, the pool options, how a setting
# names its folders, and the CPU stages that follow a run (plot_stage, correlate_stage).

export CKPT_ROOT=${CKPT_ROOT:-/scratch/zf28/ckpts}  # checkpoints; align.py reads it too
TRAIN_DATASETS=(aegis wildguardmix gretel)         # helper.TRAIN_DATASETS

# Splits comma- or space-separated values into the named array: add_values ARRAY VALUE...
add_values() {
    local -n target=$1
    local value
    local -a parts
    shift
    for value; do
        IFS=, read -ra parts <<< "$value"
        target+=("${parts[@]}")
    done
}

# pool_setup: checks response_filter (safe or refusal) and dedup (0 or 1), the pool options
# of helper.PoolOptions, and sets pool_opts (their python flags) and tag ('' for the
# defaults, else refusal, dedup or refusal_dedup: the suffix of the option's folders). With
# the refusal filter, an empty datasets array becomes (wildguardmix), the only dataset with
# refusal labels, and any other dataset is an error. Returns 2 on bad options.
pool_setup() {
    local dataset
    case $response_filter in
        safe|refusal) ;;
        *) echo "--response_filter is safe or refusal, not $response_filter" >&2; return 2 ;;
    esac
    pool_opts=(--response_filter "$response_filter") tag=""
    [[ $response_filter == safe ]] || tag=$response_filter
    (( dedup )) && pool_opts+=(--dedup) tag=${tag:+${tag}_}dedup
    [[ $response_filter == refusal ]] || return 0
    (( ${#datasets[@]} )) || datasets=(wildguardmix)
    for dataset in "${datasets[@]}"; do
        [[ $dataset == wildguardmix ]] && continue
        echo "--response_filter refusal: only wildguardmix labels refusals, not $dataset" >&2
        return 2
    done
}

# setting_setup: resolves the setting of run.sh and post.sh from tactic, selected (0 or 1),
# num_train (examples per run; empty: 1800, or 900 with --tactic) and the pool options
# (pool_setup), and sets:
#   setting         default, tactic or selected, with the pool tag and, for another size, the
#                   harmful count appended (refusal, tactic_dedup, refusal_500, default_500)
#   results         results/ for the default setting, else ablations/<setting>
#                   (ablations/gretel_tactic... for --tactic)
#   diversity_root  diversity/, diversity/<tag> with pool options, diversity/<setting> for
#                   another size
#   harmful         harmful examples per run, the diversity sample size (align.py's rounding)
#   group_by        subcategory, tactic or selection
# and defaults datasets (gretel with --tactic, wildguardmix with the refusal filter).
setting_setup() {
    (( tactic )) && datasets=(gretel)
    pool_setup || return
    (( ${#datasets[@]} )) || datasets=("${TRAIN_DATASETS[@]}")
    local base=default default_size=1800 size="" suffix
    if (( tactic )); then base=tactic default_size=900
    elif (( selected )); then base=selected
    fi
    group_by=${base/default/subcategory}
    group_by=${group_by/selected/selection}
    num_train=${num_train:-$default_size}
    harmful=$(( (2 * num_train + 3) / 6 ))  # int(num_train / 3 + 0.5), as helper.mixture_counts
    (( num_train == default_size )) || size=$harmful
    suffix=$tag${tag:+${size:+_}}$size  # refusal, 500, refusal_500 or empty
    if [[ $base == default ]]; then
        setting=${suffix:-default}
        [[ $suffix == "$size" && -n $size ]] && setting=default_$size
        results=$ROOT/results
        [[ -n $suffix ]] && results=$ROOT/ablations/$setting
    else
        setting=$base${suffix:+_$suffix}
        results=$ROOT/ablations/${setting/tactic/gretel_tactic}
    fi
    diversity_root=$ROOT/diversity${tag:+/$tag}
    [[ -n $size ]] && diversity_root=$ROOT/diversity/$setting
    return 0
}

# plot_stage: figures of the default setting's runs of seed (plot_bars_overall.py; its figures
# also need the instruct baselines in results/). Needs setting_setup's variables.
plot_stage() {
    local dataset status=0
    for dataset in "${datasets[@]}"; do
        python "$SA_DIR/plot_bars_overall.py" --seed "$seed" --suffix "${num_train}_$dataset" ||
            status=1
    done
    return "$status"
}

# correlate_stage: diversity vs harmful score over every seed scored so far, for each embedding
# model scored under diversity_root (plot_bars_diversity.py), then the summary tables
# (summarize_diversity.py). Needs setting_setup's variables.
correlate_stage() {
    local csv dataset status=0 grouping=""
    local -a names=("${datasets[@]}")
    # Tactic diversity CSVs and summaries are named <dataset>_train_tactic_* and <dataset>_tactic.
    (( tactic )) && grouping=tactic_ names=("${datasets[@]/%/_tactic}")
    for dataset in "${datasets[@]}"; do
        for csv in "$diversity_root"/*/csv/"${dataset}_train_${grouping}diversity_${harmful}_1.0.csv"; do
            [[ -f $csv ]] || { echo "No diversity CSV $csv" >&2; status=1; continue; }
            python "$SA_DIR/plot_bars_diversity.py" "$csv" --results-dir "$results" \
                --suffix "${num_train}_$dataset" || status=1
        done
    done
    python "$SA_DIR/summarize_diversity.py" --diversity_dir "$diversity_root" \
        --datasets "${names[@]}" || status=1
    return "$status"
}

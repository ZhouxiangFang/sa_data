#!/usr/bin/env bash
set -uo pipefail

# Gretel tactic setting: 300 harmful prompts from one jailbreak tactic + 300
# WildJailbreak vanilla benign + 300 general instructions = 900 examples (1:1:1),
# trained from four 20k models, then evaluated and correlated with diversity.
# Tactics with at least 300 eligible rows are trained on their own, plus the
# random and evenly baselines over those tactics (align_run.sh --group_by tactic).
# Diversity comes from
#   calculate_diversity.py --datasets gretel --group_by tactic --num_train 300 --n_clusters 6
# (diversity/<embedder>/csv/gretel_train_tactic_diversity_300_1.0.csv, the mean over
# seeds; each seed's in diversity/seed<seed>/<embedder>/csv/).
# Checkpoints and results are kept apart from other runs:
# /scratch/zf28/ckpts/gretel_tactic/ and ablations/gretel_tactic/seed<seed>/ (IF results in
# ablations/gretel_tactic/seed<seed>/if_eval/). align_run.sh trains seeds 42, 84, 126. eval_run.sh deletes checkpoints only after both
# evaluations succeed. align_run.sh skips saved checkpoints, so rerunning resumes.
# The correlate stage writes diversity/<embedder>/{csv,figs}/gretel_tactic_* (seed means;
# each seed alone in diversity/seed<seed>/<embedder>/) for the default embedders and
# diversity/summary/gretel_tactic_correlation_by_embedding_model.csv.
# SKIP_STAGES lists comma-separated stage names to skip, e.g. SKIP_STAGES="train".

# This launcher is in sa_data/script/; align_run.sh and eval_run.sh are in sa_data/.
sa_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=$(cd -- "$sa_dir/.." && pwd)
models_dir=/home/zf28/models/20k
models=(llama3.1_git20k mistral_git20k olmo2_git20k qwen2.5_git20k)
embedders=(bge-m3 qwen3-embedding-0.6b)
tag=gretel_tactic
ckpt_dir=/scratch/zf28/ckpts/$tag
results_dir=$root/ablations/$tag
failed=()

stage() {
    local name="$1" status
    shift
    echo
    if [[ ",${SKIP_STAGES:-}," == *",$name,"* ]]; then
        echo "===== $(date '+%F %T') SKIPPED: $name ====="
        return
    fi
    echo "===== $(date '+%F %T') START: $name ====="
    "$@"
    status=$?
    if (( status == 0 )); then
        echo "===== $(date '+%F %T') DONE: $name ====="
    else
        echo "===== $(date '+%F %T') FAILED: $name (exit $status) =====" >&2
        failed+=("$name")
    fi
}

correlate() {
    local embedder status=0
    for embedder in "${embedders[@]}"; do
        python "$sa_dir/plot_bars_diversity.py" \
            "$root/diversity/$embedder/csv/gretel_train_tactic_diversity_300_1.0.csv" \
            --results-dir "$results_dir" --suffix 900_gretel || status=1
    done
    python "$sa_dir/script/summarize_diversity.py" --datasets gretel_tactic || status=1
    return "$status"
}

stage train \
    "$sa_dir/align_run.sh" --models "${models[@]/#/$models_dir/}" \
    --datasets gretel --group_by tactic --num_train 900 \
    --output_dir "$ckpt_dir"
stage evaluate \
    "$sa_dir/eval_run.sh" --folder "$ckpt_dir" \
    --safe-output-dir "$results_dir" --if-output-dir "$results_dir/if_eval"
stage correlate correlate

echo
if (( ${#failed[@]} > 0 )); then
    echo "Pipeline finished with ${#failed[@]} failed stage(s):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "Pipeline finished; all stages succeeded."

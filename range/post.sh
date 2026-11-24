#!/usr/bin/env bash
# The CPU stage of a run (range/run.sh submits it after the GPU jobs; it also runs anywhere):
#   range/post.sh [--seed SEED] [--num_train N] [--response_label_filter safe|refusal]
# plot_bars_overall.py for the seed's runs, then correlate_diversity.py over every seed scored.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

seed=43 response_label_filter=safe num_train=""
while (( $# )); do
    case $1 in
        --seed|--response_label_filter|--num_train)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
setting_setup || exit 2
cd -- "$SA_DIR" || exit
status=0
python plot_bars_overall.py --results_dir "$results" --seeds "$seed" --num_train "$num_train" ||
    status=1
python correlate_diversity.py --results_dir "$results" --diversity_dir "$diversity" \
    --num_train "$num_train" || status=1
exit "$status"

#!/usr/bin/env bash
# The CPU stages of a run (figures, diversity vs harmful score, summary tables), as in
# chili/run.sh; range/run.sh submits it after the GPU jobs. Runs anywhere:
#   range/post.sh --seed SEED [--tactic] [--datasets DATASET...] [--skip plot|correlate]
#                 [--response_filter safe|refusal] [--dedup] [--num_train N]
# The setting's folders and sizes are resolved as in range/run.sh (setting_setup), and the
# stages are plot_stage (default setting only) and correlate_stage, from sa_data/settings.sh.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

seed=43 tactic=0 selected=0 datasets=() skip=() response_filter=safe dedup=0 num_train=""
while (( $# )); do
    case $1 in
        --tactic) tactic=1; shift ;;
        --dedup) dedup=1; shift ;;
        --seed|--response_filter|--num_train)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        --datasets|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
setting_setup || exit 2
[[ $setting == default ]] || skip+=(plot)

failed=()
for name in plot correlate; do
    if [[ " ${skip[*]} " == *" $name "* ]]; then
        echo "===== $(date '+%F %T') SKIPPED: $name"
        continue
    fi
    echo "===== $(date '+%F %T') START: $name (seed $seed, $setting setting)"
    if "${name}_stage"; then
        echo "===== $(date '+%F %T') DONE: $name"
    else
        echo "===== $(date '+%F %T') FAILED: $name" >&2
        failed+=("$name")
    fi
done
(( ${#failed[@]} == 0 )) || { echo "Failed: ${failed[*]}" >&2; exit 1; }

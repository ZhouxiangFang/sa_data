#!/usr/bin/env bash
set -uo pipefail

# Queued on 2026-10-25 in tmux session align-1:
#   1. Gretel safety test (validation+test) for the official and self-trained
#      (20k) instruct models, merged into their existing overall CSVs.
#   2. For aegis, wildguardmix, then gretel: train every eligible subcategory
#      plus the random and evenly baselines from each 20k model, then run IF,
#      safety and overrefusal evaluation. eval_run.sh deletes the checkpoints
#      only after both evaluations succeed.
# A failed stage is reported and the next stage still runs.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
models_dir=/home/zf28/models/20k
ckpt_root=/scratch/zf28/ckpts
failed=()

stage() {
    local name="$1"
    shift
    echo
    echo "===== $(date '+%F %T') START: $name ====="
    if "$@"; then
        echo "===== $(date '+%F %T') DONE: $name ====="
    else
        echo "===== $(date '+%F %T') FAILED: $name (exit $?) =====" >&2
        failed+=("$name")
    fi
}

stage "gretel test for instruct models" \
    "$script_dir/eval_safe_run.sh" \
    --models llama3-ins,llama3.1-ins,mistral-ins,olmo2-ins,qwen2.5-ins,qwen3-ins \
    --folder "$models_dir" --datasets gretel --merge

for dataset in aegis wildguardmix gretel; do
    ckpt_dir="$ckpt_root/20k_$dataset"
    stage "train $dataset" \
        "$script_dir/align_run.sh" --folder "$models_dir" --datasets "$dataset" \
        --output_dir "$ckpt_dir"
    stage "evaluate $dataset" "$script_dir/eval_run.sh" --folder "$ckpt_dir"
done

echo
if (( ${#failed[@]} > 0 )); then
    echo "Pipeline finished with ${#failed[@]} failed stage(s):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "Pipeline finished; all stages succeeded."

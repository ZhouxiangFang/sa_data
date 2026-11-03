#!/usr/bin/env bash
set -uo pipefail

# No-benign ablation: 600 harmful (evenly) + 600 general instructions = 1,200
# examples (harmful_rate 0.5, general_rate 0.5, no WildJailbreak benign data),
# trained from four 20k models on aegis, wildguardmix and gretel, then evaluated.
# Checkpoints and results are kept apart from other runs:
# /scratch/zf28/ckpts/h0.5_nobenign/ and ablations/h0.5_nobenign/ (IF results in
# ablations/h0.5_nobenign/if_eval/). eval_run.sh deletes checkpoints only after both
# evaluations succeed. align_run.sh skips saved checkpoints, so rerunning resumes.
# SKIP_STAGES lists comma-separated stage names to skip, e.g. SKIP_STAGES="train".

# This launcher is in sa_data/script/; align_run.sh and eval_run.sh are in sa_data/.
sa_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
models_dir=/home/zf28/models/20k
models=(llama3.1_git20k mistral_git20k olmo2_git20k qwen2.5_git20k)
tag=h0.5_nobenign
ckpt_dir=/scratch/zf28/ckpts/$tag
results_dir=$sa_dir/../ablations/$tag
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

stage train \
    "$sa_dir/align_run.sh" --models "${models[@]/#/$models_dir/}" \
    --datasets aegis wildguardmix gretel --abbrs evenly \
    --num_train 1200 --harmful_rate 0.5 --general_rate 0.5 \
    --output_dir "$ckpt_dir"
stage evaluate \
    "$sa_dir/eval_run.sh" --folder "$ckpt_dir" \
    --safe-output-dir "$results_dir" --if-output-dir "$results_dir/if_eval"

echo
if (( ${#failed[@]} > 0 )); then
    echo "Pipeline finished with ${#failed[@]} failed stage(s):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "Pipeline finished; all stages succeeded."

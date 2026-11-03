#!/usr/bin/env bash
set -uo pipefail

# Lower-harmful-rate experiment: 600 harmful (evenly) + 1,200 WildJailbreak benign
# + 1,200 general instructions = 3,000 examples (rates 0.2 / 0.4 / 0.4), trained
# from four 20k models on aegis, wildguardmix and gretel, once with vanilla benign
# data and once with equal halves of vanilla and adversarial benign data (mix).
# Each benign variant keeps its checkpoints and results apart from the 1,800-example
# runs: /scratch/zf28/ckpts/h0.2_<variant>/ and ablations/h0.2_<variant>/ (IF results
# in ablations/h0.2_<variant>/if_eval/). Each variant is trained then evaluated
# before the next starts; eval_run.sh deletes checkpoints only after both
# evaluations succeed. align_run.sh skips saved checkpoints, so rerunning resumes.
# SKIP_STAGES lists comma-separated stage names to skip, e.g.
# SKIP_STAGES="train vanilla_benign".

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
models_dir=/home/zf28/models/20k
models=(llama3.1_git20k mistral_git20k olmo2_git20k qwen2.5_git20k)
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

for variant in vanilla_benign mix; do
    tag=h0.2_${variant%_benign}
    ckpt_dir=/scratch/zf28/ckpts/$tag
    results_dir=$script_dir/../ablations/$tag
    stage "train $variant" \
        "$script_dir/align_run.sh" --models "${models[@]/#/$models_dir/}" \
        --datasets aegis wildguardmix gretel --abbrs evenly \
        --num_train 3000 --harmful_rate 0.2 --general_rate 0.4 \
        --benign_data_type "$variant" --output_dir "$ckpt_dir"
    stage "evaluate $variant" \
        "$script_dir/eval_run.sh" --folder "$ckpt_dir" \
        --safe-output-dir "$results_dir" --if-output-dir "$results_dir/if_eval"
done

echo
if (( ${#failed[@]} > 0 )); then
    echo "Pipeline finished with ${#failed[@]} failed stage(s):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "Pipeline finished; all stages succeeded."

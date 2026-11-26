#!/usr/bin/env bash
# One seed of stage 1 or stage 2 on chili, stage after stage (run it in tmux; rerun to resume).
set -uo pipefail
CHILI=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$CHILI/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/run.sh [--seed SEED] [--selected DATASET [--with_unnatural] [--mixed_diversity]]
                    [--models MODEL...]
                    [--abbrs GROUP...] [--num_train N] [--response_label_filter safe|refusal]
                    [--skip STAGE...] [--dry_run]

The chili counterpart of range/run.sh (same options, settings and output folders).
Stage 1 (default): train the git20k models on each WildGuardMix subcategory and the random /
evenly baselines, evaluate them, score the diversity of their training samples and correlate it
with the harmful score. Stage 2 (--selected DATASET: wildguardmix, wildjailbreak, alert_dpo or
nemotron): select subsets from DATASET with every select_data.py method, then train and
evaluate them; --mixed_diversity runs the ablation that scores each metric on both strata
together (metrics only), --with_unnatural the one that keeps the jailbreak types whose prompts
are not natural language (alert_dpo, nemotron; every method; select_data.py drops them by
default).
  select     stage 2: select_data.py on one free GPU (SELECT_THREADS=32, cache SELECTION_CACHE);
             if it fails, the run stops (as range's afterok)
  base       eval.py on the git20k starting models and their official instruct models, into
             results/baselines/ (only those not evaluated yet)
  train      chili/train.sh --eval: training on GPU_GROUPS, then eval.py on GPUS for each
             saved checkpoint (kept; DELETE_CKPTS=1 deletes each once evaluated)
  diversity  stage 1: calculate_diversity.py for this seed on one free GPU
  post       chili/post.sh: plot_bars_overall.py for this seed and, in stage 1,
             correlate_diversity.py over every seed scored so far
After select, a failed stage is reported and the next one still runs. Folders:
results/correlations/<setting>/ (stage 1) or results/selection/<dataset>.../ (stage 2) for the
scores, the same tree under RESPONSES_ROOT (/scratch/zf28/results) for the responses and under
figs/ for the figures, CKPT_ROOT/seed<seed>/<setting>/ for checkpoints, log/chili/ for logs.
Rerunning resumes: training skips jobs whose checkpoint or scores exist, and selection keeps the
subsets already saved.
EOF
}

seed=43 models=() abbrs=() skip=() dry_run=0 response_label_filter=safe num_train="" selected=""
mixed=0 with_unnatural=0
while (( $# )); do
    case $1 in
        --seed|--response_label_filter|--num_train|--selected)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"; shift 2 ;;
        --models|--abbrs|--skip)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        --dry_run) dry_run=1; shift ;;
        --mixed_diversity) mixed=1; shift ;;
        --with_unnatural) with_unnatural=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
setting_setup || exit 2
(( ${#models[@]} )) || models=(llama3.1_git20k mistral_git20k olmo2_git20k qwen2.5_git20k)
[[ -n $selected ]] && skip+=(diversity) || skip+=(select)
echo "===== $(date '+%F %T') seed $seed, $setting setting ($(versions torch vllm transformers deepspeed spacy))"

failed=()
# stage NAME COMMAND...: runs a stage unless skipped (with --dry_run, prints the command).
stage() {
    local name=$1 status
    shift
    if [[ " ${skip[*]} " == *" $name "* ]]; then
        echo "===== SKIPPED: $name"
        return
    fi
    if (( dry_run )) && [[ $name != train ]]; then
        echo "===== $name: would run: $*"
        return
    fi
    echo "===== $(date '+%F %T') START: $name"
    "$@"
    status=$?
    if (( status == 0 )); then
        echo "===== $(date '+%F %T') DONE: $name"
    else
        echo "===== $(date '+%F %T') FAILED: $name (exit $status)" >&2
        failed+=("$name")
    fi
}

select_opts=(--dataset "$selected" --seed "$seed" --size "$harmful" --response_label_filter
             "$response_label_filter" --threads "${SELECT_THREADS:-32}" --num_proc 16
             --cache_dir "$SELECTION_CACHE")
(( mixed )) && select_opts+=(--mixed_diversity)
(( with_unnatural )) && select_opts+=(--with_unnatural)
stage select "$CHILI/gpu.sh" python select_data.py "${select_opts[@]}"
if [[ " ${failed[*]} " == *" select "* ]]; then
    echo "Selection failed; fix it and rerun (the subsets already saved are kept)." >&2
    exit 1
fi

missing=()
for model in "${models[@]}" "${models[@]/%_git20k/-ins}"; do
    evaluated "$ROOT/results/baselines/$model.csv" || missing+=("$model")
done
if (( ${#missing[@]} )); then
    stage base "$CHILI/eval.sh" "${missing[@]}"
else
    echo "===== Base models already evaluated"
fi

train_opts=(--seed "$seed" --num_train "$num_train" --response_label_filter
            "$response_label_filter" --models "${models[@]}" --eval)
[[ -n $selected ]] && train_opts+=(--selected "$selected")
(( mixed )) && train_opts+=(--mixed_diversity)
(( with_unnatural )) && train_opts+=(--with_unnatural)
(( ${#abbrs[@]} )) && train_opts+=(--abbrs "${abbrs[@]}")
(( dry_run )) && train_opts+=(--dry_run)
stage train "$CHILI/train.sh" "${train_opts[@]}"

stage diversity "$CHILI/gpu.sh" python calculate_diversity.py --seeds "$seed" \
    --num_train "$harmful" --response_label_filter "$response_label_filter" --results_dir "$results"

post_opts=(--seed "$seed" --num_train "$num_train" --response_label_filter "$response_label_filter")
[[ -n $selected ]] && post_opts+=(--selected "$selected")
(( mixed )) && post_opts+=(--mixed_diversity)
(( with_unnatural )) && post_opts+=(--with_unnatural)
stage post bash "$CHILI/post.sh" "${post_opts[@]}"

if (( ${#failed[@]} )); then
    echo "Run finished with failed stage(s): ${failed[*]}" >&2
    exit 1
fi
echo "===== $(date '+%F %T') Run finished."

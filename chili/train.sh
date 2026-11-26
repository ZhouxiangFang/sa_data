#!/usr/bin/env bash
# Train one seed's alignment jobs (align.py, DeepSpeed) on chili's GPU groups, and with --eval
# evaluate the checkpoints once every training job has ended.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/train.sh [options] [-- align.py options]

  --seed SEED            Run seed (default: 43)
  --models MODEL...      Starting models (default: the four git20k models)
  --abbrs ABBR...        Only these groups (default: every WildGuardMix subcategory with enough
                         examples, plus the random and evenly baselines)
  --num_train N          Examples per job (default: 1800, a third harmful)
  --response_label_filter safe|refusal
                         Responses of the harmful examples (default: safe)
  --selected DATASET     Stage 2: train on every subset select_data.py selected from DATASET
                         for the seed (--abbrs then names selection methods)
  --mixed_diversity      With --selected: the subsets of the mixed-diversity ablation
  --with_unnatural       With --selected: the subsets of the ablation keeping the unnatural
                         jailbreak types
  --eval                 Then evaluate every saved checkpoint (chili/eval.sh; DELETE_CKPTS=1
                         deletes each once evaluated; default: kept)
  --eval_only            Only evaluate the saved checkpoints (implies --eval)
  --dry_run              Print the plan without running anything

Checkpoints go to CKPT_ROOT/seed<seed>/<setting>/ and scores to results/correlations/<setting>/
(stage 1) or results/selection/<dataset>.../ (stage 2), responses to the same tree under
RESPONSES_ROOT (setting_setup in chili/common.sh). Jobs whose checkpoint is saved or evaluated
are skipped, so rerunning resumes; a checkpoint or scores from other training data or settings
(a stale fingerprint, train_jobs.py) stop the run unless OVERWRITE_STALE=1 retrains them.
GPU_GROUPS (default "0,1,2,3 4,5,6,7") lists the training groups, a group takes the next job
when it is free; EFFECTIVE_BATCH (32), MICRO_BATCH (1), LEARNING_RATE (1e-5), DS_CONFIG
(ZeRO-3); see chili/common.sh. Logs: log/chili/align_<checkpoint>_<time>.log (with the peak
memory per GPU); the plan is log/chili/jobs_<time>.tsv.
EOF
}

seed=43 models=() abbrs=() align_args=() response_label_filter=safe num_train="" selected=""
mixed=0 with_unnatural=0 eval=0 eval_only=0 dry_run=0
while (( $# )); do
    case $1 in
        --models|--abbrs)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        --seed|--num_train|--response_label_filter|--selected)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"
            shift 2
            ;;
        --eval) eval=1; shift ;;
        --mixed_diversity) mixed=1; shift ;;
        --with_unnatural) with_unnatural=1; shift ;;
        --eval_only) eval=1 eval_only=1; shift ;;
        --dry_run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; align_args=("$@"); break ;;
        *) echo "Unknown option: $1 (align.py options go after --)" >&2; exit 2 ;;
    esac
done
setting_setup || exit 2
settings=(--seed "$seed" --num_train "$num_train" --response_label_filter "$response_label_filter")
[[ -n $selected ]] && settings+=(--selected "$selected")
(( mixed )) && settings+=(--mixed_diversity)
(( with_unnatural )) && settings+=(--with_unnatural)

read -ra groups <<< "${GPU_GROUPS:-0,1,2,3 4,5,6,7}"
for group in "${groups[@]}"; do
    IFS=, read -ra group_gpus <<< "$group"
    if (( EFFECTIVE_BATCH % (${#group_gpus[@]} * MICRO_BATCH) )); then
        echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH is not a multiple of GPUs x MICRO_BATCH" \
             "($((${#group_gpus[@]} * MICRO_BATCH))) for GPU group $group" >&2
        exit 2
    fi
done
[[ -f $DS_CONFIG ]] || { echo "DeepSpeed config not found: $DS_CONFIG" >&2; exit 2; }
# ZeRO-2 with gradient accumulation computes wrong gradients on DeepSpeed 0.17.6 to 0.19.5
# (issues #7718, #8224); see range/train.sh.
bad_zero2=$(python - "$DS_CONFIG" <<'EOF'
import json, sys
from importlib.metadata import version
from packaging.version import Version
stage = json.load(open(sys.argv[1])).get("zero_optimization", {}).get("stage")
deepspeed = Version(version("deepspeed"))
if stage == 2 and Version("0.17.6") <= deepspeed < Version("0.19.6"):
    print(deepspeed)
EOF
) || exit 2
if [[ -n $bad_zero2 ]]; then
    echo "DeepSpeed $bad_zero2 computes wrong ZeRO-2 gradients with gradient accumulation" \
         "(0.17.6 to 0.19.5); use DeepSpeed >= 0.19.6 (requirements.txt)." >&2
    exit 2
fi
# Put first, so settings given after -- still win.
align_args=(--deepspeed "$(realpath "$DS_CONFIG")" --per_device_train_batch_size "$MICRO_BATCH"
            --effective_batch_size "$EFFECTIVE_BATCH" --learning_rate "$LEARNING_RATE"
            "${settings[@]}" "${align_args[@]}")

plan=(--ckpt_dir "$ckpt_dir" --results_dir "$results" "${settings[@]}")
(( ${#models[@]} )) && plan+=(--models "${models[@]}")
(( ${#abbrs[@]} )) && plan+=(--abbrs "${abbrs[@]}")
jobs_tsv=$(python "$SA_DIR/train_jobs.py" "${plan[@]}" -- "${align_args[@]}" \
    2> >(grep -v NVML >&2)) || exit $?

declare -A count=([train]=0 [saved]=0 [done]=0 [stale]=0)
job_model=() job_abbr=() job_ckpt=() stale_lines=()
while IFS= read -r line; do
    [[ -n $line ]] || continue
    IFS=$'\t' read -r model abbr checkpoint status <<< "$line"
    (( ++count[$status] ))
    [[ $status == stale ]] && stale_lines+=("$line")
    [[ $status == train || ($status == stale && $OVERWRITE_STALE == 1) ]] || continue
    job_model+=("$model") job_abbr+=("$abbr") job_ckpt+=("$checkpoint")
done <<< "$jobs_tsv"
echo "Seed $seed, $setting setting: ${count[train]} to train, ${count[saved]} saved," \
     "${count[done]} evaluated, ${count[stale]} stale. Checkpoints: $ckpt_dir"
echo "Effective batch $EFFECTIVE_BATCH (micro-batch $MICRO_BATCH), learning rate" \
     "$LEARNING_RATE, $(basename "$DS_CONFIG"), GPU groups ${groups[*]}"
if (( ${#stale_lines[@]} )); then
    printf '  stale: %s\n' "${stale_lines[@]}" >&2
    if (( ! OVERWRITE_STALE )); then
        echo "These runs have a checkpoint or scores from other training data or settings" \
             "(another fingerprint, train_jobs.py). OVERWRITE_STALE=1 retrains them, replacing" \
             "their checkpoints and scores; or use another setting/CKPT_ROOT." >&2
        exit 2
    fi
fi
(( eval_only )) && job_model=()
if (( dry_run )); then
    for i in "${!job_model[@]}"; do echo "would train: $(basename "${job_ckpt[i]}")"; done
    (( eval )) && echo "would then evaluate the saved checkpoints into $results" \
                       "(responses: $RESPONSES_ROOT)"
    exit 0
fi

status=0
if (( ${#job_model[@]} )); then
    mkdir -p -- "$ckpt_dir"
    printf '%s\n' "$jobs_tsv" > "$LOG_DIR/jobs_$STAMP.tsv"

    job_label() { basename "${job_ckpt[$1]}"; }

    run_job() {
        local i=$1 log mem_log sampler status
        log="$LOG_DIR/align_$(basename "${job_ckpt[$1]}")_$STAMP.log"
        echo "  log: $log"
        {
            echo "$(date '+%F %T') $(basename "${job_ckpt[i]}") on $(hostname), GPUs $2"
            versions deepspeed torch transformers trl
            nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$2"
        } >> "$log"
        mem_log=$(mktemp)
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
            -i "$2" -l 15 > "$mem_log" 2>/dev/null &
        sampler=$!
        # A failed attempt may have left part of the checkpoint; the planner said it is incomplete.
        env -u CUDA_VISIBLE_DEVICES deepspeed --include="localhost:$2" --master_port="$3" \
            "$SA_DIR/align.py" --model "${job_model[i]}" --abbr "${job_abbr[i]}" \
            --output_dir "$ckpt_dir" --overwrite_output_dir "${align_args[@]}" >> "$log" 2>&1
        status=$?
        kill "$sampler" 2>/dev/null
        awk -F', ' '$2 > peak[$1] { peak[$1] = $2; total[$1] = $3 }
                    END { for (g in peak) printf "Peak memory on GPU %s: %d of %d MiB\n", g, peak[g], total[g] }' \
            "$mem_log" >> "$log"
        rm -f -- "$mem_log"
        echo "$(date '+%F %T') exit $status" >> "$log"
        return "$status"
    }

    run_queue "${#job_model[@]}" "${groups[@]}" || status=1
fi

if (( eval )); then
    # Re-plan: evaluate what is saved now (including checkpoints trained before this launch).
    mapfile -t saved < <(python "$SA_DIR/train_jobs.py" "${plan[@]}" -- "${align_args[@]}" \
        2>/dev/null | awk -F'\t' '$4 == "saved" { print $3 }')
    if (( ${#saved[@]} )); then
        delete=()
        (( DELETE_CKPTS )) && delete=(--delete)
        "$CHILI_DIR/eval.sh" "${saved[@]}" "${delete[@]}" -- --output_dir "$results" || status=1
    else
        echo "No saved checkpoints to evaluate."
    fi
fi
exit "$status"

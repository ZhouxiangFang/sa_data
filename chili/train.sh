#!/usr/bin/env bash
# Train one run's alignment jobs (align.py, DeepSpeed) on chili's 4-GPU groups.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/train.sh [options] [-- align.py options]

  --models MODEL...      Starting models (default: llama3.1_git20k mistral_git20k
                         olmo2_git20k qwen2.5_git20k, i.e. FocusV857/<model> on the HF Hub)
  --datasets DATASET...  Default: aegis wildguardmix gretel (gretel with --group_by tactic)
  --abbrs ABBR...        Only these groups (default: every non-benign subcategory with
                         enough examples, plus the random and evenly baselines)
  --group_by tactic      Group Gretel's harmful prompts by jailbreak tactic
  --group_by selection   Train on the subsets select_data.py saved for the seed (--abbrs
                         then names subsets, e.g. semantic_vendi_bge-m3; default: all)
  --num_train N          Examples per job (default: 1800)
  --harmful_rate R       Harmful fraction (default: 1/3)
  --general_rate R       General-instruction fraction (default: 1/3); benign gets the rest
  --seed SEED            Run seed (default: 43)
  --response_filter F    Harmful-example responses: safe (default) or refusal (wildguardmix
                         only); align.py's pool option, as is --dedup
  --dedup                Keep one row per prompt (default: every row)
  --output_dir DIR       Checkpoint folder (default: $CKPT_ROOT/seed<seed>, plus
                         /<tag> with pool options: refusal, dedup or refusal_dedup)
  --results_dir DIR      Also skip jobs already evaluated into DIR (eval.py --output_dir)

Jobs whose checkpoint is saved (or evaluated) are skipped, so rerunning resumes.
Each job's log is log/align_<checkpoint>_<time>.log. GPU_GROUPS (default
"0,1,2,3 4,5,6,7") lists the GPU groups; a group takes the next job when it is free.
EFFECTIVE_BATCH (default 32, examples per optimizer update over a group's GPUs) and
LEARNING_RATE (default 1e-5) set align.py's training; align.py options after -- (e.g.
--learning_rate) override them. The checkpoint name records neither them nor the pool
options, so give other values their own --output_dir and --results_dir (the default
--output_dir already adds the pool options' tag). Every group's GPU count x micro-batch
(default 1) must divide the effective batch. Keep the micro-batch at 1 on the 48 GB A6000s:
2 and 4 ran out of memory on 4096-token examples (llama3.1, qwen2.5).
EOF
}

models=() datasets=() abbrs=() align_args=() response_filter=safe dedup=0
group_by=subcategory num_train=1800 harmful_rate=0.3333333333333333
general_rate=0.3333333333333333 seed=43 output_dir="" results_dir=""
while (( $# )); do
    case $1 in
        --models|--datasets|--abbrs)
            list=${1#--}
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values "$list" "$1"; shift; done
            ;;
        --dedup) dedup=1; shift ;;
        --response_filter|--group_by|--num_train|--harmful_rate|--general_rate|--seed|--output_dir|--results_dir)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; align_args=("$@"); break ;;
        *) echo "Unknown option: $1 (align.py options go after --)" >&2; exit 2 ;;
    esac
done
# Pool options (align.py) reuse the checkpoint names, so their checkpoints get their own folder.
pool_setup || exit 2
output_dir=${output_dir:-$CKPT_ROOT/seed$seed${tag:+/$tag}}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
micro_batch=1  # align.py's default
# align.py keeps the last value given for an option, so options after -- win over these.
# Read them back, so the summary and the batch check below show what training will use.
for ((k = 0; k < ${#align_args[@]}; k++)); do
    arg=${align_args[k]} next=${align_args[k + 1]:-}
    case $arg in
        --learning_rate|--lr) LEARNING_RATE=$next ;;
        --learning_rate=*|--lr=*) LEARNING_RATE=${arg#*=} ;;
        --effective_batch_size) EFFECTIVE_BATCH=$next ;;
        --effective_batch_size=*) EFFECTIVE_BATCH=${arg#*=} ;;
        --per_device_train_batch_size) micro_batch=$next ;;
        --per_device_train_batch_size=*) micro_batch=${arg#*=} ;;
    esac
done
read -ra groups <<< "${GPU_GROUPS:-0,1,2,3 4,5,6,7}"
# align.py refuses an effective batch that its GPUs x micro-batch does not divide; check every
# group now instead of letting each job fail at start-up.
if ! [[ $EFFECTIVE_BATCH =~ ^[1-9][0-9]*$ && $micro_batch =~ ^[1-9][0-9]*$ ]]; then
    echo "Effective batch ($EFFECTIVE_BATCH) and micro-batch ($micro_batch) must be positive" \
         "integers" >&2
    exit 2
fi
for group in "${groups[@]}"; do
    IFS=, read -ra group_gpus <<< "$group"
    if (( EFFECTIVE_BATCH % (${#group_gpus[@]} * micro_batch) )); then
        echo "Effective batch $EFFECTIVE_BATCH is not a multiple of GPUs x micro-batch" \
             "($((${#group_gpus[@]} * micro_batch))) for GPU group $group" >&2
        exit 2
    fi
done
(( micro_batch == 1 )) ||
    echo "Warning: micro-batch $micro_batch; on the 48 GB A6000s, micro-batches of 2 and 4 ran" \
         "out of memory on 4096-token examples (llama3.1, qwen2.5)." >&2
settings=(--group_by "$group_by" --num_train "$num_train" --harmful_rate "$harmful_rate"
          --general_rate "$general_rate" --seed "$seed" "${pool_opts[@]}")

plan=(--ckpt_dir "$output_dir" "${settings[@]}")
(( ${#models[@]} )) && plan+=(--models "${models[@]}")
(( ${#datasets[@]} )) && plan+=(--datasets "${datasets[@]}")
(( ${#abbrs[@]} )) && plan+=(--abbrs "${abbrs[@]}")
[[ -n $results_dir ]] && plan+=(--results_dir "$results_dir")
jobs_tsv=$(python "$SA_DIR/train_jobs.py" "${plan[@]}") || exit $?

job_model=() job_dataset=() job_abbr=() job_ckpt=()
declare -A count=([train]=0 [saved]=0 [done]=0)
while IFS=$'\t' read -r model dataset abbr checkpoint status; do
    [[ -n $status ]] || continue  # no jobs: an empty plan
    (( ++count[$status] ))
    [[ $status == train ]] || continue
    job_model+=("$model") job_dataset+=("$dataset") job_abbr+=("$abbr") job_ckpt+=("$checkpoint")
done <<< "$jobs_tsv"
pool_note="$response_filter responses"
(( dedup )) && pool_note+=", one row per prompt"
echo "Seed $seed, $group_by groups, $num_train examples, $pool_note: ${count[train]} to train," \
     "${count[saved]} saved, ${count[done]} evaluated. Checkpoints: $output_dir"
echo "Effective batch $EFFECTIVE_BATCH (micro-batch $micro_batch), learning rate $LEARNING_RATE"
(( ${#job_model[@]} )) || exit 0

job_label() { printf '%s' "$(basename "${job_ckpt[$1]}")"; }

run_job() {
    local i=$1 log="$LOG_DIR/align_$(basename "${job_ckpt[$1]}")_$STAMP.log"
    echo "  log: $log"
    # A failed attempt may have left part of the checkpoint; the planner said it is incomplete.
    env -u CUDA_VISIBLE_DEVICES deepspeed --include="localhost:$2" --master_port="$3" \
        "$SA_DIR/align.py" --model "${job_model[i]}" --alignment_dataset "${job_dataset[i]}" \
        --abbr "${job_abbr[i]}" --output_dir "$output_dir" --overwrite_output_dir \
        "${settings[@]}" --effective_batch_size "$EFFECTIVE_BATCH" \
        --learning_rate "$LEARNING_RATE" "${align_args[@]}" >> "$log" 2>&1
}

run_queue "${#job_model[@]}" "${groups[@]}"

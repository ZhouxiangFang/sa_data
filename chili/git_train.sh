#!/usr/bin/env bash
# SFT pretrained models on general instructions (git_train.py) on chili's 4-GPU groups.
# This made the git20k starting models (now FocusV857/<model>_git20k on the HF Hub).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/git_train.sh --models MODEL... [--dataset CSV] [--output_dir DIR] [-- git_train.py options]

  --models MODEL...  llama3 llama3.1 qwen2.5 qwen3 olmo2 mistral
  --dataset CSV      Default: data/git_data_20k.csv
  --output_dir DIR   Default: /home/zf28/models (saves <DIR>/<model>_git<size>)
Each model's log is log/git_train_<model>_<time>.log; GPU_GROUPS as in train.sh.
EOF
}

models=() train_args=()
dataset=$ROOT/data/git_data_20k.csv output_dir=/home/zf28/models
while (( $# )); do
    case $1 in
        --models) shift; while (( $# )) && [[ $1 != --* ]]; do add_values models "$1"; shift; done ;;
        --dataset|--output_dir)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            printf -v "${1#--}" '%s' "$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; train_args=("$@"); break ;;
        *) echo "Unknown option: $1 (git_train.py options go after --)" >&2; exit 2 ;;
    esac
done
(( ${#models[@]} )) || { usage >&2; exit 2; }

job_label() { printf '%s' "${models[$1]}"; }

run_job() {
    local log="$LOG_DIR/git_train_${models[$1]}_$STAMP.log"
    echo "  log: $log"
    env -u CUDA_VISIBLE_DEVICES deepspeed --include="localhost:$2" --master_port="$3" \
        "$SA_DIR/git_train.py" --model "${models[$1]}" --dataset "$dataset" \
        --output_dir "$output_dir" --deepspeed "$SA_DIR/ds_config.json" "${train_args[@]}" \
        >> "$log" 2>&1
}

read -ra groups <<< "${GPU_GROUPS:-0,1,2,3 4,5,6,7}"
run_queue "${#models[@]}" "${groups[@]}"

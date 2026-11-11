#!/usr/bin/env bash
# Greedy diverse-subset selection (select_data.py) for one seed, one free GPU per dataset.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/select.sh [--seed SEED] [--datasets DATASET...] [-- select_data.py options]

Selects 600-example subsets of aegis, wildguardmix and gretel (or --datasets): one per
diversity metric (exact greedy selection) and one per baseline strategy (novelselect,
kcenter_greedy, repr_filter, qdit, kmeans, farthest; on prompt and POS embeddings), with
both embedding models, all starting from the same 50 random examples drawn with the seed
(default 43). Results: data/seed<seed>/<dataset>/<metric>[_<embedder>]_600.csv.

select_data.py options go after --, e.g. --metrics semantic_vendi lexical_hdd,
--baselines qdit, --embedding_models bge-m3, --size 300, --overwrite. Existing CSVs are kept, so rerunning
resumes. Each dataset's log is log/select_seed<seed>_<dataset>_<time>.log. GPUS (default:
every GPU) lists the GPUs to use; THREADS (default 32) is the CPU threads per dataset.
EOF
}

seed=43 datasets=() select_args=()
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --datasets)
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values datasets "$1"; shift; done
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; select_args=("$@"); break ;;
        *) echo "Unknown option: $1 (select_data.py options go after --)" >&2; exit 2 ;;
    esac
done
(( ${#datasets[@]} )) || datasets=(aegis wildguardmix gretel)
# The Vendi metrics run on CPU threads; OMP_NUM_THREADS (4, from common.sh) would cap them.
threads=${THREADS:-32}
export OMP_NUM_THREADS=$threads NUMBA_NUM_THREADS=$threads

job_label() { printf 'select %s (seed %s)' "${datasets[$1]}" "$seed"; }

run_job() {
    local dataset=${datasets[$1]} log
    log="$LOG_DIR/select_seed${seed}_${dataset}_$STAMP.log"
    echo "  log: $log"
    CUDA_VISIBLE_DEVICES=$2 python "$SA_DIR/select_data.py" --datasets "$dataset" \
        --seed "$seed" --threads "$threads" "${select_args[@]}" >> "$log" 2>&1
}

IFS=$', \n' read -d '' -ra gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)}"
run_queue "${#datasets[@]}" "${gpus[@]}"

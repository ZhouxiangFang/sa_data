#!/usr/bin/env bash
# Greedy diverse-subset selection (select_data.py) for one seed, one free GPU per dataset.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: chili/select_data.sh [--seed SEED] [--datasets DATASET...] [-- select_data.py options]

Selects 600-example subsets of aegis, wildguardmix and gretel (or --datasets) from every
harmful example with a safe response (every row: a prompt with several safe responses
is several candidates): one per diversity metric the figures show (12, exact
greedy selection), one per NovelSum baseline (novelselect, kcenter_greedy, repr_filter,
qdit, kmeans, farthest; on prompt embeddings) and the random and evenly baselines, with
qwen3-embedding-0.6b: 20 subsets per dataset. All start from the same 50 random examples
drawn with the seed (default 43); on wildguardmix the start and every subset keep the
pool's share of adversarial prompts (about 43%).
Results: data/seed<seed>/<dataset>/<metric>[_<embedder>]_600.csv.

select_data.py options go after --, e.g. --metrics semantic_vendi lexical_hdd,
--baselines qdit, --embedding_models bge-m3, --stratify none, --size 300, --overwrite,
and the pool options --response_filter refusal (wildguardmix only, then the default
dataset) and --dedup (one row per prompt), whose subsets go to
data/seed<seed>/<dataset>_<tag>/ (e.g. wildguardmix_refusal).
Existing CSVs are kept, so rerunning resumes; CSVs from before stratified selection stop
the run until they are moved away. Each dataset's log is
log/select_seed<seed>_<dataset>_<time>.log. GPUS (default: every GPU) lists the GPUs to
use; THREADS (default 32) is the CPU threads per dataset.
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
if (( ! ${#datasets[@]} )); then
    datasets=(aegis wildguardmix gretel)
    # Only wildguardmix labels refusals.
    [[ " ${select_args[*]} " =~ \ --response_filter[=\ ]refusal\  ]] && datasets=(wildguardmix)
fi
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

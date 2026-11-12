#!/usr/bin/env bash
# Submit greedy diverse-subset selection (select_data.py) for one seed as Slurm jobs on range,
# one job per dataset (1 x EVAL_GPU for the embeddings, THREADS CPUs for the Vendi metrics).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat <<'EOF'
Usage: range/select_data.sh [--seed SEED] [--datasets DATASET...] [--dry_run]
                            [-- select_data.py options]

chili/select_data.sh as Slurm jobs: selects 600-example subsets of aegis, wildguardmix and
gretel (or --datasets) from every harmful example with a safe response, one per diversity
metric the figures show and per baseline (20 per dataset), all from the same 50 random
examples of the seed (default 43). Results: data/seed<seed>/<dataset>/<name>_600.csv.

select_data.py options go after --, e.g. --metrics semantic_vendi, --baselines qdit,
--overwrite, and the pool options --response_filter refusal (wildguardmix only, then the
default dataset) and --dedup, whose subsets go to data/seed<seed>/<dataset>_<tag>/.
Existing CSVs are kept, so resubmitting resumes. Each job gets THREADS CPUs (default 32),
EVAL_MEM and SELECT_TIME (default 04:00:00); logs: log/slurm/select_s<seed>_<dataset>_<job>.log.
DEPENDENCY (e.g. afterok:<job>) holds the jobs until another job has finished.
EOF
}

seed=43 datasets=() select_args=() dry_run=0
while (( $# )); do
    case $1 in
        --seed) (( $# >= 2 )) || { echo "--seed needs a value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --datasets)
            shift
            while (( $# )) && [[ $1 != --* ]]; do add_values datasets "$1"; shift; done
            ;;
        --dry_run) dry_run=1; shift ;;
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

for dataset in "${datasets[@]}"; do
    command=(python select_data.py --datasets "$dataset" --seed "$seed" --threads "$threads"
             "${select_args[@]}")
    if (( dry_run )); then
        echo "would submit (1 GPU, $threads CPUs${DEPENDENCY:+, after $DEPENDENCY}): ${command[*]}"
        continue
    fi
    OMP_NUM_THREADS=$threads NUMBA_NUM_THREADS=$threads EVAL_CPUS=$threads \
        EVAL_TIME=${SELECT_TIME:-$EVAL_TIME} NAME=select_s${seed}_$dataset \
        "$RANGE_DIR/gpu.sh" "${command[@]}" || exit
done

# Shared by the chili launchers (8 x A6000, no scheduler); source it, don't run it.
# Sets up the nlp environment and runs job queues on GPUs that are free.
#
# A GPU is free when its lock /tmp/align_eval_gpu_<i>.lock is not held and it uses less
# than MAX_USED_MB (default 2000). Every job holds the locks of its GPUs, so holding a lock
# keeps the launchers off that GPU, e.g. to leave it to other users:
#   tmux new -d -s gpu-hold 'exec 9>/tmp/align_eval_gpu_4.lock; flock 9; sleep infinity'

SA_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(dirname -- "$SA_DIR")
LOG_DIR=${LOG_DIR:-$ROOT/log}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-10}
JOB_ATTEMPTS=${JOB_ATTEMPTS:-2}
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p -- "$LOG_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != nlp ]]; then
    set +u  # the activation scripts reference unset variables
    source /home/zf28/miniconda3/etc/profile.d/conda.sh && conda activate nlp || exit 1
    set -u
fi
# DeepSpeed compiles its CUDA ops against the pip-installed NVIDIA headers.
for include_dir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/include; do
    [[ -d $include_dir && ":${CPATH:-}:" != *":$include_dir:"* ]] &&
        export CPATH="$include_dir${CPATH:+:$CPATH}"
done
# Driver 595.84 needs matching user-space libraries; a no-op with other drivers.
if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    export LD_LIBRARY_PATH="/home/zf28/align/.nvidia-595.84/compute-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

lock_file() { printf '/tmp/align_eval_gpu_%s.lock' "$1"; }

# Succeeds when every GPU of a comma-separated list is free.
gpus_free() {
    local gpu used
    local -a gpus
    local -A used_mb=()
    while IFS=', ' read -r gpu used; do
        used_mb[$gpu]=$used
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    IFS=, read -ra gpus <<< "$1"
    for gpu in "${gpus[@]}"; do
        (( ${used_mb[$gpu]:-MAX_USED_MB} < MAX_USED_MB )) || return 1
        flock -n "$(lock_file "$gpu")" true 2>/dev/null || return 1
    done
}

# Holds the locks of a comma-separated GPU list until this (sub)shell exits; 75 if one is taken.
lock_gpus() {
    local gpu fd
    local -a gpus
    IFS=, read -ra gpus <<< "$1"
    for gpu in "${gpus[@]}"; do
        exec {fd}>"$(lock_file "$gpu")"
        flock -n "$fd" || return 75
    done
}

kill_tree() {
    local child
    for child in $(pgrep -P "$1"); do kill_tree "$child"; done
    kill "$1" 2>/dev/null
}

# run_queue N SLOT...: runs jobs 0..N-1, each on the first free slot (a comma-separated GPU
# list such as 0,1,2,3 or 5). The caller defines run_job INDEX GPUS PORT, which runs one job
# in the foreground (PORT is unique per slot), and job_label INDEX. A failed job is retried
# until it has run JOB_ATTEMPTS times. Fails if any job failed.
run_queue() {
    local njobs=$1 slot job status message last_message=""
    shift
    local -a slots=("$@") pending=() failed=() attempts=()
    local -A slot_job=()
    QUEUE_PIDS=()
    trap 'for pid in "${QUEUE_PIDS[@]}"; do kill_tree "$pid"; done; exit 130' INT TERM
    for ((job = 0; job < njobs; job++)); do pending+=("$job"); done
    while (( ${#pending[@]} + ${#QUEUE_PIDS[@]} )); do
        for slot in "${!slots[@]}"; do
            if [[ -n ${QUEUE_PIDS[$slot]:-} ]]; then
                ! kill -0 "${QUEUE_PIDS[$slot]}" 2>/dev/null || continue
                status=0
                wait "${QUEUE_PIDS[$slot]}" || status=$?
                job=${slot_job[$slot]}
                unset "QUEUE_PIDS[$slot]" "slot_job[$slot]"
                if (( status == 0 )); then
                    echo "$(date '+%F %T') Finished $(job_label "$job")"
                elif (( status == 75 )); then
                    pending=("$job" "${pending[@]}")  # another process took a GPU first
                elif (( ++attempts[job] < JOB_ATTEMPTS )); then
                    echo "$(date '+%F %T') Failed $(job_label "$job") (exit $status); requeued" >&2
                    pending+=("$job")
                else
                    echo "$(date '+%F %T') Failed $(job_label "$job") (exit $status)" >&2
                    failed+=("$job")
                fi
            fi
            (( ${#pending[@]} )) && gpus_free "${slots[slot]}" || continue
            job=${pending[0]}
            pending=("${pending[@]:1}")
            echo "$(date '+%F %T') Starting $(job_label "$job") on GPU(s) ${slots[slot]}"
            ( lock_gpus "${slots[slot]}" || exit 75
              run_job "$job" "${slots[slot]}" "$((29500 + slot))" ) &
            QUEUE_PIDS[$slot]=$!
            slot_job[$slot]=$job
        done
        message=""
        (( ${#pending[@]} && ${#QUEUE_PIDS[@]} < ${#slots[@]} )) &&
            message="${#pending[@]} job(s) waiting for free GPUs"
        [[ -z $message || $message == "$last_message" ]] || echo "$(date '+%F %T') $message"
        last_message=$message
        (( ${#pending[@]} + ${#QUEUE_PIDS[@]} )) && sleep "$POLL_SECONDS"
    done
    trap - INT TERM
    (( ${#failed[@]} == 0 )) && return
    echo "${#failed[@]} job(s) failed:" >&2
    for job in "${failed[@]}"; do echo "  $(job_label "$job")" >&2; done
    return 1
}

# Splits comma- or space-separated values into the named array: add_values ARRAY VALUE...
add_values() {
    local -n target=$1
    local value
    local -a parts
    shift
    for value; do
        IFS=, read -ra parts <<< "$value"
        target+=("${parts[@]}")
    done
}

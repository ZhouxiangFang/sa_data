# Shared by the chili launchers (chili01: 8 x RTX A6000 48 GB, no scheduler); source it.
# Activates the nlp conda env (the one env of every stage, ../ENVIRONMENT.md: torch 2.11,
# transformers 4.57.3, vLLM 0.23 with batch-invariant kernels, spaCy en_core_web_trf on cupy),
# runs job queues on free GPUs, and holds the defaults, which the environment overrides:
#
#   GPUS=all                       GPUs of one-GPU jobs (eval, diversity, selection), e.g. 0,1,2,3
#   GPU_GROUPS="0,1,2,3 4,5,6,7"   GPU groups of training jobs (full fine-tuning, one job per group)
#   MAX_USED_MB=2000               A GPU using more memory than this is busy (another user's job)
#   EFFECTIVE_BATCH=32             Examples per optimizer update
#   MICRO_BATCH=1                  Per-GPU micro-batch; accumulation is derived (8 on 4 GPUs).
#                                  2 and 4 ran out of memory on 4096-token examples on the A6000s.
#   LEARNING_RATE=1e-5             Peak learning rate (as range/common.sh)
#   DS_CONFIG=chili/ds_zero3.json  ZeRO-3: an 8B model's AdamW states do not fit in 48 GB with
#                                  ZeRO-2 on 4 GPUs (range runs ZeRO-2 on 2 x H200)
#   DELETE_CKPTS=0                 1: delete each checkpoint once it is evaluated
#   OVERWRITE_STALE=0              1: retrain runs whose checkpoint or scores come from other
#                                  training data or settings (default: refuse)
#   CKPT_ROOT=/scratch/zf28/ckpts  Checkpoints: CKPT_ROOT/seed<seed>/<setting>/
#   RESPONSES_ROOT=/scratch/zf28/results
#                                  eval.py --responses_root: the responses, in results/'s tree
#   SELECTION_CACHE=/scratch/zf28/selection_cache
#                                  select_data.py --cache_dir: POS tags and embeddings per dataset
#   LOG_DIR=log/chili              Job logs (range's are in log/slurm)
#
# /scratch on chili is chili's own local disk (not range's). Outputs, as on range: scores (small
# CSVs) in results/: baselines/ (instruct and git20k models), correlations/<setting>/ (stage 1)
# and selection/<setting>/ (stage 2); responses in the same tree under RESPONSES_ROOT; figures in
# the same tree under figs/; training data in samples/.
#
# A GPU is free when its lock /tmp/align_eval_gpu_<i>.lock is not held and it uses less than
# MAX_USED_MB. Every job holds the locks of its GPUs, so holding a lock keeps the launchers off
# that GPU, e.g. to leave it to other users:
#   tmux new -d -s gpu-hold 'exec 9>/tmp/align_eval_gpu_4.lock; flock 9; sleep infinity'

CHILI_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SA_DIR=$(dirname -- "$CHILI_DIR")
ROOT=$(dirname -- "$SA_DIR")
export CHILI_DIR SA_DIR ROOT
CKPT_ROOT=${CKPT_ROOT:-/scratch/zf28/ckpts}
RESPONSES_ROOT=${RESPONSES_ROOT:-/scratch/zf28/results}
SELECTION_CACHE=${SELECTION_CACHE:-/scratch/zf28/selection_cache}
LOG_DIR=${LOG_DIR:-$ROOT/log/chili}
mkdir -p -- "$LOG_DIR"

MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-10}
JOB_ATTEMPTS=${JOB_ATTEMPTS:-2}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32} MICRO_BATCH=${MICRO_BATCH:-1} LEARNING_RATE=${LEARNING_RATE:-1e-5}
DS_CONFIG=${DS_CONFIG:-$CHILI_DIR/ds_zero3.json}
DELETE_CKPTS=${DELETE_CKPTS:-0}
OVERWRITE_STALE=${OVERWRITE_STALE:-0}

if [[ "${CONDA_DEFAULT_ENV:-}" != nlp ]]; then
    set +u  # the activation scripts reference unset variables
    source /home/zf28/miniconda3/etc/profile.d/conda.sh && conda activate nlp || exit 1
    set -u
fi
# The Hugging Face cache (models, datasets, token) on chili is the default one in the home
# folder (NFS, no small quota); range keeps its own on /scratch.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
# Compile caches go to the local /scratch, not the NFS home: DeepSpeed warns that Triton's cache
# there (~/.triton) can hang when a job exits, and vLLM's torch.compile cache grows by about
# 50 MB per evaluated model. Nothing is built with nvcc (torch's fused AdamW, torch_adam in
# ds_zero3.json); the C helpers Triton builds use the system gcc.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/zf28/triton_cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/scratch/zf28/vllm_cache}
# Greedy decoding bypasses vLLM's FlashInfer sampler, so skip its JIT build (range/common.sh).
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/scratch/zf28}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Names one launch's logs and job lists; the PID keeps launches in the same second apart.
STAMP=$(date +%Y%m%d_%H%M%S)_$$

# add_values ARRAY VALUE...: appends comma- or space-separated values to the named array.
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

# setting_setup: the folders of a setting, exactly as range/common.sh's setting_setup (keep the
# two in sync). From seed, response_label_filter (safe or refusal), selected (stage 2: the
# selection dataset, or empty), with_unnatural (1: stage 2's ablation keeping the unnatural
# jailbreak types), mixed (1: stage 2's mixed-diversity ablation) and num_train (default 1800),
# sets the folders of a setting. Returns 2 on a bad option.
#   setting    default or refusal (stage 2: selected_<dataset>[_refusal][_with_unnatural][_mixed]),
#              plus _<harmful count> for another size (default_500, selected_wildjailbreak_500)
#   harmful    harmful examples per run (a third, rounded as helper.mixture_counts)
#   results    the setting's scores folder: results/correlations/<setting> (stage 1, with the
#              diversity scores in diversity/) or results/selection/<setting without selected_>
#   ckpt_dir   CKPT_ROOT/seed<seed>/<setting>
setting_setup() {
    case $response_label_filter in
        safe) setting="" ;;
        refusal) setting=refusal ;;
        *) echo "--response_label_filter is safe or refusal, not $response_label_filter" >&2; return 2 ;;
    esac
    if [[ -n ${selected:-} ]]; then
        setting=selected_$selected${setting:+_$setting}
        if (( ${with_unnatural:-0} )); then
            [[ $selected == alert_dpo || $selected == nemotron ]] ||
                { echo "--with_unnatural applies to alert_dpo and nemotron" >&2; return 2; }
            setting+=_with_unnatural
        fi
        (( ${mixed:-0} )) && setting+=_mixed
    elif (( ${mixed:-0} || ${with_unnatural:-0} )); then
        echo "--mixed_diversity and --with_unnatural need --selected" >&2; return 2
    else
        setting=${setting:-default}
    fi
    num_train=${num_train:-1800}
    harmful=$(( (2 * num_train + 3) / 6 ))  # int(num_train / 3 + 0.5)
    (( num_train == 1800 )) || setting+=_$harmful
    results=$ROOT/results/correlations/$setting
    [[ -n ${selected:-} ]] && results=$ROOT/results/selection/${setting#selected_}
    ckpt_dir=$CKPT_ROOT/seed$seed/$setting
}

# evaluated SCORES_CSV: eval.py's scores file of a model holds a complete evaluation (the
# harmful, overrefusal and IF averages; eval.evaluated in Python).
evaluated() {
    [[ -f $1 ]] && grep -q '^harmful,Avg,' "$1" && grep -q '^overrefusal,Avg,' "$1" &&
        grep -q '^if_prompt_loose,Avg,' "$1"
}

# versions PACKAGE...: one line with the versions of the env's packages, for the job logs.
versions() {
    python - "$@" <<'EOF' 2>/dev/null
import sys
from importlib.metadata import version
print(f"env {sys.prefix}: " + ", ".join(f"{name} {version(name)}" for name in sys.argv[1:]))
EOF
}

# all_gpus: the GPUS list (comma or space separated), default every GPU, one per line.
all_gpus() {
    local -a gpus
    IFS=$', \n' read -d '' -ra gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)}"
    printf '%s\n' "${gpus[@]}"
}

lock_file() { printf '/tmp/align_eval_gpu_%s.lock' "$1"; }

# gpus_free GPU_LIST: succeeds when every GPU of a comma-separated list is free.
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

# lock_gpus GPU_LIST: holds the locks of a comma-separated GPU list until this (sub)shell
# exits; 75 if one is taken.
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

# Shared by the nersc (NERSC Perlmutter, Slurm) launchers and batch scripts; source it.
# Activates the nlp conda env, the one env of every stage (sa_data/requirements.txt: torch 2.11,
# transformers 4.57.3, vLLM 0.23 with batch-invariant kernels), and holds the Slurm defaults, which
# the environment overrides:
#
#   ACCOUNT=m4705_g                Allocation of the GPU jobs (CPU jobs: CPU_ACCOUNT=m4705)
#   GPU_CONSTRAINT=gpu&hbm40g      GPU nodes: 4 x A100-SXM4-40GB (gpu&hbm80g: the 80 GB nodes)
#   TRAIN_NGPUS=4                  GPUs per training job (full fine-tuning); 4 is a whole node
#   TRAIN_QOS, EVAL_QOS            Default: regular for a whole node, shared for fewer GPUs;
#                                  debug for a quick test (30 minutes, 2 jobs at once)
#   MAX_TRAIN=8 MAX_EVAL=16        Jobs of one array that may run at once
#   TRAIN_TIME=06:00:00            Wall time per job (QOS maximum: 2 days; debug: 30 minutes)
#   EVAL_TIME=12:00:00
#   EFFECTIVE_BATCH=32             Examples per optimizer update
#   MICRO_BATCH=1                  Per-GPU micro-batch; accumulation steps are derived (8 on 4
#                                  GPUs), as chili trained on 4 x A6000
#   LEARNING_RATE=1e-5             Peak learning rate (align.py: cosine schedule, 10% warmup)
#   DS_CONFIG                      On 40 GB GPUs nersc/ds_zero3_offload.json: chili's ZeRO-3 with
#                                  the optimizer (fp32 weights, Adam moments) on the CPU. Plain
#                                  ZeRO-3 (nersc/ds_zero3.json, the default on hbm80g) ran out of
#                                  memory on the second update of 4,096-token examples (37 of
#                                  40 GB before it: logits of a 128k vocabulary); with offload an
#                                  8B model peaks at ~23 GB (nersc/memtest.py, 2026-11-24).
#                                  ZeRO-2 keeps all bf16 weights on every GPU and fits neither.
#   TRAIN_OMP_THREADS=16           CPU threads per training rank (the offloaded AdamW step)
#   DELETE_CKPTS=0                 1: train.sh --eval deletes each checkpoint once it is evaluated
#   OVERWRITE_STALE=0              1: train.sh retrains runs whose checkpoint or scores come from
#                                  other training data or settings (default: refuses to submit)
#   CKPT_ROOT=$SCRATCH/ckpts       Checkpoints: CKPT_ROOT/seed<seed>/<setting>/
#   RESPONSES_ROOT=$SCRATCH/results
#                                  eval.py --responses_root: the responses, in results/'s tree
#   SELECTION_CACHE=$SCRATCH/selection_cache
#                                  select_data.py --cache_dir: POS tags and embeddings per dataset
#
# Outputs: scores (small CSVs) in results/ as on RANGE; the responses in the same tree under
# RESPONSES_ROOT; figures under figs/; logs in log/slurm/.
# $SCRATCH purges files that have not been read for 8 weeks.
#
# Perlmutter's GPUs are A100s (SM80). There vLLM's batch-invariant mode replaces the matmuls with
# its Triton kernels (Hopper keeps cuBLAS with split-k off), so evaluations repeat exactly on
# A100s but can differ slightly from RANGE's H100/H200 evaluations.

NERSC_DIR=${NERSC_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
SA_DIR=$(dirname -- "$NERSC_DIR")
ROOT=$(dirname -- "$SA_DIR")
export NERSC_DIR SA_DIR ROOT
SCRATCH=${SCRATCH:-/pscratch/sd/${USER:0:1}/$USER}
CKPT_ROOT=${CKPT_ROOT:-$SCRATCH/ckpts}
LOG_DIR=${LOG_DIR:-$ROOT/log/slurm}
mkdir -p -- "$LOG_DIR"

ACCOUNT=${ACCOUNT:-m4705_g} CPU_ACCOUNT=${CPU_ACCOUNT:-m4705}
GPU_CONSTRAINT=${GPU_CONSTRAINT:-gpu&hbm40g}
TRAIN_NGPUS=${TRAIN_NGPUS:-4} TRAIN_QOS=${TRAIN_QOS:-} TRAIN_TIME=${TRAIN_TIME:-06:00:00}
EVAL_QOS=${EVAL_QOS:-} EVAL_TIME=${EVAL_TIME:-12:00:00}
MAX_TRAIN=${MAX_TRAIN:-8} MAX_EVAL=${MAX_EVAL:-16}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32} MICRO_BATCH=${MICRO_BATCH:-1} LEARNING_RATE=${LEARNING_RATE:-1e-5}
if [[ $GPU_CONSTRAINT == *hbm80g* ]]; then
    DS_CONFIG=${DS_CONFIG:-$NERSC_DIR/ds_zero3.json}
else
    DS_CONFIG=${DS_CONFIG:-$NERSC_DIR/ds_zero3_offload.json}
fi
TRAIN_OMP_THREADS=${TRAIN_OMP_THREADS:-16}
DELETE_CKPTS=${DELETE_CKPTS:-0}
OVERWRITE_STALE=${OVERWRITE_STALE:-0}

if [[ "${CONDA_DEFAULT_ENV:-}" != nlp ]]; then
    set +u  # the activation scripts reference unset variables
    source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh" && conda activate nlp || exit 1
    set -u
fi
# The default cudatoolkit module puts CUDA 13.2's forward-compatibility libcuda (595.x) on
# LD_LIBRARY_PATH, ahead of the node's driver (580.x). Nothing is built with nvcc (torch brings its
# CUDA 13.0 libraries and DeepSpeed runs torch's fused AdamW), and RANGE runs on its driver's
# libcuda, so drop the toolkit and its paths.
if [[ ${LD_LIBRARY_PATH:-} == */cuda* || ${LOADEDMODULES:-} == *cudatoolkit* ]]; then
    set +u
    { type module && module unload cudatoolkit; } &>/dev/null
    set -u
    LD_LIBRARY_PATH=$(tr : '\n' <<< "${LD_LIBRARY_PATH:-}" | grep -v -e /cuda -e /hpc_sdk/ | paste -sd:)
    export LD_LIBRARY_PATH
    unset CUDA_HOME
fi
# Triton and torch.compile build small C helpers at run time with CC: GCC 14, as on RANGE.
export CC=/opt/cray/pe/gcc-native/14/bin/gcc CXX=/opt/cray/pe/gcc-native/14/bin/g++
# The Hugging Face cache (models, datasets) is on $SCRATCH (/home has a 40 GB quota); the token
# stays in /home (hf auth login writes ~/.cache/huggingface/token), which is never purged.
export HF_HOME=${HF_HOME:-$SCRATCH/huggingface}
if [[ -z ${HF_TOKEN_PATH:-} && -f $HOME/.cache/huggingface/token ]]; then
    export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
fi
RESPONSES_ROOT=${RESPONSES_ROOT:-$SCRATCH/results}
SELECTION_CACHE=${SELECTION_CACHE:-$SCRATCH/selection_cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$SCRATCH/vllm_cache}  # torch.compile artifacts
# vLLM's FlashInfer top-k/top-p sampler is JIT-compiled at engine start. Every eval call
# decodes greedily (temperature 0), which vLLM does with an argmax that bypasses this sampler,
# so the PyTorch sampler gives the same outputs without compiling anything in each job.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$SCRATCH}
# DeepSpeed warns that Triton's autotune cache on a network file system can hang when it exits.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_$USER}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# gpu_opts N [QOS]: sbatch options for N GPUs of one node (GPU_CONSTRAINT). QOS defaults to
# regular for a whole node (4 GPUs, exclusive) and shared below that; a shared job gets 32 CPUs
# per GPU and memory in proportion (about 60 GB per GPU).
gpu_opts() {
    local qos=${2:-}
    [[ -n $qos ]] || { (( $1 >= 4 )) && qos=regular || qos=shared; }
    echo "--account=$ACCOUNT --constraint=$GPU_CONSTRAINT --qos=$qos --gpus-per-node=$1" \
         "--cpus-per-task=$(( 32 * $1 ))"
}

# cpu_opts CPUS MEM: sbatch options for a CPU job (CPU_ACCOUNT, QOS shared).
cpu_opts() {
    echo "--account=$CPU_ACCOUNT --constraint=cpu --qos=shared --cpus-per-task=$1 --mem=$2"
}

# submit SBATCH_ARGS... SCRIPT [ARGS...]: sbatch with the shared options (the arguments must name
# the account, constraint and QOS: gpu_opts or cpu_opts); prints the job ID and, when
# JOB_IDS_FILE is set, appends it there.
submit() {
    local id
    id=$(sbatch --parsable --export=ALL "$@") || return
    id=${id%%;*}
    [[ -n ${JOB_IDS_FILE:-} ]] && echo "$id" >> "$JOB_IDS_FILE"
    echo "$id"
}

# Names one launch's logs and job lists; host and PID keep launches in the same second apart.
STAMP=$(date +%Y%m%d_%H%M%S)_${HOSTNAME%%.*}_$$

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

# setting_setup: from seed, response_label_filter (safe or refusal), selected (stage 2: the
# selection dataset, or empty), mixed (1: stage 2's mixed-diversity ablation) and num_train
# (default 1800), sets the folders of a setting. Returns 2 on a bad option. (As in
# range/common.sh, so both clusters write the same trees.)
#   setting    default or refusal (stage 2: selected_<dataset>[_refusal][_mixed]), plus
#              _<harmful count> for another size (default_500, selected_wildjailbreak_500)
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
        (( ${mixed:-0} )) && setting+=_mixed
    elif (( ${mixed:-0} )); then
        echo "--mixed_diversity needs --selected" >&2; return 2
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

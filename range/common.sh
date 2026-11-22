# Shared by the range (Rice RANGE cluster, Slurm) launchers and batch scripts; source it.
# Activates the nlp conda env, the one env of every stage (sa_data/requirements.txt: torch 2.11,
# transformers 4.57.3, vLLM 0.23 with batch-invariant kernels), and holds the Slurm defaults, which
# the environment overrides:
#
#   TRAIN_GPU=h200 TRAIN_NGPUS=2   GPUs per training job (full fine-tuning)
#   EVAL_GPU=hopper                GPU of an eval/diversity job: hopper (an H200 or an H100,
#                                  whichever frees first), h200, h100 or 47gb. Batch-invariant
#                                  decoding repeats exactly on one GPU type; types can differ slightly.
#   MAX_TRAIN=8 MAX_EVAL=16        Jobs of one array that may run at once (8 training
#                                  jobs use 16 of the 64 H200s)
#   TRAIN_TIME/EVAL_TIME=04:00:00  Wall time per job (partition maximum: 1 day)
#   EFFECTIVE_BATCH=32             Examples per optimizer update, as on chili (4 GPUs x 1 x 8). Keep
#                                  RANGE and chili on the same batch and lr: at equal settings their
#                                  results agree to within training noise, so seeds can be pooled.
#   MICRO_BATCH=4                  Per-GPU micro-batch; accumulation steps are derived (4 on 2 GPUs).
#                                  Same updates as micro-batch 1, about 3x faster.
#   LEARNING_RATE=1e-5             Peak learning rate (align.py: cosine schedule, 10% warmup). In the
#                                  Nov 2026 lr x batch ablation (ablations/lrbs), 1e-5 vs 5e-6 left
#                                  harmful unchanged, cut over-refusal 3.4 points, cost 2 IFEval points.
#   DS_CONFIG=range/ds_zero2.json  ZeRO-2. Needs DeepSpeed >= 0.19.6 (in nlp since Nov 9, 2026): 0.17.6
#                                  to 0.19.5 compute wrong gradients with accumulation, and train.sh
#                                  refuses those unless ALLOW_ZERO2=1. sa_data/ds_config.json (chili's
#                                  ZeRO-3) gives the same updates.
#   PARTITION=commons ACCOUNT=     Slurm partition / account (account: the default one)

RANGE_DIR=${RANGE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
SA_DIR=$(dirname -- "$RANGE_DIR")
ROOT=$(dirname -- "$SA_DIR")
export RANGE_DIR SA_DIR ROOT
source "$SA_DIR/settings.sh"
LOG_DIR=${LOG_DIR:-$ROOT/log/slurm}
mkdir -p -- "$LOG_DIR"

PARTITION=${PARTITION:-commons}
ACCOUNT=${ACCOUNT:-}
TRAIN_GPU=${TRAIN_GPU:-h200} TRAIN_NGPUS=${TRAIN_NGPUS:-2}
TRAIN_CPUS=${TRAIN_CPUS:-16} TRAIN_MEM=${TRAIN_MEM:-128G} TRAIN_TIME=${TRAIN_TIME:-04:00:00}
EVAL_GPU=${EVAL_GPU:-hopper}
EVAL_CPUS=${EVAL_CPUS:-8} EVAL_MEM=${EVAL_MEM:-64G} EVAL_TIME=${EVAL_TIME:-04:00:00}
MAX_TRAIN=${MAX_TRAIN:-8} MAX_EVAL=${MAX_EVAL:-16}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32} MICRO_BATCH=${MICRO_BATCH:-4} LEARNING_RATE=${LEARNING_RATE:-1e-5}
DS_CONFIG=${DS_CONFIG:-$RANGE_DIR/ds_zero2.json}

if [[ "${CONDA_DEFAULT_ENV:-}" != nlp ]]; then
    set +u  # the activation scripts reference unset variables
    source /home/zf28/miniconda3/etc/profile.d/conda.sh && conda activate nlp || exit 1
    set -u
fi
# Nothing is built with nvcc: DeepSpeed runs torch's own AdamW (torch_adam in the DeepSpeed
# configs), which ships compiled. Triton and torch.compile build small C helpers at run time with
# CC, GCC 14 from the modules (compute nodes have no system g++). GCC 14's libraries need
# GLIBCXX_3.4.32+, so the env's libstdc++ was upgraded to conda-forge's (conda revision 1).
export CC=/opt/apps/software/GCCcore/14.2.0/bin/gcc CXX=/opt/apps/software/GCCcore/14.2.0/bin/g++
# The Hugging Face cache (models, datasets, token) is on scratch: /home has a 50 GB quota.
# ~/.cache/huggingface is a symlink to it as well.
export HF_HOME=${HF_HOME:-/scratch/zf28/huggingface}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/scratch/zf28/vllm_cache}  # torch.compile artifacts
# vLLM's FlashInfer top-k/top-p sampler is JIT-compiled at engine start. Every eval call
# decodes greedily (temperature 0), which vLLM does with an argmax that bypasses this sampler,
# so the PyTorch sampler gives the same outputs without compiling anything in each job.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/scratch/zf28}
# DeepSpeed warns that Triton's autotune cache on NFS (~/.triton) can hang when it exits.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_$USER}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# gpu_opts N: sbatch options for N GPUs of type EVAL_GPU. Slurm cannot ask for "h100 or h200"
# in --gres, so hopper asks for any N GPUs and excludes the nodes that have other GPUs
# (bg4u7g1 and bg4u9g1, 47gb + 24gb: a 24 GB card is too small for eval).
gpu_opts() {
    if [[ $EVAL_GPU == hopper ]]; then
        local others
        others=$(sinfo -h -N -p "$PARTITION" -o '%N %G' | awk '$2 !~ /h100|h200/ { print $1 }' |
                 sort -u | paste -sd,)
        echo "--gres=gpu:$1${others:+ --exclude=$others}"
    else
        echo "--gres=gpu:$EVAL_GPU:$1"
    fi
}

# submit SBATCH_ARGS... SCRIPT [ARGS...]: sbatch with the shared options; prints the job ID
# and, when JOB_IDS_FILE is set, appends it there.
submit() {
    local id
    local -a common=(--parsable --partition="$PARTITION" --export=ALL)
    [[ -n $ACCOUNT ]] && common+=(--account="$ACCOUNT")
    id=$(sbatch "${common[@]}" "$@") || return
    id=${id%%;*}
    [[ -n ${JOB_IDS_FILE:-} ]] && echo "$id" >> "$JOB_IDS_FILE"
    echo "$id"
}

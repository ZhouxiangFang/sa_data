# Shared by the range (Rice RANGE cluster, Slurm) launchers and batch scripts; source it.
# Sets up the nlp environment and holds the Slurm defaults, which the environment overrides:
#
#   TRAIN_GPU=h200 TRAIN_NGPUS=2   GPUs per training job (ZeRO-3 full fine-tuning)
#   EVAL_GPU=hopper                GPU of an eval/diversity job: hopper (an H200 or an H100,
#                                  whichever frees first), or one type: h100, h200, 47gb
#   MAX_TRAIN=4 MAX_EVAL=4         Jobs of one array that may run at once
#   TRAIN_TIME/EVAL_TIME=04:00:00  Wall time per job (partition maximum: 1 day)
#   EFFECTIVE_BATCH=32             Examples per optimizer update, as on chili (4 GPUs x 1 x 8)
#   MICRO_BATCH=4                  Per-GPU micro-batch; accumulation steps are derived (4)
#   DS_CONFIG=range/ds_zero2.json  DeepSpeed config. ZeRO-2 keeps the parameters on every GPU
#                                  (about 72 GB of an H200's 141 GB), so unlike ZeRO-3 it gathers no
#                                  weights in each forward and backward pass; sa_data/ds_config.json is
#                                  chili's ZeRO-3 one. Each update still has the same 32 examples.
#   PARTITION=commons ACCOUNT=     Slurm partition / account (account: the default one)

RANGE_DIR=${RANGE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
SA_DIR=$(dirname -- "$RANGE_DIR")
ROOT=$(dirname -- "$SA_DIR")
export RANGE_DIR SA_DIR ROOT
LOG_DIR=${LOG_DIR:-$ROOT/log/slurm}
mkdir -p -- "$LOG_DIR"

PARTITION=${PARTITION:-commons}
ACCOUNT=${ACCOUNT:-}
TRAIN_GPU=${TRAIN_GPU:-h200} TRAIN_NGPUS=${TRAIN_NGPUS:-2}
TRAIN_CPUS=${TRAIN_CPUS:-16} TRAIN_MEM=${TRAIN_MEM:-128G} TRAIN_TIME=${TRAIN_TIME:-04:00:00}
EVAL_GPU=${EVAL_GPU:-hopper}
EVAL_CPUS=${EVAL_CPUS:-8} EVAL_MEM=${EVAL_MEM:-64G} EVAL_TIME=${EVAL_TIME:-04:00:00}
MAX_TRAIN=${MAX_TRAIN:-4} MAX_EVAL=${MAX_EVAL:-4}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32} MICRO_BATCH=${MICRO_BATCH:-4}
DS_CONFIG=${DS_CONFIG:-$RANGE_DIR/ds_zero2.json}
STAMP=$(date +%Y%m%d_%H%M%S)

if [[ "${CONDA_DEFAULT_ENV:-}" != nlp ]]; then
    set +u  # the activation scripts reference unset variables
    source /home/zf28/miniconda3/etc/profile.d/conda.sh && conda activate nlp || exit 1
    set -u
fi
# DeepSpeed JIT-builds FusedAdam on the first training job (cached in ~/.cache/torch_extensions)
# with nvcc from the CUDA module and GCC 14 from the modules: compute nodes have no system g++.
# GCC 14's libraries need GLIBCXX_3.4.32+, so the env's libstdc++ was upgraded to conda-forge's
# (conda revision 1; the old 11.2 one stopped at 3.4.29). Only nvcc is put on PATH; the CUDA
# module's libraries stay off LD_LIBRARY_PATH so torch and vLLM keep their bundled CUDA.
export CUDA_HOME=${CUDA_HOME:-/opt/apps/software/CUDA/12.8.0}
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/opt/apps/software/GCCcore/14.2.0/bin/gcc CXX=/opt/apps/software/GCCcore/14.2.0/bin/g++
for include_dir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/include; do
    [[ -d $include_dir && ":${CPATH:-}:" != *":$include_dir:"* ]] &&
        export CPATH="$include_dir${CPATH:+:$CPATH}"
done
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

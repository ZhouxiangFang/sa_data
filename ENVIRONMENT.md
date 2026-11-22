# Environment and fixed settings

What a run depends on, and how to rebuild it on RANGE (Slurm, H200/H100) or chili (8 × A6000).
Branch `vllm033-torch214`, 2026-11-22.

## One env for every stage

`/scratch/zf28/envs/align_t211`, a venv over the `nlp` conda env's Python 3.12, pinned by
`requirements.txt`. Training, evaluation, diversity, data selection and plots all run in it, so
training and evaluation see the same torch, transformers and tokenizers.

| Package | Version | Why this one |
|---|---|---|
| torch | 2.11.0 (CUDA 13.0) | the version vLLM 0.23.0 pins; PyTorch builds for CUDA 13.0 up to 2.14 (2.15: 13.2+ only) |
| vllm | 0.23.0 | newest vLLM that still accepts transformers 4.x; reads `VLLM_BATCH_INVARIANT` (0.11 ignored it) |
| transformers | 4.57.3 | unchanged; transformers 5.20 (vLLM ≥ 0.24 needs ≥ 5.5) tokenized WildGuard wrongly |
| trl / deepspeed / accelerate | 0.25.1 / 0.19.6 / 1.10.1 | unchanged (DeepSpeed ≥ 0.19.6 for ZeRO-2 with accumulation) |

The old `nlp` env (torch 2.8, vLLM 0.11) stays as it is; `ALIGN_VENV=` (empty) runs a stage in it.
`eval.py` refuses vLLM 0.11, since its outputs depend on batching.

## Requirements outside pip

- **NVIDIA driver ≥ 580** (CUDA 13.0 wheels). RANGE H200 nodes: 580.173.02. Chili: check
  `nvidia-smi`; `chili/common.sh` adds user-space libraries for driver 595.84.
- **CUDA 13.0.2 toolkit (nvcc 13.0.88)** at `/scratch/zf28/cuda/13.0.2` on both clusters, for
  DeepSpeed's FusedAdam JIT build (training only); the `common.sh` of each cluster sets `CUDA_HOME`
  to it and unsets CPATH. It is the CUDA torch 2.11 is built with. Not other versions:
  - nvcc 13.3 (RANGE's module) drops a `typename` from torch's `ATen/core/List_inl.h:202` and the
    build fails with every GCC from 12 to 15 (job 420705).
  - CUDA 13's headers refuse an nvcc of another minor version ("CUDA compiler and CUDA toolkit
    headers are incompatible"), so the venv's pip CUDA packages (13.0 runtime headers, nvcc 13.2,
    cccl 13.3) can't be used either.

  GCC 14 (RANGE: `/opt/apps`, compute nodes have no g++); GCC ≤ 15.
- **Python 3.12.11**, the `nlp` conda env's.
- **Hugging Face snapshots**: the code loads each repo's `main`, so check that the cache
  (`$HF_HOME/hub/*/refs/main`; RANGE `HF_HOME=/scratch/zf28/huggingface`) holds these revisions:

  | Models | Revision | Datasets | Revision |
  |---|---|---|---|
  | FocusV857/llama3.1_git20k | `05b0426c0fac` | allenai/wildguardmix | `d29c47f41c8b` |
  | FocusV857/mistral_git20k | `4bfe7aacd29d` | allenai/wildjailbreak | `5ddc12a7894f` |
  | FocusV857/olmo2_git20k | `c2c5e0434584` | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | `d86bb8bedff5` |
  | FocusV857/qwen2.5_git20k | `4d223cec223a` | gretelai/gretel-safety-alignment-en-v1 | `7f220b493d70` |
  | meta-llama/Meta-Llama-3.1-8B-Instruct | `0e9e39f249a1` | bench-llm/or-bench | `e36d8b80e818` |
  | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41d` | Paul/XSTest | `f600c994b256` |
  | allenai/OLMo-2-1124-7B-Instruct | `470b1fba1ae0` | google/IFEval | `966cd89545d6` |
  | Qwen/Qwen2.5-7B-Instruct | `a09a35458c70` | Babelscape/ALERT | `c76d0637f457` |
  | allenai/wildguard | `cbba4823f3e8` | Babelscape/ALERT_DPO | `7542cdc795ca` |
  | BAAI/bge-m3 | `5617a9f61b02` | nvidia/Nemotron-SFT-Safety-v2 | `8a40a63c9a1a` |
  | Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d` | | |

  IFBench is pinned by commit in `requirements.txt`.
- **NLTK data** in `~/nltk_data` for the IF scorers (`range/prefetch.sh`).

## Build

```bash
PY=~/miniconda3/envs/nlp/bin/python          # Python 3.12; keeps nlp's OpenSSL 3.0.x
V=/scratch/zf28/envs/align_t211/bin/python
$PY -m venv /scratch/zf28/envs/align_t211
$V -m pip install -U pip setuptools wheel
$V -m pip install torch==2.11.0 numpy==2.2.6 ninja packaging psutil py-cpuinfo hjson msgpack einops "pydantic>=2" tqdm==4.67.1
DS_BUILD_OPS=0 $V -m pip install --no-build-isolation deepspeed==0.19.6   # built against that torch
$V -m pip install -r ~/align/sa_data/requirements.txt
$V -m pip check
```

CUDA 13.0.2 toolkit (1.8 GB; the conda package cache goes to /scratch, /home has a quota):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
CONDA_PKGS_DIRS=/scratch/zf28/conda_pkgs conda create -y -p /scratch/zf28/cuda/13.0.2 \
    --override-channels -c nvidia -c conda-forge "cuda-version=13.0" "cuda-nvcc-impl=13.0.88" \
    "cuda-cudart-dev=13.0.96" "cuda-cccl=13.0.*" libcublas-dev libcusparse-dev libcusolver-dev \
    libcurand-dev cuda-nvrtc-dev cuda-nvtx-dev cuda-profiler-api
ln -sfn targets/x86_64-linux/include /scratch/zf28/cuda/13.0.2/include   # torch looks for
ln -sfn targets/x86_64-linux/lib /scratch/zf28/cuda/13.0.2/lib64         # include/ and lib64/
/scratch/zf28/cuda/13.0.2/bin/nvcc --version                              # V13.0.88
```

/scratch purges old files; rebuild the venv from `requirements.txt` and the toolkit when they
disappear.

## Running

**RANGE**: `range/common.sh` activates nlp, then `ALIGN_VENV`, for every job; nothing to do.

| Variable | Default | Meaning |
|---|---|---|
| `ALIGN_VENV` | `/scratch/zf28/envs/align_t211` | empty = the old nlp env |
| `EVAL_GPU` | `h200` | keep evals on one GPU type (batch invariance holds per GPU type) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | greedy decoding bypasses it; avoids a JIT build per job |
| `DS_CONFIG` | `range/ds_zero2.json` | ZeRO-2 |

**Chili**: `chili/common.sh` activates the same `ALIGN_VENV` and toolkit (build both at those paths first).
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
chili/train.sh --models llama3.1_git20k --datasets wildguardmix --abbrs random --seed 43 --output_dir DIR
chili/eval.sh MODEL_OR_CKPT -- --output_dir DIR
```

## Fixed training settings

| | RANGE | chili |
|---|---|---|
| GPUs per job | 2 × H200 | 4 × A6000 |
| ZeRO | 2 (`range/ds_zero2.json`) | 3 (`ds_config.json`) |
| Micro-batch × accumulation | 4 × 4 | 1 × 8 |
| Effective batch | 32 | 32 |

Both: lr 1e-5, cosine schedule, 10% warmup, 2 epochs, weight decay 0.01, max grad norm 1, bf16
(TF32 on), max length 4096, completion-only loss, seeds 43 / 86 / 129. Seeds from the two clusters
can be pooled only if these match.

## Determinism measures

- **Training data**: one sample file per seed and setting (`data/train_samples/seed<S>/...`),
  reused by every model and by the diversity scores; `align.py` logs each file's SHA-256.
- **Training**: `seed` and `data_seed` = run seed. Not bitwise deterministic (TF32 on, no
  `full_determinism`): reruns agree to about 0.6% in grad norm.
- **Generation**: greedy, vLLM seed 42, `VLLM_BATCH_INVARIANT=1` (eval.py default) with the
  FLASH_ATTN backend; eval.py prints `vLLM 0.23.0: batch-invariant, FLASH_ATTN attention`. Outputs
  are independent of batch size and order on one GPU type (checked: see Checks); H100, H200 and
  A6000 can differ.
- **IF scoring**: `eval_if.evaluate` seeds `random` and langdetect (`SCORING_SEED = 0`); before,
  the same responses scored up to 0.55 IFEval points apart.
- **Tokenizers**: eval.py refuses a model whose tokenizer splits a probe text unlike its own
  SentencePiece `tokenizer.model` (transformers 5.20 did this to WildGuard).
- **A6000**: batch invariance is supported (compute capability ≥ 8.0), but vLLM's tuned fast path
  covers Ada, Hopper and Blackwell only, so expect slower evals.

## Checks

**Eval repeatability** (passed 2026-11-21, H200): evaluate one model twice and once with
`--limit 100`, then `python compare_evals.py RUN_A RUN_B MODEL...` and
`python compare_evals.py RUN_A RUN_LIMIT100 MODEL... --limit 100`. For llama3.1, mistral, olmo2 and
qwen2.5 `_git20k` (jobs 420696, 420697, 420698), all 9 test sets had 100% identical responses and
WildGuard labels in both comparisons, and identical scores between the two full runs.

**New env vs old training** (done 2026-11-22, H200): train llama3.1_git20k / wildguardmix / random / seed 43
twice in nlp (`ALIGN_VENV=`, torch 2.8: jobs 420689, 420690, done) and once in the align env (job
420723), and evaluate the three checkpoints in the align env (420716, 420717, 420725). The two
torch 2.8 runs give the training noise. The training logs must show these digests:

| File (`data/train_samples/seed43/`) | SHA-256 (12) |
|---|---|
| `wildguardmix/random_600.jsonl` | `a8183614dcbc` |
| `wildjailbreak/vanilla_benign_600.jsonl` | `82f0807ab815` |
| `general/git_data_20k_600.jsonl` | `5312175b53c2` |

```bash
python compare_train_logs.py t28_a=LOG t28_b=LOG t211=LOG
python /scratch/zf28/ckpts/abtest/weight_diff.py BASE_MODEL_DIR CKPT...
python compare_evals.py RES_T28_A RES_T28_B MODEL       # and RES_T211 against each
```

Pass: the align-env run differs from each torch 2.8 run no more than the two torch 2.8 runs differ
from each other (per-step loss, grad norm), every checkpoint moved from its base model, and the
align-env checkpoint's harmful / over-refusal / IF scores are within the two torch 2.8 ones' spread.

Result: the align env differs from torch 2.8 about as much as two torch 2.8 runs differ from each
other. Training is not bitwise repeatable in either env, so a rerun of the same seed moves scores by
up to about 1 point per test set.

| | torch 2.8 A vs B | align env vs torch 2.8 A / B |
|---|---|---|
| Loss, mean relative difference | 0.037% | 0.035% / 0.032% |
| Grad norm, mean (max) relative difference | 0.33% (1.4%) | 0.46% (2.4%) / 0.70% (3.1%) |
| Relative weight change from base | 5.004e-3 / 5.003e-3 | 5.005e-3 |
| Identical responses (IFEval, aegis, or-bench-hard) | 30%, 53%, 26% | 31%, 52%, 25% / 32%, 53%, 26% |
| Largest score difference, one test set | 0.75 | 0.66 / 1.0 |

| Score | torch 2.8 A | torch 2.8 B | align env |
|---|---|---|---|
| Harmful, Avg | 9.33 | 9.13 | 9.44 |
| Over-refusal, Avg | 16.34 | 16.15 | 16.04 |
| IF, Average | 36.56 | 36.30 | 36.99 |

Every result evaluated in this env needs its training rerun too: checkpoints are deleted after
evaluation, and batch-invariant decoding changes eval outputs.

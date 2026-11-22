# Environment and fixed settings

What a run depends on, and how to rebuild it on RANGE (Slurm, H200/H100) or chili (8 × A6000).
Branch `vllm033-torch214`, 2026-11-21.

## One env for every stage

`/scratch/zf28/envs/align_t211`, a venv over the `nlp` conda env's Python 3.12, pinned by
`requirements.txt`. Training, evaluation, diversity, data selection and plots all run in it, so
training and evaluation see the same torch, transformers and tokenizers.

| Package | Version | Why this one |
|---|---|---|
| torch | 2.11.0 (CUDA 13.0) | the version vLLM 0.23.0 pins |
| vllm | 0.23.0 | newest vLLM that still accepts transformers 4.x; reads `VLLM_BATCH_INVARIANT` (0.11 ignored it) |
| transformers | 4.57.3 | unchanged; transformers 5.20 (vLLM ≥ 0.24 needs ≥ 5.5) tokenized WildGuard wrongly |
| trl / deepspeed / accelerate | 0.25.1 / 0.19.6 / 1.10.1 | unchanged (DeepSpeed ≥ 0.19.6 for ZeRO-2 with accumulation) |

The old `nlp` env (torch 2.8, vLLM 0.11) stays as it is; `ALIGN_VENV=` (empty) runs a stage in it.
`eval.py` refuses vLLM 0.11, since its outputs depend on batching.

## Requirements outside pip

- **NVIDIA driver ≥ 580** (CUDA 13.0 wheels). RANGE H200 nodes: 580.173.02. Chili: check
  `nvidia-smi`; `chili/common.sh` adds user-space libraries for driver 595.84.
- **A full CUDA 13 toolkit** for DeepSpeed's FusedAdam JIT build (training only), used with its own
  headers only: CUDA 13's headers refuse an nvcc of another minor version ("CUDA compiler and CUDA
  toolkit headers are incompatible"). The venv's pip CUDA packages can't be used for this (13.0
  runtime headers, nvcc 13.2, cccl 13.3). RANGE: `range/common.sh` sets the CUDA 13.3.0 module and
  unsets CPATH; GCC 14 from `/opt/apps` (compute nodes have no g++). Chili: export `CUDA_HOME` to a
  CUDA 13.x toolkit; `chili/common.sh` unsets CPATH and warns if nvcc isn't 13.x. GCC ≤ 15.
- **Hugging Face cache**: RANGE `HF_HOME=/scratch/zf28/huggingface`. Model and dataset revisions
  are **not pinned**; use the same snapshots on both clusters.
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

/scratch purges old files; rebuild the venv from `requirements.txt` when it disappears.

## Running

**RANGE**: `range/common.sh` activates nlp, then `ALIGN_VENV`, for every job; nothing to do.

| Variable | Default | Meaning |
|---|---|---|
| `ALIGN_VENV` | `/scratch/zf28/envs/align_t211` | empty = the old nlp env |
| `EVAL_GPU` | `h200` | keep evals on one GPU type (batch invariance holds per GPU type) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | greedy decoding bypasses it; avoids a JIT build per job |
| `DS_CONFIG` | `range/ds_zero2.json` | ZeRO-2 |

**Chili**: `chili/common.sh` activates the same `ALIGN_VENV` (build it at that path first).
```bash
export CUDA_HOME=<CUDA 13 toolkit> VLLM_USE_FLASHINFER_SAMPLER=0
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
  are independent of batch size and order on one GPU type; H100, H200 and A6000 can differ.
- **IF scoring**: `eval_if.evaluate` seeds `random` and langdetect (`SCORING_SEED = 0`); before,
  the same responses scored up to 0.55 IFEval points apart.
- **Tokenizers**: eval.py refuses a model whose tokenizer splits a probe text unlike its own
  SentencePiece `tokenizer.model` (transformers 5.20 did this to WildGuard).
- **A6000**: batch invariance is supported (compute capability ≥ 8.0), but vLLM's tuned fast path
  covers Ada, Hopper and Blackwell only, so expect slower evals.

## Checks

**Eval repeatability**: evaluate one model twice and once with `--limit 100`, then
`python compare_evals.py RUN_A RUN_B MODEL...` (expect identical responses) and
`python compare_evals.py RUN_A RUN_LIMIT100 MODEL... --limit 100`.

**New env vs old training**: train llama3.1_git20k / wildguardmix / random / seed 43 twice in nlp
(`ALIGN_VENV=`, torch 2.8) and twice in the align env. The training logs must show these digests:

| File (`data/train_samples/seed43/`) | SHA-256 (12) |
|---|---|
| `wildguardmix/random_600.jsonl` | `a8183614dcbc` |
| `wildjailbreak/vanilla_benign_600.jsonl` | `82f0807ab815` |
| `general/git_data_20k_600.jsonl` | `5312175b53c2` |

```bash
python compare_train_logs.py t28_a=LOG t28_b=LOG t211_a=LOG t211_b=LOG
python /scratch/zf28/ckpts/abtest/weight_diff.py BASE_MODEL_DIR CKPT...
```

Pass: the new env differs from torch 2.8 no more than the two torch 2.8 runs differ from each
other (per-step loss, grad norm), every checkpoint moved from its base model, and the four
checkpoints' harmful / over-refusal / IF scores agree within run noise.

Every result evaluated in this env needs its training rerun too: checkpoints are deleted after
evaluation, and batch-invariant decoding changes eval outputs.

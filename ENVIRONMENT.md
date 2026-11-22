# Environments and fixed settings

What a run depends on, and how to rebuild it on RANGE (Slurm, H200/H100) or chili (8 × A6000).
Written 2026-11-21 on branch `vllm033-torch214`.

## The three environments

| Env | Used for | Key versions | Pinned by |
|---|---|---|---|
| conda `nlp` | training (current default), diversity, data selection, plots | Python 3.12, torch 2.8.0 (CUDA 12.8), transformers 4.57.3, TRL 0.25.1, DeepSpeed 0.19.6, accelerate 1.10.1 | `requirements.txt` |
| venv `/scratch/zf28/envs/vllm0330` | evaluation (`eval.py`) | vLLM 0.33.0, torch 2.14.0 (CUDA 13.0), transformers 5.20.2 | `requirements-eval.txt` |
| venv `/scratch/zf28/envs/train_t214` | training on torch 2.14 (**under test**) | torch 2.14.0 (CUDA 13.0); everything else as in `requirements.txt` | `requirements.txt` with torch swapped |

Evaluation has its own env because vLLM pins its own torch, and vLLM 0.11 (in `nlp`) silently
ignores `VLLM_BATCH_INVARIANT`. `eval.py` refuses to run on a vLLM that ignores it.

Both venvs are built from nlp's Python, so they keep its OpenSSL (RANGE injects libcrypto 3.2.2
into Slurm jobs; keep nlp's OpenSSL at 3.0.x). /scratch purges old files, so rebuild the venvs from
the requirements files when they disappear.

## Requirements outside pip

- **NVIDIA driver ≥ 580** for the torch 2.14 wheels (CUDA 13.0). RANGE H200 nodes: 580.173.02.
  Chili: `chili/common.sh` adds user-space libraries when the driver is 595.84.
- **CUDA 13 `nvcc`** for DeepSpeed's FusedAdam JIT build under torch 2.14 (CUDA 12.8 for nlp).
  RANGE: module `/opt/apps/software/CUDA/13.3.0` (12.8.0 for nlp). Host compiler: GCC 14 from
  `/opt/apps/software/GCCcore/14.2.0` on RANGE (compute nodes have no g++); GCC ≤ 15 elsewhere.
- **Hugging Face cache**: RANGE `HF_HOME=/scratch/zf28/huggingface`. Model and dataset
  revisions are **not pinned**; prefetch the same snapshots on both clusters.
- **NLTK data** in `~/nltk_data` (IFEval/IFBench scorers); `range/prefetch.sh` downloads it.

## Build

```bash
PY=~/miniconda3/envs/nlp/bin/python   # Python 3.12

# Eval env
$PY -m venv /scratch/zf28/envs/vllm0330
/scratch/zf28/envs/vllm0330/bin/python -m pip install -r ~/align/sa_data/requirements-eval.txt

# Training env on torch 2.14 (DeepSpeed must be built against the torch already installed)
$PY -m venv /scratch/zf28/envs/train_t214
V=/scratch/zf28/envs/train_t214/bin/python
$V -m pip install -U pip setuptools wheel
$V -m pip install torch==2.14.0 numpy==2.2.6 ninja packaging psutil py-cpuinfo hjson msgpack einops "pydantic>=2" tqdm==4.67.1
DS_BUILD_OPS=0 $V -m pip install --no-build-isolation deepspeed==0.19.6
sed 's/^torch==2.8.0$/torch==2.14.0/' ~/align/sa_data/requirements.txt > /tmp/req_t214.txt
$V -m pip install -r /tmp/req_t214.txt && $V -m pip check
```

Then append the CUDA 13 toolkit to `/scratch/zf28/envs/train_t214/bin/activate`:

```bash
export CUDA_HOME=/opt/apps/software/CUDA/13.3.0   # RANGE; on chili its CUDA 13 path
export PATH="$CUDA_HOME/bin:$PATH"
unset CPATH                                       # RANGE: drop nlp's CUDA 12 headers
```

Without a CUDA 13 toolkit (untested): `$V -m pip install nvidia-cuda-nvcc nvidia-cuda-crt
nvidia-cuda-cccl` and `CUDA_HOME=/scratch/zf28/envs/train_t214/lib/python3.12/site-packages/nvidia/cu13`.

Check: `pip install --dry-run -r <requirements file>` inside each env should install nothing.

## Running

**RANGE.** `range/common.sh` sets everything; override through the environment.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_VENV` | `/scratch/zf28/envs/vllm0330` | eval.sbatch activates it; empty = nlp (eval.py then refuses) |
| `TRAIN_VENV` | empty (nlp) | `/scratch/zf28/envs/train_t214` trains on torch 2.14 |
| `EVAL_GPU` | `hopper` (H100 or H200) | use `h200` for strictly repeatable evals (see below) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | greedy decoding bypasses it; avoids a JIT build per job |
| `DS_CONFIG` | `range/ds_zero2.json` | ZeRO-2; needs DeepSpeed ≥ 0.19.6 |

**Chili.** The launchers have no venv options: activate nlp, then the venv, then run the launcher
(`common.sh` keeps the venv first on PATH and takes its CUDA headers).

```bash
conda activate nlp && source /scratch/zf28/envs/train_t214/bin/activate   # torch 2.14 training
chili/train.sh --models llama3.1_git20k --datasets wildguardmix --abbrs random --seed 43 --output_dir DIR

conda activate nlp && source /scratch/zf28/envs/vllm0330/bin/activate     # evaluation
export VLLM_USE_FLASHINFER_SAMPLER=0
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
(TF32 on), max length 4096, completion-only loss, seed 43 / 86 / 129. At the same settings, RANGE
and chili agree to within training noise, so seeds can be pooled only if these match.

## Determinism measures

- **Training data**: one sample file per seed and run setting
  (`data/train_samples/seed<S>/...`), drawn once and reused by every model and by the diversity
  scores; `align.py` logs each file's SHA-256 (first 12 hex digits).
- **Training**: `seed` and `data_seed` = run seed. Not bitwise deterministic (no
  `full_determinism`, TF32 on): reruns match to about 0.6% in grad norm.
- **Generation**: greedy (temperature 0), vLLM seed 42, `VLLM_BATCH_INVARIANT=1` (eval.py
  default) with the FLASH_ATTN backend. eval.py prints `vLLM 0.33.0: batch-invariant, FLASH_ATTN
  attention`. Batch invariance makes outputs independent of batch size and order **on one GPU
  type**; H100, H200 and A6000 can still differ from each other.
- **IF scoring**: `eval_if.evaluate` seeds `random` and langdetect (`SCORING_SEED = 0`). Before,
  the same responses scored up to 0.55 IFEval points apart.
- **A6000**: batch invariance is supported (compute capability ≥ 8.0), but vLLM's tuned fast
  path covers Ada, Hopper and Blackwell only, so expect slower evals.

## Checks

**Eval repeatability.** Evaluate the same model twice into two folders, and once with `--limit 100`:

```bash
python compare_evals.py RUN_A RUN_B llama3.1_git20k ...           # expect 100% identical responses
python compare_evals.py RUN_A RUN_LIMIT100 llama3.1_git20k ... --limit 100
```

**torch 2.14 training.** Train llama3.1_git20k / wildguardmix / random / seed 43 twice on torch 2.8
(nlp) and twice on torch 2.14. The training logs must show these sample digests:

| File (`data/train_samples/seed43/`) | SHA-256 (12) |
|---|---|
| `wildguardmix/random_600.jsonl` | `a8183614dcbc` |
| `wildjailbreak/vanilla_benign_600.jsonl` | `82f0807ab815` |
| `general/git_data_20k_600.jsonl` | `5312175b53c2` |

```bash
python compare_train_logs.py t28_a=LOG t28_b=LOG t214_a=LOG t214_b=LOG
python /scratch/zf28/ckpts/abtest/weight_diff.py BASE_MODEL_DIR CKPT...
```

Pass: torch 2.14 vs 2.8 per-step loss and grad norm differ no more than the two 2.8 runs do;
every checkpoint moved from the base (the ZeRO-2 bug left weights bit-identical while the loss
looked normal); harmful / over-refusal / IF scores of all four checkpoints agree within run noise.

## Status (2026-11-21)

- Eval env: built on RANGE; imports and unit tests pass; GPU repeatability jobs 420671–420675
  (H200 twice, H200 `--limit 100`, H100) queued.
- torch 2.14 training env: built on RANGE; imports pass; check jobs 420676/420677 (torch 2.8)
  and 420679/420680 (torch 2.14) queued.
- Every result evaluated after this change needs its training rerun as well: checkpoints are
  deleted after evaluation, and the new vLLM changes eval outputs.

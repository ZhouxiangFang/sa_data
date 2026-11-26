# Environment and fixed settings

What a run depends on, and how to rebuild it on RANGE (Slurm, H200/H100).

## Environment

One conda env, `nlp` (Python 3.12.11), for every stage: training, evaluation, diversity, data
selection and plots. `requirements.txt` lists what the code uses; `constraints.txt` pins every
package, dependencies included, to the versions verified on 2026-11-22.

| Package | Version | Why |
|---|---|---|
| torch | 2.11.0 (CUDA 13.0) | the version vLLM 0.23.0 pins |
| vllm | 0.23.0 | batch-invariant decoding; newest vLLM that accepts transformers 4.x |
| transformers | 4.57.3 | 5.x tokenizes WildGuard wrongly |
| deepspeed | 0.19.6 | 0.17.6–0.19.5 compute wrong ZeRO-2 gradients with accumulation |

Outside pip:
- **NVIDIA driver ≥ 580** (the CUDA 13.0 libraries come with torch). RANGE: 580.173.02.
- **A C compiler** for the small helpers Triton and torch.compile build at run time (RANGE: GCC 14
  from `/opt/apps`, set by `range/common.sh`). No CUDA toolkit: nothing is compiled with nvcc.
- **NLTK data** in `~/nltk_data` for the IF scorers (`range/prefetch.sh`).
- **Hugging Face snapshots**: the code loads each repo's `main`; check that the cache
  (`$HF_HOME/hub/*/refs/main`, RANGE `HF_HOME=/scratch/zf28/huggingface`) holds these revisions.
  IFBench is pinned by commit in `requirements.txt`.

  | Model | Revision | Dataset | Revision |
  |---|---|---|---|
  | FocusV857/llama3.1_git20k | `05b0426c0fac` | allenai/wildguardmix | `d29c47f41c8b` |
  | FocusV857/mistral_git20k | `4bfe7aacd29d` | allenai/wildjailbreak | `5ddc12a7894f` |
  | FocusV857/olmo2_git20k | `c2c5e0434584` | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | `d86bb8bedff5` |
  | FocusV857/qwen2.5_git20k | `4d223cec223a` | sorry-bench/sorry-bench-202503 | `612a4e1f45db` |
  | meta-llama/Meta-Llama-3.1-8B-Instruct | `0e9e39f249a1` | bench-llm/or-bench | `e36d8b80e818` |
  | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41d` | Paul/XSTest | `f600c994b256` |
  | allenai/OLMo-2-1124-7B-Instruct | `470b1fba1ae0` | google/IFEval | `966cd89545d6` |
  | Qwen/Qwen2.5-7B-Instruct | `a09a35458c70` | Babelscape/ALERT | `c76d0637f457` |
  | allenai/wildguard | `cbba4823f3e8` | Babelscape/ALERT_DPO | `7542cdc795ca` |
  | BAAI/bge-m3 | `5617a9f61b02` | nvidia/Nemotron-SFT-Safety-v2 | `8a40a63c9a1a` |
  | Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d` | | |

## Build

In `sa_data`. Keep nlp's conda OpenSSL at 3.0.x (RANGE preloads libcrypto 3.2.2 into Slurm jobs).
When upgrading an nlp that still has torch 2.8, first uninstall xformers, cupy-cuda12x and the
`nvidia-*-cu12` packages.

```bash
P=~/miniconda3/envs/nlp/bin/python
export PIP_CACHE_DIR=/scratch/zf28/pip_cache          # /home has a 50 GB quota
$P -m pip install -c constraints.txt torch==2.11.0 numpy ninja packaging psutil py-cpuinfo hjson msgpack einops pydantic tqdm
DS_BUILD_OPS=0 $P -m pip install -c constraints.txt --no-build-isolation deepspeed   # against that torch
$P -m pip install -c constraints.txt -r requirements.txt
$P -m pip check
```

## Running

**RANGE**: `range/common.sh` activates nlp for every job.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_GPU` | `hopper` | an H200 or H100, whichever frees first (`h200`, `h100`, `47gb` also work) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | greedy decoding bypasses it; avoids a JIT build per job |
| `DS_CONFIG` | `range/ds_zero2.json` | ZeRO-2 |

## Fixed training settings

2 × H200 per job, ZeRO-2 (`range/ds_zero2.json`), micro-batch 4 × 4 accumulation steps =
effective batch 32, torch's fused AdamW (`"torch_adam": true, "fused": true`), lr 1e-5, cosine
schedule, 10% warmup, 2 epochs, weight decay 0.01, max grad norm 1, bf16 (TF32 on), max length
4096, completion-only loss, seeds 43 / 86 / 129. (`chili/`, for chili01's 8 × A6000, runs
ZeRO-3 with 4 × A6000 × 1 × 8; it agreed within training noise.)

## Determinism

- **Training data**: one file per seed and setting (`samples/stage1/`, `samples/common/`,
  `samples/stage2_selected/`), reused by every model and by the diversity scores; `align.py` logs
  each file's SHA-256 and its fingerprint (`run.json`).
- **Training**: `seed` and `data_seed` = run seed. Not bitwise repeatable (TF32, no
  `full_determinism`): a same-seed rerun moves scores by up to about 1 point per test set.
- **Generation**: greedy, vLLM seed 42, `VLLM_BATCH_INVARIANT=1` with FLASH_ATTN (eval.py; it
  prints `vLLM 0.23.0: batch-invariant, FLASH_ATTN attention` and refuses a vLLM that ignores the
  flag). Outputs do not depend on batch size or order; different GPU types can differ slightly.
- **IF scoring**: `eval_if.evaluate` seeds `random` and langdetect.
- **Tokenizers**: eval.py refuses a tokenizer that splits a probe text unlike the model's own
  SentencePiece `tokenizer.model`.

Results from before 2026-11-22 (torch 2.8, vLLM 0.11, DeepSpeed FusedAdam; kept in
`../tmp_results`) are not comparable with new ones: rerun training and evaluation.

## Verification (2026-11-22, H200)

**Eval repeatability**: llama3.1, mistral, olmo2 and qwen2.5 `_git20k` evaluated twice, and once
with `--limit 100`: 100% identical responses and WildGuard labels on all 9 test sets
(`python compare_evals.py RUN_A RUN_B MODEL...`, add `--limit 100` for the short run).

**Training**: llama3.1_git20k / wildguardmix / random / seed 43, trained twice in the old env
(torch 2.8), then in this env with DeepSpeed's FusedAdam and with torch's AdamW; every checkpoint
evaluated in this env. The new runs differ from the old ones as much as the two old runs differ
from each other.

| | torch 2.8 A | torch 2.8 B | FusedAdam | torch AdamW |
|---|---|---|---|---|
| Loss, mean relative difference to torch 2.8 A | – | 0.037% | 0.035% | 0.038% |
| Grad norm, mean relative difference to torch 2.8 A | – | 0.33% | 0.46% | 0.32% |
| Identical IFEval responses with torch 2.8 A | – | 30% | 31% | 31% |
| Harmful, Avg | 9.33 | 9.13 | 9.44 | 9.51 |
| Over-refusal, Avg | 16.34 | 16.15 | 16.04 | 16.23 |
| IF, Average | 36.56 | 36.30 | 36.99 | 37.28 |

```bash
python compare_train_logs.py A=LOG B=LOG ...           # per-step loss and grad norm
python weight_diff.py BASE_MODEL_DIR CKPT...            # how far each checkpoint moved
python compare_evals.py RESULTS_A RESULTS_B CKPT_NAME   # responses, labels, scores
```

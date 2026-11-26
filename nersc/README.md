# nersc: Slurm launchers for NERSC Perlmutter

The same scripts, options and output trees as `../range` (see its README for what each stage
does), adapted to Perlmutter's 4 × A100 GPU nodes. Every script prints its options with `-h`;
resource defaults are in `common.sh` and are overridden through the environment. Every job runs
in the `nlp` env (`../ENVIRONMENT.md`). `diff range/X nersc/X` shows what differs.

| Script | What it does |
| --- | --- |
| `prefetch.sh` | Run once: caches the models and datasets in `$SCRATCH/huggingface` |
| `run.sh` | One seed of stage 1 or 2: (select →) base evals, train array → eval array, diversity, post |
| `train.sh` | Training array (`--eval` adds the eval array, `--eval_only` evaluates saved checkpoints) |
| `eval.sh` | Eval array for any models |
| `gpu.sh` | Any command on `GPUS` A100s (default 1; `GPUS=0`: a CPU job) |
| `post.sh` | Figures and (stage 1) correlations; CPU, runs anywhere |
| `memtest.py` | Does training fit? A few updates with align.py's settings, worst case first |
| `ds_zero3*.json` | DeepSpeed configs: chili's ZeRO-3, and the same with the optimizer on the CPU |
| `*.sbatch` | The batch scripts the launchers submit |

```bash
cd ~/align/sa_data
nersc/prefetch.sh                                                # once, on a login node
nersc/eval.sh llama3.1_git20k llama3.1-ins qwen2.5_git20k ...     # baselines -> results/baselines/
nersc/train.sh --models llama3.1_git20k --abbrs random --seed 43 --eval
nersc/run.sh --seed 43 --dry_run                                 # a whole stage-1 seed
squeue --me; sqs; tail -f ../log/slurm/align_<job>_<task>.log
```

Before the first job:
- `~/align/data` (not in git) must hold RANGE's copy of the hand-copied inputs:
  `git_data_20k.csv`, `subcategory_stats.json` and `AILuminate.csv` (an eval test set). Copy
  them, don't rebuild them. Each seed's training draws go to `~/align/samples/` (`stage1/`,
  `common/`, `stage2_selected/`), written by the first run that needs them; the draws depend
  only on the seed and these inputs, so a checkpoint's fingerprint (`run.json`, which hashes
  the sample files) should match RANGE's for the same run.
- `hf auth login` (the git20k models are private; Llama 3.1, WildGuard, WildGuardMix,
  WildJailbreak and Sorry-Bench are gated). The token stays in `~/.cache/huggingface/token`;
  `common.sh` points `HF_TOKEN_PATH` there and keeps the cache on `$SCRATCH`.

Resources (`common.sh`):
- Allocation `m4705_g` (`ACCOUNT`), nodes `gpu&hbm40g` (`GPU_CONSTRAINT`; `gpu&hbm80g` for the
  80 GB A100s). CPU jobs (post, the stage 2 follow-up) use `m4705` and QOS shared.
- Training: one whole node (QOS regular), 4 × A100-40GB, ZeRO-3, micro-batch 1 × 8
  accumulation = effective batch 32, lr 1e-5, as chili trained (ENVIRONMENT.md: chili's ZeRO-3
  runs agreed with RANGE's ZeRO-2 runs within training noise). On 40 GB GPUs the optimizer runs
  on the CPU (`ds_zero3_offload.json`: chili's `ds_zero3.json` plus `offload_optimizer`, the
  same fused AdamW in fp32). Measured with `memtest.py` (2026-11-24, 4,096-token examples, the
  longest a run sees):

  | config | llama3.1_git20k | qwen2.5_git20k |
  |---|---|---|
  | `ds_zero3.json` | OOM on update 2 (37 of 40 GB before it) | not run |
  | `ds_zero3_offload.json` | 22.6 GB peak; 16.6 s/update long, 11.3 s real | 23.1 GB; 15.5 s, 10.3 s |

  A run (2 epochs of 1,800 examples, 114 updates) trains in ~20–25 minutes. On `gpu&hbm80g`
  nodes `common.sh` uses plain `ds_zero3.json`. ZeRO-2 fits neither (all bf16 weights on every
  GPU).
- Eval: one A100 in QOS shared (32 CPUs, ~60 GB of memory). `EVAL_QOS=debug` / `TRAIN_QOS=debug`
  for a test within 30 minutes.
- Stage 2 selection takes a whole node (it needs ~192 GB of memory).

Notes:
- **Evaluations are A100 evaluations.** vLLM's batch-invariant mode uses Triton matmuls on SM80
  (cuBLAS on Hopper), so a Perlmutter evaluation repeats exactly on Perlmutter (40 and 80 GB
  A100s have the same 108 SMs) but can differ slightly from RANGE's H100/H200 evaluations.
  Compare like with like (`compare_evals.py`), or evaluate Perlmutter checkpoints on RANGE.
- The default `cudatoolkit` module is unloaded: it puts CUDA 13.2's forward-compatibility
  libcuda ahead of the node's driver (580). Nothing is compiled with nvcc; Triton and
  torch.compile build small C helpers with the PE's GCC 14.
- `$SCRATCH` (HF cache, checkpoints, responses, vLLM compile cache) purges files not read for
  8 weeks: rerun `prefetch.sh` if models disappear.
- `common.sh` sets `RESPONSES_ROOT` (eval.py `--responses_root`, `$SCRATCH/results`) and
  `SELECTION_CACHE` (select_data.py `--cache_dir`, `$SCRATCH/selection_cache`), as
  `range/common.sh` does with /scratch/zf28.

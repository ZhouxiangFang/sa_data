# chili: launchers for chili01 (8 × RTX A6000 48 GB, no scheduler)

The chili counterpart of `range/`: same options, settings and output folders, but jobs run
directly on free GPUs instead of through Slurm. Every script prints its options with `-h`;
defaults are in `common.sh` and are overridden through the environment. Every job runs in the
`nlp` conda env (`../ENVIRONMENT.md`; `common.sh` activates it, and each job log starts with the
package versions it ran with). Run long launches in tmux.

| Script | What it does |
| --- | --- |
| `prefetch.sh` | Caches the models, datasets and NLTK data in `~/.cache/huggingface` |
| `run.sh` | One seed of stage 1 or 2: (select →) base evals → train → eval → diversity → post |
| `train.sh` | Training queue on `GPU_GROUPS` (`--eval` then evaluates, `--eval_only` evaluates saved checkpoints) |
| `eval.sh` | `eval.py` on any models, one free GPU each |
| `gpu.sh` | Any one-GPU command, from `sa_data/` |
| `post.sh` | Figures and (stage 1) correlations; CPU |

```bash
cd ~/align/sa_data
chili/prefetch.sh
GPUS=0,1,2,3 chili/eval.sh {llama3.1,mistral,olmo2,qwen2.5}{_git20k,-ins}  # baselines
chili/run.sh --seed 43 --dry_run                       # stage 1: print the plan
chili/run.sh --seed 43                                 # stage 1; rerun to resume
chili/run.sh --seed 43 --selected wildjailbreak        # stage 2 on one dataset
chili/run.sh --seed 43 --selected nemotron --with_unnatural   # stage 2 ablation
tail -f ../log/chili/eval_<model>_<time>.log
```

GPUs: a GPU is free when its lock `/tmp/align_eval_gpu_<i>.lock` is not held and it uses less than
`MAX_USED_MB` (2000), so GPUs used by other people are skipped. One-GPU jobs (eval, selection,
diversity) use `GPUS` (default: all eight, e.g. `GPUS=0,1,2,3`), training jobs `GPU_GROUPS`
(default `"0,1,2,3 4,5,6,7"`, one job per group; a group waits until all its GPUs are free, e.g.
`GPU_GROUPS="0,1,2,3"` when others use 4–7). Hold a lock to keep the launchers off a GPU:
`tmux new -d -s gpu-hold 'exec 9>/tmp/align_eval_gpu_4.lock; flock 9; sleep infinity'`.

Training differs from range only where the A6000s need it: ZeRO-3 (`ds_zero3.json`; ZeRO-2 does
not fit an 8B model's optimizer states in 48 GB on 4 GPUs) and micro-batch 1 × 8 accumulation on 4
GPUs (effective batch 32; micro-batches of 2 and 4 ran out of memory on 4096-token examples).
Learning rate, epochs, schedule, optimizer (torch's fused AdamW) and data are the same, so
checkpoints carry the same fingerprint as range's, but the numerics differ and evaluation on
A6000s can differ slightly from Hopper GPUs: compare chili and range results only with that in
mind.

Stage 2 selection (`select_data.py`) and the stage 1 diversity scores POS-tag with spaCy
`en_core_web_trf` on the GPU (cupy); tags and embeddings are cached in
`$SELECTION_CACHE/<tagger>/<dataset>/` (default `/scratch/zf28/selection_cache`). If the selection
fails, `run.sh` stops before training; rerun the same command to resume.

Outputs (paths from `~/align`, as on range):
- `results/`: scores only, one CSV per evaluated model (every metric with an `Avg` row).
  - `baselines/<model>.csv`: the official instruct and git20k starting models
  - `correlations/<setting>/` (stage 1) and
    `selection/<dataset>[_refusal][_with_unnatural][_mixed]/` (stage 2)
- `$RESPONSES_ROOT` (default `/scratch/zf28/results/`, chili's local `/scratch`, not range's;
  passed as `eval.py --responses_root`): the same tree, the responses and per-subcategory
  breakdowns.
- `figs/`: the same tree, the figures.
- `log/chili/`: job logs and training plans (range's are in `log/slurm/`).
- Checkpoints: `$CKPT_ROOT/seed<S>/<setting>/` (default `/scratch/zf28/ckpts`; kept unless
  `DELETE_CKPTS=1`).
- Training data: `samples/stage1/`, `samples/common/`, `samples/stage2_selected/` (written by the
  runs); `data/` holds only the inputs copied by hand (`git_data_20k.csv`, `AILuminate.csv`,
  `subcategory_stats.json`).

Results from the old layout and env (before 2026-11-22) are in `~/align/tmp_results/`.

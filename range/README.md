# range: Slurm launchers for the RANGE cluster

The same workflow as `chili/`, submitted as Slurm jobs instead of run on free local GPUs.
Every script prints its options with `-h`; resource defaults live in `common.sh` and are
overridden through the environment.

| Script | What it does |
| --- | --- |
| `prefetch.sh` | Run once on the login node: caches the models and datasets in `~/.cache/huggingface` |
| `run.sh` | A whole seed: train array → eval array (task i after training task i) + diversity → plot/correlate |
| `train.sh` | Training array (`--eval` adds the eval array, `--eval_only` evaluates saved checkpoints) |
| `eval.sh` | Eval array for any models |
| `gpu.sh` | Any one-GPU command (`GPUS=0`: CPU only) |
| `post.sh` | Plot + correlate stages (CPU; runs anywhere) |
| `*.sbatch` | The batch scripts the launchers submit |

```bash
cd ~/align/sa_data
range/prefetch.sh
range/run.sh --seed 43 --dry_run        # print the plan and sbatch commands
range/run.sh --seed 43                  # submit; rerun later to resume or retry failures
squeue --me; tail -f ../log/slurm/align_<job>_<task>.log
```

Defaults: training uses 2 × H200 with ZeRO-3, micro-batch 1 × 16 accumulation steps, which
gives chili's effective batch of 32 and the same examples in every update. Eval uses one H200 or
H100, whichever frees first (`EVAL_GPU=hopper`; or `h100`, `h200`, `47gb`). At most
`MAX_TRAIN=4` training and `MAX_EVAL=4` eval jobs run at once. Logs are in `log/slurm/`.

Notes:
- The first training job JIT-compiles DeepSpeed's FusedAdam into `~/.cache/torch_extensions`.
  That build needs `nvcc` from `/opt/apps/software/CUDA/12.8.0` and the system GCC 11; GCC 14
  produces a library the conda libstdc++ can't load. `common.sh` sets both.
- `/scratch` is purged: the old `/scratch/zf28/huggingface` cache has lost its files. Keep
  the HF cache on /home. Checkpoints on /scratch are deleted once they are evaluated.

# range: Slurm launchers for the RANGE cluster

The same workflow as `chili/`, submitted as Slurm jobs instead of run on free local GPUs.
Every script prints its options with `-h`; resource defaults live in `common.sh` and are
overridden through the environment.

| Script | What it does |
| --- | --- |
| `prefetch.sh` | Run once on the login node: caches the models and datasets in `/scratch/zf28/huggingface` |
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

Defaults: training uses 2 × H200 with ZeRO-2 (`range/ds_zero2.json`), micro-batch 4 × 4 accumulation
steps (effective batch 32, as on chili) and learning rate 1e-5. Change them with `EFFECTIVE_BATCH`,
`MICRO_BATCH`, `LEARNING_RATE` and `DS_CONFIG`. The lr comes from a Nov 2026 lr × batch ablation
(`ablations/lrbs`): compared with 5e-6, it left harmful scores unchanged, cut over-refusal by 3.4 points
and cost 2 IFEval points. At the same batch and lr, RANGE and chili agree to within training noise, so keep
both clusters on one setting. Chili uses ZeRO-3 (`sa_data/ds_config.json`). ZeRO-2 needs DeepSpeed ≥ 0.19.6,
which nlp has since Nov 9, 2026. With gradient accumulation, 0.17.6 to 0.19.5 compute wrong gradients:
the seed 43/86/129 runs before Nov 9, on 0.18.0, came out bit-identical to the base models. `train.sh`
refuses ZeRO-2 on those versions unless `ALLOW_ZERO2=1`. Eval uses one H200 or
H100, whichever frees first (`EVAL_GPU=hopper`; or `h100`, `h200`, `47gb`). At most
`MAX_TRAIN=8` training and `MAX_EVAL=16` eval jobs run at once. Logs are in `log/slurm/`.

Notes:
- The first training job JIT-compiles DeepSpeed's FusedAdam into `~/.cache/torch_extensions`.
  It is built with GCC 14 from `/opt/apps` (compute nodes have no system g++) and `nvcc` from
  `/opt/apps/software/CUDA/12.8.0`. The env's libstdc++ was upgraded so GCC 14 builds load. `common.sh` sets both.
- /home has a 50 GB quota, so the whole HF cache (`HF_HOME`, including the token) is in
  `/scratch/zf28/huggingface`, and `~/.cache/huggingface` is a symlink to it. /scratch purges
  old files: an earlier cache there lost all of them. If models disappear, rerun
  `prefetch.sh`. If the token disappears, run `hf auth login` again.
  Checkpoints on /scratch are deleted once they are evaluated.

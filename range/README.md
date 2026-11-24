# range: Slurm launchers for the RANGE cluster

Every script prints its options with `-h`; resource defaults are in `common.sh` and are
overridden through the environment. Every job runs in the `nlp` env (`../ENVIRONMENT.md`).

| Script | What it does |
| --- | --- |
| `prefetch.sh` | Run once: caches the models and datasets in `/scratch/zf28/huggingface` |
| `run.sh` | One seed of stage 1 or 2: (select →) base evals, train array → eval array, diversity, post |
| `train.sh` | Training array (`--eval` adds the eval array, `--eval_only` evaluates saved checkpoints) |
| `eval.sh` | Eval array for any models |
| `gpu.sh` | Any one-GPU command (`GPUS=0`: CPU only) |
| `post.sh` | Figures and (stage 1) correlations; CPU, runs anywhere |
| `*.sbatch` | The batch scripts the launchers submit |

```bash
cd ~/align/sa_data
range/prefetch.sh
range/run.sh --seed 43 --dry_run                       # stage 1: print the plan
range/run.sh --seed 43                                 # stage 1: submit; rerun to resume
range/run.sh --seed 43 --selected wildjailbreak        # stage 2 on one dataset
range/run.sh --seed 43 --selected wildguardmix --response_label_filter refusal
squeue --me; tail -f ../log/slurm/align_<job>_<task>.log
```

**Stage 1** trains the four git20k models on each WildGuardMix subcategory with at least 600
eligible rows and on the random / evenly baselines (1,800 examples: a third harmful, a third
WildJailbreak vanilla benign, a third `data/git_data_20k.csv`), evaluates them (`eval.py`: harm on
five test sets, XSTest-safe + OR-Bench-Hard-1k overrefusal, IFEval + IFBench prompt-loose), scores
the diversity of each harmful sample (`calculate_diversity.py`: 12 metrics, two embedders) and
correlates it with the harmful score (`correlate_diversity.py`). Results: `results/`,
`diversity/`, figures in `results/figs/` and `diversity/summary/`.

**Stage 2** (`--selected DATASET`) first selects 600-row subsets from WildGuardMix, WildJailbreak,
ALERT-DPO or Nemotron-SFT-Safety-v2 with the 12 metrics and six baselines (`select_data.py`).
Vanilla and adversarial rows are never mixed: each subset keeps their natural shares, and each
stratum is selected on its own (a metric's greedy step draws a fresh pool of 5,000 unselected rows
of the stratum and scores the metric on that stratum's selected rows only). A CPU follow-up job then reruns
`run.sh ... --skip select` to train and evaluate every subset. Subsets:
`data/selected/seed<seed>/<dataset>[_refusal]/`, results: `ablations/selected_<dataset>[_refusal]/`.
Tags and embeddings are cached in `/scratch/zf28/selection_cache/<dataset>/`.

Defaults: training on 2 × H200 with ZeRO-2 (`ds_zero2.json`, DeepSpeed ≥ 0.19.6: 0.17.6–0.19.5
compute wrong ZeRO-2 gradients with accumulation, and `train.sh` refuses them), micro-batch 4 × 4
accumulation (effective batch 32), lr 1e-5 (an lr × batch ablation: vs 5e-6 it left harm unchanged,
cut over-refusal 3.4 points and cost 2 IFEval points). Eval on one H200 or H100
(`EVAL_GPU=hopper`). At most `MAX_TRAIN=8` training and `MAX_EVAL=16` eval jobs run at once.
Checkpoints are kept after evaluation (`DELETE_CKPTS=1` deletes each once it is evaluated). `--response_label_filter
refusal` and `--num_train N` make their own setting (`setting_setup` in `common.sh`). Logs:
`log/slurm/`.

Notes:
- Nothing is compiled with nvcc (torch's fused AdamW). Triton and torch.compile build small C
  helpers with GCC 14 from `/opt/apps` (compute nodes have no system g++); `common.sh` sets
  `CC`/`CXX`.
- /home has a 50 GB quota, so the HF cache (`HF_HOME`, token included) is on /scratch, which
  purges old files: rerun `prefetch.sh` if models disappear, `hf auth login` if the token does.
- The stage 2 follow-up job runs `run.sh` on a compute node, so it needs `sbatch` there; if it
  fails, rerun the same command with `--skip select`.

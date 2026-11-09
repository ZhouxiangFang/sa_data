"""Compare training configs submitted with range/train.sh --eval: speed, safety, over-refusal and
instruction following per config.

    python range/compare_runs.py NAME=TRAIN_ARRAY_ID ... [--results_root DIR] [--seed 43] [--csv OUT]

NAME is the config's folder under --results_root (its train.sh --results_dir); TRAIN_ARRAY_ID is
the training array whose plan (log/slurm/jobs_*.tsv, found from the array's logs) lists the jobs.
Prints one table each for speed, harmful score, over-refusal and instruction following (means over
a config's evaluated jobs; "change" = trained minus its own base model, averaged), with a base row,
then harmful / over-refusal / IFEval per training set and per model. All scores are percentages.
Base scores come from results/<model>/ and results/if_eval/.
"""
import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "log" / "slurm"
HARMFUL = ["Avg", "wildguardmix", "aegis", "wildjailbreak", "ailuminate", "gretel"]
OVERREFUSAL = ["Avg", "xstest", "or-bench-hard"]
IF_BENCHMARKS = {"IFEval": "ifeval", "IFBench": "ifbench"}
IF_METRICS = ["prompt_strict", "instruction_strict", "prompt_loose", "instruction_loose"]


def read_scores(path, column, prefix, names):
    values = dict.fromkeys((f"{prefix}_{name}" for name in names), float("nan"))
    if path.exists():
        frame = pd.read_csv(path)
        for dataset, value in zip(frame["dataset"], frame[column]):
            if dataset in names:
                values[f"{prefix}_{dataset}"] = float(value)
    return values


def read_if(path):
    values = {f"{short}_{metric}": float("nan") for short in IF_BENCHMARKS.values() for metric in IF_METRICS}
    if path.exists():
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            short = IF_BENCHMARKS.get(row["benchmark"])
            if short:
                for metric in IF_METRICS:
                    values[f"{short}_{metric}"] = 100 * float(row[metric])
    return values


def scores(harmful_csv, overrefusal_csv, if_csv):
    return {**read_scores(harmful_csv, "harmful_score_pct", "harmful", HARMFUL),
            **read_scores(overrefusal_csv, "overrefusal_score_pct", "overrefusal", OVERREFUSAL),
            **read_if(if_csv)}


def plan_of(array_id):
    """The jobs_*.tsv a training array was submitted with (named in its task logs or sacct)."""
    for log in sorted(LOG_DIR.glob(f"align_{array_id}_*.log")):
        match = re.search(r"(\S*jobs_\d+_\d+\.tsv)", log.read_text(errors="ignore")[:20000])
        if match:
            return Path(match.group(1))
    out = subprocess.run(["sacct", "-j", str(array_id), "-n", "-X", "-P", "-o", "SubmitLine%2000"],
                         capture_output=True, text=True).stdout
    match = re.search(r"(\S*jobs_\d+_\d+\.tsv)", out)
    return Path(match.group(1)) if match else None


def elapsed_minutes(array_id):
    out = subprocess.run(["sacct", "-j", str(array_id), "-n", "-X", "-P", "-o", "JobID,State,ElapsedRaw"],
                         capture_output=True, text=True).stdout
    minutes = {}
    for line in out.splitlines():
        job, state, seconds = line.split("|")
        if "_" in job and state == "COMPLETED" and seconds.isdigit():
            minutes[int(job.split("_")[1])] = int(seconds) / 60
    return minutes


def train_runtime_minutes(array_id, task):
    log = LOG_DIR / f"align_{array_id}_{task}.log"
    if not log.exists():
        return float("nan")
    match = re.findall(r"'train_runtime': ([0-9.]+)", log.read_text(errors="ignore"))
    return float(match[-1]) / 60 if match else float("nan")


def table(frame, order, columns, change_columns, title, prefix=""):
    """Means per config of `columns` (+ change columns), plus a base row over the same jobs;
    `prefix` is dropped from the printed column names."""
    body = frame.groupby("config", sort=False)[columns + [f"{c}_change" for c in change_columns]].mean()
    body = body.reindex(order)
    body.loc["base"] = frame[[f"base_{c}" for c in columns]].mean().set_axis(columns)
    if prefix:
        body.columns = [c.removeprefix(prefix) for c in body.columns]
    print(f"\n{title}")
    print(body.round(2).to_string())


def pivot(frame, order, value, index_columns, base_label=True):
    out = frame.pivot_table(index="config", columns=index_columns, values=value, aggfunc="mean").reindex(order)
    if base_label:
        out.loc["base"] = frame.pivot_table(columns=index_columns, values=f"base_{value}", aggfunc="mean").iloc[0]
    return out.round(2).to_string()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="NAME=TRAIN_ARRAY_ID")
    parser.add_argument("--results_root", type=Path, default=ROOT / "ablations" / "lrbs")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    base, rows = {}, []
    for spec in args.configs:
        name, array_id = spec.split("=")
        plan = plan_of(array_id)
        if plan is None or not plan.exists():
            print(f"{name}: no plan found for array {array_id}")
            continue
        jobs = pd.read_csv(plan, sep="\t", header=None).iloc[:, :3]
        jobs.columns = ["model", "dataset", "abbr"]
        elapsed = elapsed_minutes(array_id)
        for task, job in jobs.iterrows():
            model, dataset, abbr = job["model"], job["dataset"], job["abbr"]
            if model not in base:
                base[model] = scores(ROOT / "results" / model / f"{model}_overall.csv",
                                     ROOT / "results" / model / f"{model}_overrefusal.csv",
                                     ROOT / "results" / "if_eval" / f"{model}_if_summary.csv")
            run_dir = args.results_root / name / f"seed{args.seed}"
            run = f"{model}_1800_{dataset}"
            group = f"{dataset}_{abbr}"
            trained = scores(run_dir / run / group / f"{group}_overall.csv",
                             run_dir / run / group / f"{group}_overrefusal.csv",
                             run_dir / "if_eval" / run / f"{group}_if_summary.csv")
            rows.append(dict(config=name, model=model, dataset=dataset, abbr=abbr,
                             speed_train_min=train_runtime_minutes(array_id, task),
                             speed_job_min=elapsed.get(task, float("nan")),
                             **trained,
                             **{f"base_{k}": v for k, v in base[model].items()},
                             **{f"{k}_change": trained[k] - v for k, v in base[model].items()}))
    frame = pd.DataFrame(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
    order = list(dict.fromkeys(frame["config"]))
    pd.set_option("display.width", 250, "display.max_columns", 40)

    counts = frame.groupby("config", sort=False).agg(
        trained=("speed_train_min", "count"), evaluated=("harmful_Avg", "count"),
        train_min=("speed_train_min", "mean"), job_min=("speed_job_min", "mean")).reindex(order)
    print("Speed (minutes per job; train = train_runtime, job = Slurm elapsed incl. load and save)")
    print(counts.round(2).to_string())

    done = frame.dropna(subset=["harmful_Avg"])
    table(done, order, [f"harmful_{t}" for t in HARMFUL], ["harmful_Avg"],
          "Harmful score % (lower is safer)", prefix="harmful_")
    table(done, order, [f"overrefusal_{t}" for t in OVERREFUSAL], ["overrefusal_Avg"],
          "Over-refusal % on benign prompts (lower is better)", prefix="overrefusal_")
    if_columns = [f"{b}_{m}" for b in IF_BENCHMARKS.values() for m in ("prompt_strict", "instruction_strict")]
    table(done, order, if_columns, ["ifeval_prompt_strict", "ifbench_prompt_strict"],
          "Instruction following % (higher is better)")

    for value, label in [("harmful_Avg", "Harmful Avg"), ("overrefusal_Avg", "Over-refusal Avg"),
                         ("ifeval_prompt_strict", "IFEval prompt strict")]:
        print(f"\n{label} by training set (mean over models)")
        print(pivot(done, order, value, ["dataset", "abbr"]))
        print(f"\n{label} by model (mean over training sets)")
        print(pivot(done, order, value, "model"))


if __name__ == "__main__":
    main()

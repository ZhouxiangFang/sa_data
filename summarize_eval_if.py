#!/usr/bin/env python3
"""Summarize IFBench and IFEval scores before and after safety fine-tuning.

The post-training score for each model is the arithmetic mean across the
available subcategory summary files for the selected dataset and training size.
The output uses three header rows to represent metric -> benchmark -> training stage.

Example:
    python summarize_eval_if.py
    python summarize_eval_if.py --dataset gretel --num-train 1800
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import fmean


BENCHMARKS = ("IFBench", "IFEval")
METRICS = (
    "prompt_loose",
    "instruction_loose",
    "prompt_strict",
    "instruction_strict",
)
DEFAULT_MODELS = (
    "llama3.1_git20k",
    "llama3_git20k",
    "mistral_git20k",
    "olmo2_git20k",
    "qwen2.5_git20k",
    "qwen3_git20k",
)
STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "subcategory_stats.json"


def expected_subcategories(dataset):
    with STATS_PATH.open() as file:
        stats = json.load(file)
    return {entry["abbr"] for entry in stats[dataset].values()}


def load_summary(path):
    """Load and validate the two benchmark rows in one summary CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"benchmark", *METRICS}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path}: missing columns: {missing}")

        rows = {}
        for row in reader:
            benchmark = row["benchmark"]
            if benchmark not in BENCHMARKS:
                continue
            if benchmark in rows:
                raise ValueError(f"{path}: duplicate {benchmark} row")
            rows[benchmark] = {
                metric: float(row[metric]) for metric in METRICS
            }

    missing_benchmarks = set(BENCHMARKS) - rows.keys()
    if missing_benchmarks:
        missing = ", ".join(sorted(missing_benchmarks))
        raise ValueError(f"{path}: missing benchmark rows: {missing}")
    return rows


def find_after_summaries(input_dir, model, dataset="wildguardmix", num_train=800):
    """Return one post-training summary path per safety subcategory."""
    summaries = {}
    # Current eval_if.py groups summaries by model, training size and dataset.
    # Retain support for older flat checkpoint filenames.
    layouts = [
        (input_dir / f"{model}_{num_train}_{dataset}", f"{dataset}_", "_if_summary.csv"),
        (input_dir, f"{model}_{dataset}_", f"_{num_train}_if_summary.csv"),
        (input_dir, f"home-zf28-models-20k-{model}_{dataset}_", f"_{num_train}_if_summary.csv"),
        (input_dir, f"{model}_{num_train}_{dataset}_", "_if_summary.csv"),
    ]
    for directory, prefix, suffix in layouts:
        for path in directory.glob(f"{prefix}*{suffix}"):
            subcategory = path.name[len(prefix) : -len(suffix)]
            if subcategory in summaries:
                raise ValueError(
                    f"Duplicate {model}/{subcategory} summaries: "
                    f"{summaries[subcategory]} and {path}"
                )
            summaries[subcategory] = path

    if not summaries:
        raise FileNotFoundError(
            f"No post-training {dataset}/{num_train} summary files found for {model} in {input_dir}"
        )
    return summaries


def column_keys():
    """Return metric -> benchmark -> stage column keys."""
    return [
        (metric, benchmark, stage)
        for metric in METRICS
        for benchmark in BENCHMARKS
        for stage in ("before", "after")
    ]


def summarize_model(input_dir, model, dataset="wildguardmix", num_train=800):
    """Build one comparison row and return its included subcategories."""
    before_path = input_dir / f"{model}_if_summary.csv"
    if not before_path.is_file():
        raise FileNotFoundError(f"Missing baseline summary: {before_path}")

    before = load_summary(before_path)
    after_paths = find_after_summaries(input_dir, model, dataset, num_train)
    after = [load_summary(path) for path in after_paths.values()]

    row = {"model": model}
    for metric in METRICS:
        for benchmark in BENCHMARKS:
            row[(metric, benchmark, "before")] = before[benchmark][metric]
            row[(metric, benchmark, "after")] = fmean(
                summary[benchmark][metric] for summary in after
            )
    return row, set(after_paths)


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    default_input = project_dir / "results" / "if_eval"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing *_if_summary.csv files (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: INPUT_DIR/if_scores_before_after_DATASET_SIZE.csv; legacy name for wildguardmix/800)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Baseline model stems to include, in row order",
    )
    parser.add_argument("--dataset", choices=("wildguardmix", "aegis", "gretel"),
                        default="wildguardmix", help="Safety training dataset")
    parser.add_argument("--num-train", "--num_train", type=int, default=800,
                        help="Training examples per checkpoint (default: 800)")
    args = parser.parse_args()
    if args.num_train <= 0:
        parser.error("--num-train must be positive")
    return args


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    filename = ("if_scores_before_after.csv"
                if (args.dataset, args.num_train) == ("wildguardmix", 800)
                else f"if_scores_before_after_{args.dataset}_{args.num_train}.csv")
    output = args.output or input_dir / filename
    output = output.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    rows = []
    expected = expected_subcategories(args.dataset)
    for model in args.models:
        try:
            row, subcategories = summarize_model(input_dir, model, args.dataset, args.num_train)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from error
        rows.append(row)

        missing = expected - subcategories
        extra = subcategories - expected
        status = f"{model}: averaged {len(subcategories)} subcategories"
        if missing:
            status += f"; missing {', '.join(sorted(missing))}"
        if extra:
            status += f"; unexpected {', '.join(sorted(extra))}"
        print(status, file=sys.stderr)

    columns = column_keys()
    average_row = {"model": "avg"}
    for column in columns:
        average_row[column] = fmean(row[column] for row in rows)
    rows.append(average_row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", *(metric for metric, _, _ in columns)])
        writer.writerow(
            ["benchmark", *(benchmark for _, benchmark, _ in columns)]
        )
        writer.writerow(["stage", *(stage for _, _, stage in columns)])
        writer.writerows(
            [row["model"], *(f"{row[column]:.3f}" for column in columns)]
            for row in rows
        )

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

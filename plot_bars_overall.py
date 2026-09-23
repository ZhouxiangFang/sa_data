"""Plot harmful and prompt-loose IF scores for six models and their mean.

Run from any directory: python sa_data/plot_bars_overall.py
Figures default to figs/<baseline>_<size> (for example, figs/git20k_800),
inferred from the loaded runs. Use --output-dir to choose an explicit folder.
Harmful scores use the precomputed Avg row (a macro-average over safety
datasets); IFBench and IFEval prompt_loose fractions are converted to percent.
IF scores are averaged equally across IFBench and IFEval. Official and
self-trained instruct baselines come first, followed by subcategories ranked
by descending harmful score. Pale gray bars beside the harmful bars show the
secondary IF metric. The average figure gives each model equal weight.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("llama3", "llama3.1", "mistral", "olmo2", "qwen2.5", "qwen3")
METRICS = {
    "harmful_score_pct": ("Overall harmful ↓", "#D55E00"),
    "if_prompt_loose_pct": ("Mean IFBench / IFEval prompt_loose ↑", "#999999"),
}


def discover_runs(results_dir, model):
    """Map shared x-axis version labels to actual result directory names."""
    runs = {"-ins": f"{model}-ins"}
    # Existing files use _git20k; also accept the user's -20k convention.
    candidates = [f"{model}{suffix}" for suffix in ("_git20k", "-20k", "-git20k")]
    candidates = [name for name in candidates if (results_dir / name).is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"{model}: expected one 20k baseline, found {candidates}")
    runs["-20k"] = candidates[0]
    prefix = candidates[0] + "_"
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            runs[path.name[len(prefix):]] = path.name
    if len(runs) == 2:
        raise ValueError(f"{model}: no subcategory runs found under {results_dir}")
    return runs


def read_score(path, row_column, row_name, score_column, scale=1.0):
    data = pd.read_csv(path)
    rows = data.loc[data[row_column] == row_name, score_column]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one {row_name!r} row")
    score = float(rows.iloc[0]) * scale
    if not np.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f"{path}: invalid {score_column} percentage: {score}")
    return score


def load_scores(results_dir, if_eval_dir):
    rows = []
    versions = None
    for model in MODELS:
        runs = discover_runs(results_dir, model)
        if versions is None:
            versions = list(runs)
        elif set(runs) != set(versions):
            raise ValueError(
                f"{model}: versions differ from {MODELS[0]}; "
                f"missing={sorted(set(versions) - set(runs))}, "
                f"extra={sorted(set(runs) - set(versions))}"
            )
        for version, run in runs.items():
            overall = results_dir / run / f"{run}_overall.csv"
            summary = if_eval_dir / f"{run}_if_summary.csv"
            rows.append({
                "model": model,
                "version": version,
                "run": run,
                "harmful_score_pct": read_score(
                    overall, "dataset", "Avg", "harmful_score_pct"
                ),
                "ifbench_prompt_loose_pct": read_score(
                    summary, "benchmark", "IFBench", "prompt_loose", 100.0
                ),
                "ifeval_prompt_loose_pct": read_score(
                    summary, "benchmark", "IFEval", "prompt_loose", 100.0
                ),
            })
    data = pd.DataFrame(rows)
    data["if_prompt_loose_pct"] = data[
        ["ifbench_prompt_loose_pct", "ifeval_prompt_loose_pct"]
    ].mean(axis=1)
    return data, versions


def default_output_dir(data, versions):
    """Infer the figure folder from baseline names and safety training size."""
    baselines = {
        row.run[len(row.model):].lstrip("_-")
        for row in data.loc[data["version"] == "-20k"].itertuples()
    }
    sizes = {
        int(version.rsplit("_", 1)[1])
        for version in versions if version not in ("-ins", "-20k")
    }
    if len(baselines) != 1 or len(sizes) != 1:
        raise ValueError(
            "Cannot infer one output folder from mixed baselines or training "
            "sizes; specify --output-dir"
        )
    return ROOT / "figs" / f"{next(iter(baselines))}_{next(iter(sizes))}"


def plot_bars(scores, versions, title, output_path):
    baselines = ["-ins", "-20k"]
    subcategories = scores.reindex([
        version for version in versions if version not in baselines
    ]).sort_values(
        "harmful_score_pct", ascending=False, kind="stable"
    )
    versions = baselines + subcategories.index.tolist()
    scores = scores.reindex(versions)
    x = np.arange(len(versions), dtype=float)
    x[len(baselines):] += 0.6
    width = 0.34
    offset = 0.19
    fig, ax = plt.subplots(figsize=(max(12, len(versions) * 0.85), 7))
    label, color = METRICS["if_prompt_loose_pct"]
    if_bars = ax.bar(
        x + offset, scores["if_prompt_loose_pct"], width, label=label, color=color,
        alpha=0.25, edgecolor="none", zorder=2,
    )
    label, color = METRICS["harmful_score_pct"]
    bars = ax.bar(
        x - offset, scores["harmful_score_pct"], width, label=label, color=color,
        zorder=3,
    )
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.bar_label(if_bars, fmt="%.1f", padding=4, fontsize=7, color="0.5")

    labels = []
    training_sets = set()
    for version in versions:
        if version == "-ins":
            labels.append("Official instruct")
        elif version == "-20k":
            labels.append("Self-trained instruct")
        else:
            dataset, subcategory_and_size = version.split("_", 1)
            subcategory, size = subcategory_and_size.rsplit("_", 1)
            labels.append(subcategory)
            training_sets.add((dataset, int(size)))
    training_info = "; ".join(
        f"{dataset}, {size:,} examples per subcategory"
        for dataset, size in sorted(training_sets)
    )
    ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=14)
    ax.axvline((x[1] + x[2]) / 2, color="0.65", linestyle="--", linewidth=1)
    ymax = min(105, np.ceil(scores[list(METRICS)].to_numpy().max() / 10) * 10 + 10)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, min(ymax, 100) + 1, 10 if ymax <= 70 else 20))
    ax.set_ylabel("Score (%)", fontsize=18)
    ax.set_xlabel(
        f"Model version / training subcategory ({training_info})\n"
        "Self-trained instruct: 20k training examples",
        fontsize=16, labelpad=12,
    )
    ax.set_title(title, pad=48, fontsize=23)
    ax.legend(
        handles=[bars, if_bars], loc="lower center",
        bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=13,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--if-eval-dir", type=Path,
        help="IF summary directory (default: RESULTS_DIR/if_eval)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output folder (default: ROOT/figs/<baseline>_<size>, inferred from runs)",
    )
    args = parser.parse_args()
    try:
        data, versions = load_scores(
            args.results_dir, args.if_eval_dir or args.results_dir / "if_eval"
        )
        output_dir = args.output_dir or default_output_dir(data, versions)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        scores = data.loc[data["model"] == model].set_index("version")
        plot_bars(scores, versions, model, output_dir / f"{model}_overall.png")

    score_columns = [column for column in data if column.endswith("_pct")]
    average = data.groupby("version")[score_columns].mean().reindex(versions)
    plot_bars(
        average, versions, f"Average across {len(MODELS)} models (equal weight)",
        output_dir / "average_overall.png",
    )
    # Export the exact plotted numbers, including the six-model means.
    average = average.reset_index().assign(model="Average", run="")
    table_path = output_dir / "overall_scores.csv"
    pd.concat([data, average], ignore_index=True).to_csv(
        table_path, index=False, float_format="%.2f"
    )
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()

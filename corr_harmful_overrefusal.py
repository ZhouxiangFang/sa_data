"""Correlate harmful and overrefusal rates across training runs.

Example (run from any directory):
    python sa_data/corr_harmful_overrefusal.py --datasets aegis wildguardmix gretel

For each training dataset, the points are the runs in
results/<model>_git20k_<size>_<dataset>/ (every subcategory except benign, plus
the random and evenly baselines; instruct baselines are left out). Overrefusal is the Avg row
of <run>_overrefusal.csv; scores are read with plot_bars_overall.load_scores, so
they match the plotted bars. These harmful rates are correlated with it:
    overall  the Avg row, a macro-average over the five safety test datasets
    <test>   each safety test dataset; the one matching the training dataset
             is the "same" rate
    other    the equal-weight mean of the four test datasets other than the
             training dataset
Figures use "Average", the correlation of the equally weighted model-mean
scores, as in plot_bars_diversity.py. The CSVs also keep each model's
coefficients and "Mean coefficient", the mean of the per-model coefficients.
Outputs go to figs/harmful_overrefusal/ (or --output-dir): per training dataset,
<dataset>/harmful_vs_overrefusal.png (one scatter panel per harmful rate),
<dataset>/correlations.png (Pearson and Spearman bars per harmful rate, * marks
p < 0.05) and <dataset>/correlations.csv; and a combined correlations.csv.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from plot_bars_overall import (DEFAULT_MODELS, MODELS, ROOT, TEST_DATASETS, load_scores,
                               training_subcategory)


CORRELATIONS = {"pearson": pearsonr, "spearman": spearmanr}
HARMFUL_RATES = ("overall", *TEST_DATASETS, "other")
# The first two slots of the categorical palette used by plot_bars_diversity.py.
METHOD_COLORS = ("#2a78d6", "#eb6834")
SCATTER_COLOR = "#2a78d6"


def harmful_rates(data, dataset):
    """Add one column per harmful rate, named as in HARMFUL_RATES."""
    others = [f"{test}_harmful_score_pct" for test in TEST_DATASETS if test != dataset]
    return data.assign(
        overall=data["harmful_score_pct"],
        other=data[others].mean(axis=1),
        **{test: data[f"{test}_harmful_score_pct"] for test in TEST_DATASETS},
    )


def rate_label(rate, dataset):
    name = TEST_DATASETS[dataset]
    if rate == "overall":
        return "Overall harmful (5 test sets)"
    if rate == "other":
        return f"Other 4 test sets harmful (excl. {name})"
    if rate == dataset:
        return f"{name} harmful (same as training)"
    return f"{TEST_DATASETS[rate]} harmful"


def format_p(p_value):
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def plot_scatter(scores, table, dataset, models, path):
    """One panel per harmful rate: model-mean harmful vs. overrefusal per training run."""
    columns = 4
    rows = -(-len(HARMFUL_RATES) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.4 * rows),
                             layout="constrained", squeeze=False)
    x = scores["overrefusal_score_pct"].to_numpy()
    for ax, rate in zip(axes.ravel(), HARMFUL_RATES):
        y = scores[rate].to_numpy()
        ax.scatter(x, y, s=48, color=SCATTER_COLOR, edgecolors="white", linewidths=1.5,
                   zorder=3)
        for xi, yi, version in zip(x, y, scores.index):
            ax.annotate(training_subcategory(version), (xi, yi), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="0.25")
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, np.polyval(np.polyfit(x, y, 1), line_x), color=SCATTER_COLOR,
                alpha=0.65, linewidth=2)
        stats = table.loc[table["harmful_rate"] == rate].set_index("method")
        ax.text(0.97, 0.97, "\n".join(
            f"{method.title()}: {stats.at[method, 'coefficient']:+.3f} "
            f"({format_p(stats.at[method, 'p_value'])})" for method in CORRELATIONS
        ), transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_title(rate_label(rate, dataset), fontsize=12,
                     fontweight="bold" if rate == dataset else "normal")
        ax.set_xlabel("Overrefusal (%) ↓", fontsize=10)
        ax.set_ylabel("Harmful score (%) ↓", fontsize=10)
        ax.margins(x=0.12, y=0.22)
        style_axis(ax)
    for ax in axes.ravel()[len(HARMFUL_RATES):]:
        ax.set_visible(False)
    fig.suptitle(f"{dataset}: harmful vs. overrefusal across {len(scores)} training runs, "
                 f"averaged over {len(models)} models ({', '.join(models)})\n"
                 "One point per training subcategory; overrefusal = mean of "
                 "XSTest / OR-Bench-Hard", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def plot_correlations(table, dataset, models, path):
    """Pearson and Spearman bars per harmful rate; * marks p < 0.05."""
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    width = 0.36
    x = np.arange(len(HARMFUL_RATES), dtype=float)
    for index, (method, color) in enumerate(zip(CORRELATIONS, METHOD_COLORS)):
        rows = table.loc[table["method"] == method].set_index("harmful_rate")
        rows = rows.reindex(HARMFUL_RATES)
        positions = x + (index - 0.5) * width
        ax.bar(positions, rows["coefficient"], width * 0.92, color=color,
               label=method.title(), zorder=3)
        for position, row in zip(positions, rows.itertuples()):
            star = "*" if row.p_value < 0.05 else ""
            ax.annotate(f"{row.coefficient:.2f}{star}", (position, row.coefficient),
                        xytext=(0, -4), textcoords="offset points", ha="center",
                        va="top", fontsize=10, color="0.2")
    ax.axhline(0, color="0.3", linewidth=1)
    # Separate the overall rate and the other-tests mean from the single test datasets.
    for boundary in (0.5, len(HARMFUL_RATES) - 1.5):
        ax.axvline(boundary, color="0.65", linestyle="--", linewidth=1)
    labels = [rate_label(rate, dataset).replace(" harmful", "\nharmful", 1)
              for rate in HARMFUL_RATES]
    ax.set_xticks(x, labels, fontsize=10)
    for label, rate in zip(ax.get_xticklabels(), HARMFUL_RATES):
        if rate == dataset:
            label.set_fontweight("bold")
    ax.set_ylim(-1.1, 0.1 if (table["coefficient"] <= 0).all() else 1.1)
    ax.set_ylabel("Coefficient: harmful vs. overrefusal", fontsize=13)
    style_axis(ax)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    n_runs = table["n_runs"].iloc[0]
    fig.suptitle(f"{dataset}: harmful vs. overrefusal correlation per test dataset\n"
                 f"{n_runs} training runs, scores averaged over {len(models)} models; "
                 "* p < 0.05", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def correlate(scores, model, dataset):
    rows = []
    for rate in HARMFUL_RATES:
        for method, function in CORRELATIONS.items():
            result = function(scores[rate], scores["overrefusal_score_pct"])
            rows.append({
                "train_dataset": dataset, "model": model, "harmful_rate": rate,
                "same_as_training": rate == dataset, "method": method,
                "n_runs": len(scores), "coefficient": float(result.statistic),
                "p_value": float(result.pvalue),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(TEST_DATASETS),
                        default=["aegis", "wildguardmix", "gretel"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS),
                        help="Models to correlate and average (default: %(default)s)")
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "harmful_overrefusal")
    args = parser.parse_args()

    tables = []
    for dataset in args.datasets:
        data, _ = load_scores(args.results_dir, args.results_dir / "if_eval", args.models,
                              f"{args.train_size}_{dataset}")
        data = harmful_rates(data.loc[~data["version"].isin(["-ins", "-20k"])], dataset)
        columns = ["overrefusal_score_pct", *HARMFUL_RATES]
        scores = {model: data.loc[data["model"] == model].set_index("version")[columns]
                  for model in args.models}
        scores["Average"] = data.groupby("version")[columns].mean()
        rows = []
        for model, model_scores in scores.items():
            rows += correlate(model_scores, model, dataset)
        table = pd.DataFrame(rows)
        per_model = table.loc[table["model"].isin(args.models)]
        mean = per_model.groupby(["harmful_rate", "same_as_training", "method"],
                                 sort=False).agg(
            n_runs=("n_runs", "first"), coefficient=("coefficient", "mean"),
        ).reset_index().assign(train_dataset=dataset, model="Mean coefficient")
        table = pd.concat([table, mean], ignore_index=True)
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(dataset_dir / "correlations.csv", index=False, float_format="%.4f")
        average = table.loc[table["model"] == "Average"]
        plot_scatter(scores["Average"], average, dataset, args.models,
                     dataset_dir / "harmful_vs_overrefusal.png")
        plot_correlations(average, dataset, args.models, dataset_dir / "correlations.png")
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    path = args.output_dir / "correlations.csv"
    combined.to_csv(path, index=False, float_format="%.4f")
    print(f"Saved {path}\n")
    for dataset, table in combined.groupby("train_dataset", sort=False):
        wide = table.loc[table["method"] == "pearson"].pivot_table(
            index="model", columns="harmful_rate", values="coefficient", sort=False)
        print(f"{dataset} Pearson (n = {table['n_runs'].iloc[0]} runs per model)")
        print(wide.round(2).to_string(), "\n")


if __name__ == "__main__":
    main()

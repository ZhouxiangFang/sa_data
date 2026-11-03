"""Correlate training-data diversity with post-training overrefusal.

Example (run from any directory):
    python sa_data/script/corr_diversity_overrefusal.py --datasets aegis wildguardmix gretel

The counterpart of plot_bars_diversity.py with overrefusal (the Avg row of
<run>_overrefusal.csv, a macro-average over XSTest and OR-Bench-Hard; lower is
better) in place of the harmful score. Diversity is read from
diversity/<embedder>/csv/<dataset>_train_diversity_*.csv and overrefusal
from results/<model>_git20k_<size>_<dataset>/, through
plot_bars_overall.load_scores, so it matches the plotted bars. As in
plot_bars_diversity.py, the random and evenly baselines are correlated with
the subcategories unless --no-baselines is given, the benign training
subcategory is dropped, --exclude
subcategories are plotted but left out of correlations, Self-BLEU is negated so larger always means more
diverse, and "Average" correlates the equally weighted model-mean overrefusal;
"Mean coefficient" averages the per-model coefficients.
Outputs go to diversity/overrefusal_correlation/ (or --output-dir): per training
dataset, <dataset>/<model>_<group>_diversity_vs_overrefusal.png for the lexical,
semantic and structural groups (every model and the average),
<dataset>/correlations.png (Pearson and Spearman heatmaps, * marks p < 0.05) and
<dataset>/correlations.csv; and a combined correlations.csv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sa_data/ holds the shared modules this script imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_bars_diversity import (CORRELATIONS, FIGURE_METRICS, GROUPS, LABELS, PLOT_METHODS,
                                 correlation_table, join_scores, load_diversity, plot_scatter,
                                 save_figure)
from plot_bars_overall import (BASELINE_ABBRS, DEFAULT_MODELS, DIVERSITY_DIR, MODELS, RESULTS_DIR,
                               TEST_DATASETS, load_scores, training_subcategory)


SCORE = "overrefusal_score_pct"


def load_overrefusal(results_dir, models, suffix):
    """Overrefusal per model and training subcategory, without instruct baselines."""
    data, _ = load_scores(results_dir, results_dir / "if_eval", models, suffix)
    data = data.loc[~data["version"].isin(["-ins", "-20k"])]
    return data.assign(abbr=data["version"].map(training_subcategory))[
        ["model", "abbr", "run", SCORE]
    ]


def diversity_csv(diversity_dir, embedder, dataset):
    paths = sorted((diversity_dir / embedder / "csv").glob(
        f"{dataset}_train_diversity_*.csv"
    ))
    if len(paths) != 1:
        raise ValueError(f"{embedder}/{dataset}: expected one diversity CSV, found {paths}")
    return paths[0]


def plot_heatmaps(table, dataset, context, path, points="subcategories"):
    """Metric x model coefficients, one heatmap per plotted method."""
    models = list(table["model"].unique())
    fig, axes = plt.subplots(1, len(PLOT_METHODS), figsize=(18, 10), sharey=True,
                             layout="constrained")
    for ax, method in zip(axes, PLOT_METHODS):
        rows = table.loc[table["method"] == method]
        values = rows.pivot(index="metric", columns="model", values="coefficient")
        values = values.reindex(index=FIGURE_METRICS, columns=models)
        pvalues = rows.pivot(index="metric", columns="model", values="p_value")
        pvalues = pvalues.reindex(index=FIGURE_METRICS, columns=models)
        image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for (i, j), value in np.ndenumerate(values.to_numpy()):
            if not np.isfinite(value):
                continue
            star = "*" if pvalues.iat[i, j] < 0.05 else ""
            ax.text(j, i, f"{value:+.2f}{star}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > 0.6 else "0.15")
        # Separate the lexical, semantic and structural rows, and the per-model columns.
        groups = [metric.split("_", 1)[0] for metric in FIGURE_METRICS]
        for i in range(1, len(groups)):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color="white", linewidth=3)
        ax.axvline(len(models) - 2.5, color="white", linewidth=3)
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right",
                      rotation_mode="anchor", fontsize=11)
        ax.set_yticks(range(len(FIGURE_METRICS)), [
            f"{LABELS[metric]} {'↑' if metric.endswith('↑') else '↓ (negated)'}"
            for metric in FIGURE_METRICS
        ], fontsize=11)
        for label, metric in zip(ax.get_yticklabels(), FIGURE_METRICS):
            label.set_color(GROUPS[metric.split("_", 1)[0]][1])
        ax.set_title(f"{method.title()} correlation", fontsize=15)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.6,
                 label="Coefficient (negative: more diverse, less overrefusal)")
    n = table.loc[table["model"] != "Mean coefficient", "n_subcategories"]
    count = f"{n.min()}–{n.max()}" if n.nunique() > 1 else f"{n.iloc[0]}"
    fig.suptitle(f"{dataset}: training-data diversity vs. post-training overrefusal\n"
                 f"{context}; {count} {points}; * p < 0.05 (not computed for "
                 "mean coefficients)", fontsize=15)
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(TEST_DATASETS),
                        default=["aegis", "wildguardmix", "gretel"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS),
                        help="Models to correlate and average (default: %(default)s)")
    parser.add_argument("--embedder", default="bge-m3",
                        help="Embedding-model folder under --diversity-dir (default: bge-m3)")
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Subcategories to exclude from correlations")
    parser.add_argument("--no-baselines", dest="baselines", action="store_false",
                        help="Leave out the random and evenly baselines (default: include them)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--diversity-dir", type=Path, default=DIVERSITY_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=DIVERSITY_DIR / "overrefusal_correlation")
    args = parser.parse_args()

    methods = list(CORRELATIONS)
    excluded = {abbr.lower() for abbr in args.exclude}
    tables = []
    for dataset in args.datasets:
        diversity = load_diversity(diversity_csv(args.diversity_dir, args.embedder, dataset),
                                   args.baselines)
        suffix = f"{args.train_size}_{dataset}"
        safety = load_overrefusal(args.results_dir, args.models, suffix)
        pairs = join_scores(diversity, safety, SCORE)
        pairs["included_in_correlation"] = ~pairs["abbr"].str.lower().isin(excluded)
        table = correlation_table(pairs, methods, SCORE)
        per_model = table.loc[table["model"].isin(args.models)]
        mean = per_model.groupby(["group", "metric", "method"], sort=False).agg(
            n_subcategories=("n_subcategories", "first"), coefficient=("coefficient", "mean"),
        ).reset_index().assign(model="Mean coefficient", safety_metric=SCORE)
        table = pd.concat([table, mean], ignore_index=True)

        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(dataset_dir / "correlations.csv", index=False, float_format="%.4f")
        context = (f"{dataset}, embeddings: {args.embedder}, results: git20k_{suffix}; "
                   "overrefusal = mean of XSTest / OR-Bench-Hard")
        for model, scores in pairs.groupby("model", sort=False):
            title = (model if model != "Average"
                     else f"Average across {len(args.models)} models")
            for group in GROUPS:
                plot_scatter(scores, table.loc[table["model"] == model], title, context,
                             dataset_dir / f"{model.lower()}_{group}_diversity_vs_overrefusal.png",
                             group, score=SCORE)
        points = ("training sets (subcategories + random/evenly baselines)"
                  if pairs["abbr"].isin(BASELINE_ABBRS).any() else "subcategories")
        plot_heatmaps(table, dataset, f"embeddings: {args.embedder}",
                      dataset_dir / "correlations.png", points)
        tables.append(table.assign(train_dataset=dataset))

    path = args.output_dir / "correlations.csv"
    pd.concat(tables, ignore_index=True).to_csv(path, index=False, float_format="%.4f")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

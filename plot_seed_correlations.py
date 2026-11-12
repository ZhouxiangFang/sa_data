"""Bar chart of diversity-safety correlations averaged over seeds, one figure per dataset.

Reads each seed's correlations, diversity/seed<seed>/<embedder>/csv/
<dataset>_diversity_safety_correlations.csv (written by plot_bars_diversity.py),
for every --embedders folder, and keeps the model-average harmful score rows.
As in summarize_diversity.py, a seed's value for a metric is the mean of its
Pearson and Spearman coefficients (--methods). Each metric gets one bar per
embedding model: the mean of those values over --seeds, with whiskers of +-1 SD
across seeds (ddof=1, clipped to [-1, 1]) and the individual seeds as dots.
All figures share one y-range. Lexical and most
structural metrics do not use embeddings, so their bars match across embedders.
Self-BLEU is negated in the CSVs, so for every metric a negative coefficient
means more diverse training data, less harmful model. Figures show the 12
plot_bars_diversity.FIGURE_METRICS; the CSV has every metric.

Writes <output_dir>/<dataset>_by_embedder_seeds<seeds>.{png,pdf,csv}.

Example:
    python sa_data/plot_seed_correlations.py --seeds 43 86
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from helper import TRAIN_DATASETS, seed_dir
from plot_bars_diversity import (FIGURE_METRICS, GROUPS, LABELS, METRICS, PLOT_METHODS,
                                 TEST_DATASETS)
from plot_bars_overall import DIVERSITY_DIR

# Folder: (display name, color). Colors are categorical slots 1 and 2 of the
# reference palette, as in plot_bars_diversity.GROUPS.
EMBEDDERS = {
    "bge-m3": ("BGE-M3", "#2a78d6"),
    "qwen3-embedding-0.6b": ("Qwen3-Embedding-0.6B", "#eb6834"),
}
SEED_MARKERS = ("o", "D", "s", "^", "v")
INK = "#1f1f1f"
MUTED = "#5f6368"
GRID = "#e6e6e6"
BAND = 0.8  # Width of one metric's bars together.


def seed_values(diversity_dir, seed, embedder, dataset, methods):
    """Metric -> mean coefficient over methods for the model-average harmful score, and n_runs."""
    path = (diversity_dir / seed_dir(seed) / embedder / "csv"
            / f"{dataset}_diversity_safety_correlations.csv")
    table = pd.read_csv(path)
    table = table[(table["model"] == "Average") & table["method"].isin(methods)
                  & (table["safety_metric"] == "harmful_score_pct")]
    missing = set(methods).difference(table["method"])
    if missing:
        raise ValueError(f"{path} has no {sorted(missing)} rows")
    return table.groupby("metric")["coefficient"].mean(), int(table["n_runs"].iloc[0])


def seed_table(diversity_dir, seeds, embedders, dataset, methods):
    """Long table: one row per metric and embedder, a column per seed, mean and sd."""
    blocks, runs = [], set()
    for embedder in embedders:
        columns = {}
        for seed in seeds:
            columns[f"seed{seed}"], n_runs = seed_values(diversity_dir, seed, embedder,
                                                         dataset, methods)
            runs.add(n_runs)
        block = pd.DataFrame(columns).reindex(list(METRICS))
        values = block[[f"seed{seed}" for seed in seeds]]
        block["mean"] = values.mean(axis=1)
        block["sd"] = values.std(axis=1, ddof=1)
        blocks.append(block.rename_axis("metric").reset_index().assign(embedder=embedder))
    columns = ["metric", "embedder", *(f"seed{seed}" for seed in seeds), "mean", "sd"]
    return pd.concat(blocks, ignore_index=True)[columns], runs


def whiskers(mean, sd):
    """+-1 SD, clipped to the [-1, 1] range of a correlation coefficient."""
    return np.vstack([np.minimum(sd, mean + 1), np.minimum(sd, 1 - mean)])


def y_limits(tables, seeds):
    """(bottom, top tick, top) shared by every figure: -1 up to the highest dot or whisker."""
    tables = [table[table["metric"].isin(FIGURE_METRICS)] for table in tables]
    highest = max(
        np.nanmax(np.concatenate([table[[f"seed{seed}" for seed in seeds]].to_numpy(dtype=float).ravel(),
                                  (table["mean"] + whiskers(table["mean"], table["sd"])[1]).to_numpy()]))
        for table in tables)
    top = max(0.25, highest) + 0.12  # Headroom for the group headers.
    return -1.0, np.floor((top - 0.1) * 4) / 4, top


def plot(table, seeds, runs, dataset, embedders, methods, path, limits):
    metrics = FIGURE_METRICS
    x = np.arange(len(metrics))
    groups = [metric.split("_", 1)[0] for metric in metrics]
    width = BAND / len(embedders)
    low, top_tick, high = limits

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    seed_offsets = (np.linspace(-0.22, 0.22, len(seeds)) * width if len(seeds) > 1 else [0.0])
    for i, embedder in enumerate(embedders):
        rows = table[table["embedder"] == embedder].set_index("metric").reindex(metrics)
        position = x - BAND / 2 + width * (i + 0.5)
        ax.bar(position, rows["mean"], width=width * 0.9, color=EMBEDDERS[embedder][1], zorder=2)
        ax.errorbar(position, rows["mean"], yerr=whiskers(rows["mean"].to_numpy(),
                                                           rows["sd"].to_numpy()),
                    fmt="none", ecolor=INK,
                    elinewidth=1, capsize=2, capthick=1, zorder=3)
        for seed, offset, marker in zip(seeds, seed_offsets, SEED_MARKERS):
            ax.scatter(position + offset, rows[f"seed{seed}"], marker=marker, s=12,
                       facecolor="white", edgecolor=INK, linewidth=0.8, zorder=4)

    # Thin dividers and headers between the metric groups.
    starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(groups)
        if start:
            ax.axvline(start - 0.5, color="#c8c8c8", linewidth=1, zorder=0)
        ax.text((start + end - 1) / 2, high - 0.02, GROUPS[groups[start]][0],
                ha="center", va="top", fontsize=10.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[metric] for metric in metrics], rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, len(metrics) - 0.4)
    ax.set_ylim(low, high)
    ax.set_yticks(np.arange(low, top_tick + 1e-9, 0.25))
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    method_name = " & ".join(method.title() for method in methods)
    ax.set_ylabel(f"Correlation with harmful score\n(mean of {method_name})",
                  fontsize=10.5, color=INK)

    run_text = (f"{next(iter(runs))} runs per seed" if len(runs) == 1
                else f"{min(runs)}-{max(runs)} runs per seed")
    seed_text = ", ".join(str(seed) for seed in seeds)
    ax.set_title(f"{TEST_DATASETS.get(dataset, dataset)}: training-data diversity vs. "
                 "harmful score", loc="left", fontsize=13, color=INK, pad=46)
    ax.text(0, 1.02, f"Model-average harmful score, {run_text}. Bars: mean over seeds "
            f"{seed_text}; whiskers: ±1 SD; dots: single seeds. "
            "Negative = more diverse, less harmful.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")

    handles = [Patch(color=EMBEDDERS[embedder][1], label=EMBEDDERS[embedder][0])
               for embedder in embedders]
    handles += [Line2D([], [], marker=marker, linestyle="none", markerfacecolor="white",
                       markeredgecolor=INK, markersize=4.5, label=f"Seed {seed}")
                for seed, marker in zip(seeds, SEED_MARKERS)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.07), ncols=len(handles),
              fontsize=9, frameon=False, handlelength=1.2, columnspacing=1.4, borderaxespad=0)

    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(f"{path}{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 86])
    parser.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    parser.add_argument("--embedders", nargs="+", default=list(EMBEDDERS), choices=list(EMBEDDERS),
                        help="Embedding-model folders, one bar each (default: %(default)s)")
    parser.add_argument("--methods", nargs="+", default=list(PLOT_METHODS),
                        choices=["pearson", "spearman", "kendall"],
                        help="Coefficients averaged per seed (default: %(default)s)")
    parser.add_argument("--diversity_dir", type=Path, default=DIVERSITY_DIR)
    parser.add_argument("--output_dir", type=Path, help="Default: <diversity_dir>/summary/figs")
    args = parser.parse_args()
    output_dir = args.output_dir or args.diversity_dir / "summary" / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "-".join(str(seed) for seed in args.seeds)
    tables = {dataset: seed_table(args.diversity_dir, args.seeds, args.embedders, dataset,
                                  args.methods) for dataset in args.datasets}
    limits = y_limits([table for table, _ in tables.values()], args.seeds)
    for dataset, (table, runs) in tables.items():
        path = output_dir / f"{dataset}_by_embedder_seeds{seed_tag}"
        table.to_csv(f"{path}.csv", index=False, float_format="%.4f")
        plot(table, args.seeds, runs, dataset, args.embedders, args.methods, path, limits)
        print(f"{dataset}: {path}.png (.pdf, .csv)")


if __name__ == "__main__":
    main()

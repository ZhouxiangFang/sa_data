"""Correlate training-data diversity with the trained models' harmful score.

    python correlate_diversity.py
    python correlate_diversity.py --results_dir ../ablations/refusal --diversity_dir ../diversity/refusal

A run's harmful score is the Avg row of its <results_dir>/seed<S>/<model>_<num_train>_wildguardmix/
wildguardmix_<abbr>/wildguardmix_<abbr>_overall.csv: the mean harmful-response rate over the five
harm test sets (eval.py). Each run is paired with the diversity of its own training sample (same
group and seed: <diversity_dir>/seed<S>/<embedder>/csv/, from calculate_diversity.py). A point is
one training group (subcategory, random or evenly):
  <model>   that model's harmful score and the diversity, averaged over its seeds
  Average   the models' mean harmful score per group and seed (only where every model has the
            run), averaged over seeds
Pearson and Spearman coefficients are computed across the groups, per embedder, model and metric:
on the seed-mean points (seed "mean") and on each seed's points alone (seed <S>). POS Self-BLEU
is negated first, so for every metric a negative coefficient means more diverse training data,
less harmful model.

Writes to <diversity_dir>/summary/:
  wildguardmix_<n>_correlations.csv  every coefficient with its p-value and point count
  wildguardmix_<n>_points.csv        the seed-mean points
  wildguardmix_<n>_<embedder>.png    Average model: harmful score vs each metric (seed means;
                                     error bars: SD across seeds)
  wildguardmix_<n>_correlations.png  Average model, per metric and embedder: mean over seeds of
                                     the per-seed (Pearson + Spearman) / 2, whiskers ±1 SD
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from calculate_diversity import EMBEDDING_MODEL_IDS, METRICS, diversity_csv, embedding_name
from helper import BASELINE_ABBRS, DATASET, GIT20K_MODELS, ROOT, mixture_counts, seed_dir

LABELS = {
    "lexical_hdd↑": "HD-D", "lexical_distinct_2↑": "Distinct-2",
    "semantic_vendi↑": "Vendi", "semantic_distsum_cosine↑": "DistSum",
    "semantic_cluster_inertia↑": "Cluster inertia",
    "structural_self_bleu_4↓": "POS Self-BLEU-4", "structural_self_bleu_5↓": "POS Self-BLEU-5",
    "structural_distinct_5↑": "POS Distinct-5", "structural_distinct_6↑": "POS Distinct-6",
    "structural_pos_vendi↑": "POS-embedding Vendi",
    "structural_knn_cosine_1↑": "POS KNN-1", "structural_knn_cosine_3↑": "POS KNN-3",
}
# Categorical slots 1-3 of the reference palette (they pass every all-pairs check).
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")
GROUP_COLORS = dict(zip(("lexical", "semantic", "structural"), SLOTS))
EMBEDDERS = {embedding_name(model_id): model_id.split("/")[-1] for model_id in EMBEDDING_MODEL_IDS}
METHODS = {"pearson": pearsonr, "spearman": spearmanr}
INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#e6e6e6"
SEED_MARKERS = ("o", "D", "s", "^", "v")


def sign(metric):
    """+1 if higher values mean more diversity, -1 if lower values do (POS Self-BLEU)."""
    return -1 if metric.endswith("↓") else 1


def load_diversity(diversity_dir, embedder, harmful_count):
    """Every scored seed's rows of one embedder, with a seed column."""
    name = diversity_csv(diversity_dir, embedder, harmful_count).name
    paths = sorted(Path(diversity_dir).glob(f"seed*/{embedder}/csv/{name}"))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def load_runs(results_dir, num_train, models, abbrs, seeds):
    """One row per evaluated run: model, abbr, seed, harmful (Avg harmful_score_pct)."""
    rows = []
    for seed in seeds:
        for model in models:
            for abbr in abbrs:
                group = f"{DATASET}_{abbr}"
                path = (Path(results_dir) / seed_dir(seed) / f"{model}_{num_train}_{DATASET}"
                        / group / f"{group}_overall.csv")
                if path.exists():
                    table = pd.read_csv(path).set_index("dataset")
                    rows.append({"model": model.removesuffix("_git20k"), "abbr": abbr,
                                 "seed": seed, "harmful": float(table.loc["Avg",
                                                                          "harmful_score_pct"])})
    return pd.DataFrame(rows, columns=["model", "abbr", "seed", "harmful"])


def pair(runs, diversity):
    """(per-seed points, seed-mean points): each run with its own sample's diversity, plus the
    Average model where every model has the run."""
    runs = runs.merge(diversity[["abbr", "seed", *METRICS]], on=["abbr", "seed"],
                      validate="many_to_one")
    n_models = runs["model"].nunique()
    complete = runs.groupby(["abbr", "seed"]).filter(lambda rows: len(rows) == n_models)
    average = complete.groupby(["abbr", "seed"], as_index=False).agg(
        harmful=("harmful", "mean"), **{metric: (metric, "first") for metric in METRICS})
    by_seed = pd.concat([runs, average.assign(model="Average")], ignore_index=True)
    grouped = by_seed.groupby(["model", "abbr"], sort=False)
    means = grouped[["harmful", *METRICS]].mean()
    means.insert(1, "harmful_sd", grouped["harmful"].std())
    means.insert(2, "n_seeds", grouped.size())
    return by_seed, means.reset_index()


def correlation_rows(points, seed):
    """Pearson and Spearman of harmful vs each (signed) metric across groups, per model."""
    rows = []
    for model, group in points.groupby("model", sort=False):
        for metric in METRICS:
            x, y = sign(metric) * group[metric].to_numpy(), group["harmful"].to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            for method, function in METHODS.items():
                coefficient = p_value = np.nan
                if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    result = function(x, y)
                    coefficient, p_value = float(result.statistic), float(result.pvalue)
                rows.append({"model": model, "metric": metric, "method": method, "seed": seed,
                             "n_groups": len(x), "coefficient": coefficient, "p_value": p_value})
    return rows


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_scatter(points, table, title, path):
    """One panel per metric: Average harmful score vs diversity of the seed-mean points."""
    average = points[points["model"] == "Average"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 12.5), layout="constrained")
    for ax, metric in zip(axes.ravel(), METRICS):
        color = GROUP_COLORS[metric.split("_", 1)[0]]
        x, y = sign(metric) * average[metric], average["harmful"]
        baseline = average["abbr"].isin(BASELINE_ABBRS)
        ax.errorbar(x, y, yerr=average["harmful_sd"].fillna(0), fmt="none", ecolor=color,
                    alpha=0.35, elinewidth=1, capsize=2, zorder=2)
        ax.scatter(x[~baseline], y[~baseline], s=48, color=color, edgecolors="white",
                   linewidths=1.5, zorder=3)
        ax.scatter(x[baseline], y[baseline], s=58, marker="s", color=color, edgecolors=INK,
                   linewidths=1, zorder=4)
        for xi, yi, abbr in zip(x, y, average["abbr"]):
            ax.annotate(abbr, (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=7,
                        color=MUTED)
        if len(x) >= 3 and np.ptp(x) > 0:
            line = np.linspace(x.min(), x.max(), 50)
            ax.plot(line, np.polyval(np.polyfit(x, y, 1), line), color=color, linewidth=1.5,
                    alpha=0.7)
        stats = table[(table["model"] == "Average") & (table["metric"] == metric)]
        text = []
        for method in METHODS:
            rows = stats[stats["method"] == method]
            value = rows.loc[rows["seed"] == "mean", "coefficient"].iloc[0]
            seeds = rows.loc[rows["seed"] != "mean", "coefficient"].astype(float)
            line = f"{method.title()} {value:+.2f}"
            if len(seeds) >= 2:
                line += f"  (seeds {seeds.mean():+.2f} ± {seeds.std():.2f})"
            text.append(line)
        ax.text(0.03, 0.97, "\n".join(text), transform=ax.transAxes, va="top", fontsize=8.5,
                color=INK, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        prefix = "−" if sign(metric) < 0 else ""
        ax.set_title(LABELS[metric], fontsize=11.5, color=INK, loc="left")
        ax.set_xlabel(f"{prefix}{LABELS[metric]} (more diverse →)", fontsize=9, color=MUTED)
        ax.set_ylabel("Harmful score (%)", fontsize=9, color=MUTED)
        ax.margins(x=0.15, y=0.25)
        style(ax)
    handles = [Line2D([], [], marker="o", linestyle="", color=MUTED, label="Subcategory"),
               Line2D([], [], marker="s", linestyle="", color=MUTED, markeredgecolor=INK,
                      label="Random / evenly baseline"),
               Line2D([], [], color=MUTED, label="Least-squares fit"),
               Line2D([], [], color=MUTED, alpha=0.5, marker="_", label="±1 SD across seeds")]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=14, color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def plot_seed_bars(table, embedders, title, path):
    """Average model: per metric and embedder, mean over seeds of (Pearson + Spearman) / 2."""
    per_seed = table[(table["model"] == "Average") & (table["seed"] != "mean")]
    values = (per_seed.groupby(["embedder", "metric", "seed"])["coefficient"].mean()
              .unstack("seed"))
    x = np.arange(len(METRICS))
    width = 0.8 / len(embedders)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    seeds = list(values.columns)
    for i, embedder in enumerate(embedders):
        rows = values.loc[embedder].reindex(METRICS).astype(float)
        mean, sd = rows.mean(axis=1), rows.std(axis=1).fillna(0)
        position = x - 0.4 + width * (i + 0.5)
        ax.bar(position, mean, width * 0.9, color=SLOTS[i], zorder=2, label=EMBEDDERS[embedder])
        ax.errorbar(position, mean, yerr=[np.minimum(sd, mean + 1), np.minimum(sd, 1 - mean)],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        offsets = np.linspace(-0.22, 0.22, len(seeds)) * width if len(seeds) > 1 else [0.0]
        for seed, offset, marker in zip(seeds, offsets, SEED_MARKERS):
            ax.scatter(position + offset, rows[seed], marker=marker, s=12, facecolor="white",
                       edgecolor=INK, linewidth=0.8, zorder=4)
    # Dividers between the lexical, semantic and structural metrics.
    groups = [metric.split("_", 1)[0] for metric in METRICS]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axvline(i - 0.5, color=GRID, linewidth=1, zorder=0)
    ax.set_xticks(x, [LABELS[metric] for metric in METRICS], rotation=35, ha="right",
                  rotation_mode="anchor", fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, len(METRICS) - 0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Correlation with harmful score\n(mean of Pearson & Spearman)", fontsize=10,
                  color=INK)
    style(ax)
    handles = [Patch(color=SLOTS[i], label=EMBEDDERS[embedder])
               for i, embedder in enumerate(embedders)]
    handles += [Line2D([], [], marker=marker, linestyle="none", markerfacecolor="white",
                       markeredgecolor=INK, markersize=4.5, label=f"Seed {seed}")
                for seed, marker in zip(seeds, SEED_MARKERS)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(handles),
              fontsize=9, frameon=False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--diversity_dir", type=Path, default=ROOT / "diversity")
    parser.add_argument("--num_train", type=int, default=1800,
                        help="Examples per run (a third are the harmful sample)")
    parser.add_argument("--models", nargs="+", default=list(GIT20K_MODELS))
    args = parser.parse_args()
    harmful_count = mixture_counts(args.num_train)[0]
    output_dir = args.diversity_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{DATASET}_{harmful_count}"
    embedders = [embedder for embedder in EMBEDDERS
                 if (args.diversity_dir / embedder / "csv"
                     / diversity_csv("", embedder, harmful_count).name).exists()]
    if not embedders:
        parser.error(f"No {stem} diversity CSVs in {args.diversity_dir}; "
                     "run calculate_diversity.py first")

    tables, all_points = [], []
    for embedder in embedders:
        diversity = load_diversity(args.diversity_dir, embedder, harmful_count)
        seeds = sorted(diversity["seed"].unique())
        runs = load_runs(args.results_dir, args.num_train, args.models,
                         diversity["abbr"].unique(), seeds)
        if runs.empty:
            parser.error(f"No evaluated runs in {args.results_dir} for seeds {seeds}")
        by_seed, points = pair(runs, diversity)
        rows = correlation_rows(points, "mean")
        for seed, seed_points in by_seed.groupby("seed"):
            rows += correlation_rows(seed_points, str(seed))
        table = pd.DataFrame(rows).assign(embedder=embedder)
        tables.append(table)
        all_points.append(points.assign(embedder=embedder))
        counts = runs.groupby("seed").size()
        print(f"{embedder}: {len(runs)} runs ("
              + ", ".join(f"seed {seed}: {count}" for seed, count in counts.items())
              + f"), {points['abbr'].nunique()} groups")
        title = (f"Training-data diversity vs. harmful score, average of {runs['model'].nunique()} "
                 f"models ({EMBEDDERS[embedder]} embeddings; points: means over seeds "
                 f"{', '.join(map(str, seeds))})")
        plot_scatter(points, table, textwrap.fill(title, 120),
                     output_dir / f"{stem}_{embedder}.png")

    table = pd.concat(tables, ignore_index=True)
    table = table[["embedder", "model", "metric", "method", "seed", "n_groups", "coefficient",
                   "p_value"]]
    table.to_csv(output_dir / f"{stem}_correlations.csv", index=False, float_format="%.4f")
    pd.concat(all_points, ignore_index=True).to_csv(output_dir / f"{stem}_points.csv",
                                                    index=False, float_format="%.6g")
    summary = (table[(table["model"] == "Average") & (table["seed"] == "mean")]
               .pivot_table(index="metric", columns=["embedder", "method"], values="coefficient")
               .reindex(list(METRICS)))
    print("Average model, seed-mean points (negative: more diverse, less harmful):")
    print(summary.round(3).to_string())
    plot_seed_bars(table, embedders, "Training-data diversity vs. harmful score (average model). "
                   "Negative = more diverse, less harmful.",
                   output_dir / f"{stem}_correlations.png")
    print(f"Saved {output_dir / f'{stem}_correlations.csv'} and {stem}_points.csv")


if __name__ == "__main__":
    main()

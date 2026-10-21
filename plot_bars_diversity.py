"""Plot a diversity CSV and its relationship with post-training safety.

Example (run from any directory):
    python sa_data/plot_bars_diversity.py \
        results/diversity/bge-m3/csv/wildguardmix_train_diversity_600_1.0.csv \
        --suffix 1800_wildguardmix --corr all

Each embedding model has one folder, results/diversity/<model>/, with csv/ (the
diversity CSVs from calculate_diversity.py and this script's correlations) and
figs/. Every output file name starts with the dataset.

The metrics are the lexical_*, semantic_* and structural_* columns written by
calculate_diversity.py, whose ↑/↓ suffix gives the direction of more diversity.
By default three figures, <dataset>_average_<group>_diversity_vs_harmful.png for
the lexical, semantic and structural groups (one scatter panel per metric in the
group against the equally weighted model-mean harmful score), and
<dataset>_diversity_safety_correlations.csv are written. --all-plots adds
per-metric bar charts, the same three figures per model, correlation bar charts,
<dataset>_average_<group>_diversity_vs_harmful_<test>.png per test dataset, and
the plotted scores (<dataset>_diversity_safety_scores.csv and per-test CSVs).
Bar charts display bounded ratios (HD-D, Distinct, Self-BLEU) as percentages and
other metrics in their original units. CSV exports keep the original units.
Figures show, average and rank Pearson and Spearman coefficients only; Kendall
(largely redundant with Spearman) is kept in the correlation CSVs. Figures also
omit POS Self-BLEU-1, which saturates near 0.999; the CSVs keep it.
Safety uses the precomputed Avg harmful_score_pct (lower is safer), matching
plot_bars_overall.py. The per-test plots use each test dataset's harmful score,
averaged equally across models.
Their plotted scores and correlations are exported in average_diversity_safety_*
CSVs with a test_dataset column. Correlations use negative Self-BLEU so larger always
means more diverse. Average correlations are computed against equally weighted
model-mean harmful scores on subcategories available for every model; they
are not averages of correlation coefficients. The pooled overall diversity
reference is excluded from correlations. The benign subcategory and any --exclude
subcategories are plotted but excluded from correlation coefficients and fitted
lines. CSV exports contain plotted pairs
and coefficients, including sample counts and p-values.
Run names encode dataset, subcategory, and size, but not the mixture fraction;
use --suffix to select grouped result folders for the intended configuration.
Outputs are named by dataset only, so use --output-dir to keep the results of
another training size or variant apart.
"""

import argparse
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
# Column prefix: (section title, color). Colors are the first three slots of the
# reference categorical palette, which stay distinguishable in scatter plots.
GROUPS = {
    "lexical": ("Lexical", "#2a78d6"),
    "semantic": ("Semantic", "#eb6834"),
    "structural": ("Structural", "#1baf7a"),
}
# Column: display label. The ↑/↓ suffix marks the direction of more diversity.
LABELS = {
    "lexical_hdd↑": "HD-D",
    "lexical_distinct_2↑": "Distinct-2",
    "semantic_vendi↑": "Vendi score",
    "semantic_distsum_cosine↑": "DistSum (cosine)",
    "semantic_cluster_inertia↑": "Cluster inertia",
    "semantic_radius↑": "Radius",
    **{f"structural_self_bleu_{n}↓": f"POS Self-BLEU-{n}" for n in range(1, 6)},
    "structural_distinct_1_4↑": "POS Distinct-1–4 avg",
    "structural_ngram_vendi↑": "POS n-gram Vendi",
    "structural_pos_vendi↑": "POS-embedding Vendi",
    "structural_knn_cosine↑": "POS KNN cosine distance",
}
# Column: (display label, direction of increasing diversity, color).
METRICS = {
    column: (label, -1 if column.endswith("↓") else 1, GROUPS[column.split("_", 1)[0]][1])
    for column, label in LABELS.items()
}
# Self-BLEU-1 saturates near 0.999 on POS tags; CSVs keep it, figures leave it out.
FIGURE_METRICS = [metric for metric in METRICS if metric != "structural_self_bleu_1↓"]
# Bounded ratios that bar charts display as percentages.
PERCENT_METRICS = {"lexical_hdd↑", "lexical_distinct_2↑", "structural_distinct_1_4↑",
                   *(f"structural_self_bleu_{n}↓" for n in range(1, 6))}
GRID_COLUMNS = 4
# Background of the two metrics with the lowest mean correlation in a figure.
HIGHLIGHT = "#E1F5FE"
CORRELATIONS = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}
# Methods that figures show and average; CSVs keep every computed method.
PLOT_METHODS = ("pearson", "spearman")
TEST_DATASETS = {
    "wildguardmix": "WildGuardMix",
    "aegis": "Aegis",
    "gretel": "Gretel",
    "wildjailbreak": "WildJailbreak",
    "ailuminate": "AILuminate",
}


def load_diversity(path):
    data = pd.read_csv(path)
    missing = {"abbr", *METRICS} - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if data.empty or data["abbr"].isna().any() or data["abbr"].duplicated().any():
        raise ValueError(f"{path}: expected nonempty, unique subcategory abbreviations")
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
        if not data.loc[data["abbr"] != "overall", metric].notna().any():
            raise ValueError(f"{path}: no subcategory values for {metric}")
    return data


def load_safety(results_dir, dataset, train_size, abbreviations, suffix=None,
                test_dataset="Avg"):
    """Read one test score from grouped results or legacy flat runs by size."""
    pattern = re.compile(rf"^(.+)_{re.escape(dataset)}_(.+)_{train_size}$")
    group_suffix = "_" + (suffix or f"{train_size}_{dataset}")
    candidates = []
    rows = []
    for directory in sorted(results_dir.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.endswith(group_suffix):
            model = directory.name[:-len(group_suffix)]
            for abbr in sorted(abbreviations):
                category = f"{dataset}_{abbr}"
                run_dir = directory / category
                if run_dir.is_dir():
                    candidates.append((model, abbr, f"{directory.name}/{category}",
                                       run_dir / f"{category}_overall.csv"))
        if suffix is not None:
            continue
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        model, abbr = match.groups()
        if abbr in abbreviations:
            candidates.append((model, abbr, directory.name,
                               directory / f"{directory.name}_overall.csv"))
    for model, abbr, run, path in candidates:
        # Use the run's exact filename, never an arbitrary glob match.
        if not path.exists():
            print(f"Warning: missing {path}; skipping run")
            continue
        data = pd.read_csv(path)
        scores = data.loc[data["dataset"] == test_dataset, "harmful_score_pct"]
        if scores.empty and test_dataset != "Avg":
            # Older evaluations may predate a newly added test dataset.
            print(f"Warning: {path} has no {test_dataset} row; skipping this test dataset for the run")
            continue
        if len(scores) != 1:
            raise ValueError(f"{path}: expected exactly one {test_dataset} row")
        score = float(scores.iloc[0])
        if not np.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f"{path}: invalid harmful percentage {score}")
        # Display model names in the same way as plot_bars_overall.py.
        model = re.sub(r"(?:_git20k|-git20k|-20k)$", "", model)
        rows.append({"model": model, "abbr": abbr, "run": run,
                     "harmful_score_pct": score})
    result = pd.DataFrame(rows, columns=["model", "abbr", "run", "harmful_score_pct"])
    if result.duplicated(["model", "abbr"]).any():
        raise ValueError("Multiple runs match the same model and subcategory")
    return result


def join_scores(diversity, safety):
    """Add the equal-model average using only complete subcategory coverage."""
    subs = diversity.loc[diversity["abbr"] != "overall"]
    pairs = safety.merge(subs, on="abbr", validate="many_to_one")
    pairs["model_count"] = 1
    wide = safety.pivot(index="abbr", columns="model", values="harmful_score_pct")
    complete = wide.dropna()
    if complete.empty:
        print("Warning: no subcategories shared by all models; no average plot")
        return pairs
    average = complete.mean(axis=1).rename("harmful_score_pct").reset_index()
    average = average.merge(subs, on="abbr", validate="one_to_one")
    average = average.assign(model="Average", run="", model_count=len(wide.columns))
    return pd.concat([pairs, average], ignore_index=True)


def correlation_mask(scores):
    """Keep benign and explicitly excluded subcategories out of statistics."""
    included = ~scores["abbr"].str.lower().eq("benign")
    if "included_in_correlation" in scores:
        included &= scores["included_in_correlation"]
    return included


def finite_pairs(scores, metric, for_correlation=False):
    sign = METRICS[metric][1]
    x = sign * scores[metric].to_numpy(dtype=float)
    y = scores["harmful_score_pct"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if for_correlation:
        valid &= correlation_mask(scores).to_numpy()
    return x[valid], y[valid], scores.loc[valid, "abbr"].tolist()


def correlation_table(pairs, methods):
    rows = []
    for model, scores in pairs.groupby("model", sort=False):
        for metric in METRICS:
            x, y, _ = finite_pairs(scores, metric, for_correlation=True)
            for method in methods:
                coefficient = pvalue = np.nan
                if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    result = CORRELATIONS[method](x, y)
                    coefficient, pvalue = float(result.statistic), float(result.pvalue)
                rows.append({
                    "model": model, "group": metric.split("_", 1)[0],
                    "metric": metric, "method": method,
                    "diversity_multiplier": METRICS[metric][1],
                    "safety_metric": "harmful_score_pct", "n_subcategories": len(x),
                    "coefficient": coefficient, "p_value": pvalue,
                })
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=13)


def save_figure(fig, path):
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def plot_diversity(data, metric, context, path):
    label, sign, color = METRICS[metric]
    reference = data.loc[data["abbr"] == "overall"]
    subs = data.loc[data["abbr"] != "overall"].sort_values(
        metric, ascending=sign > 0, kind="stable"
    )
    ordered = pd.concat([reference, subs])
    x = np.arange(len(ordered), dtype=float)
    if len(reference):
        x[1:] += 0.6
    fig, ax = plt.subplots(figsize=(max(12, len(ordered) * 0.85), 7))
    colors = ["#999999" if abbr == "overall" else color for abbr in ordered["abbr"]]
    # Display bounded ratios as percentages; other metrics keep their native units.
    scale = 100 if metric in PERCENT_METRICS else 1
    values = ordered[metric] * scale
    bars = ax.bar(x, values, 0.65, color=colors, zorder=3)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            # Three significant digits keep small values such as radius readable.
            ax.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, ordered["abbr"], rotation=45, ha="right", rotation_mode="anchor")
    if len(reference) and len(subs):
        ax.axvline((x[0] + x[1]) / 2, color="0.65", linestyle="--", linewidth=1)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.02 * abs(high), 1e-9)
    ax.set_ylim(max(0, low - span * 0.15), high + span * 0.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.set_ylabel("Score (%)" if scale == 100 else "Score", fontsize=18)
    ax.set_xlabel(f"Training subcategory ({context})\nIncreasing diversity from left to right",
                  fontsize=16, labelpad=12)
    ax.set_title(f"Diversity measured by {label} ({'↑' if sign > 0 else '↓'})",
                 pad=20, fontsize=23)
    style_axis(ax)
    save_figure(fig, path)


def plot_scatter_panel(ax, scores, correlations, metric, test_dataset, show_ylabel,
                       bold=False, highlight=False):
    label, sign, color = METRICS[metric]
    x, y, abbreviations = finite_pairs(scores, metric)
    excluded_abbrs = set(scores.loc[~correlation_mask(scores), "abbr"])
    excluded = np.array([abbr in excluded_abbrs for abbr in abbreviations], dtype=bool)
    ax.scatter(x[~excluded], y[~excluded], color=color, s=48, edgecolors="white",
               linewidths=1, zorder=3)
    if excluded.any():
        ax.scatter(x[excluded], y[excluded], facecolors="none", edgecolors="0.3",
                   marker="D", s=60, zorder=4)
    for xi, yi, abbr in zip(x, y, abbreviations):
        ax.annotate(abbr, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7)
    corr_x, corr_y, _ = finite_pairs(scores, metric, for_correlation=True)
    if len(corr_x) >= 3 and np.ptp(corr_x) > 0 and np.ptp(corr_y) > 0:
        line_x = np.linspace(corr_x.min(), corr_x.max(), 100)
        ax.plot(line_x, np.polyval(np.polyfit(corr_x, corr_y, 1), line_x),
                color=color, alpha=0.65, linewidth=1.5)
    stats = correlations.loc[correlations["metric"] == metric]
    summary = []
    for row in stats.itertuples():
        value = f"{row.coefficient:+.3f}" if np.isfinite(row.coefficient) else "undefined"
        summary.append(f"{row.method.title()}: {value}")
    if len(stats) > 1:
        mean = stats["coefficient"].mean()
        summary.append(f"Mean: {mean:+.3f}" if np.isfinite(mean) else "Mean: undefined")
    weight = "bold" if bold else "normal"
    ax.text(0.03, 0.97, "\n".join(summary), transform=ax.transAxes, va="top",
            fontsize=9, fontweight=weight,
            bbox={"facecolor": HIGHLIGHT if highlight else "white", "alpha": 0.85,
                  "edgecolor": "none"})
    ax.set_title(f"{label} {'↑' if sign > 0 else '↓'}", fontsize=14, pad=8, fontweight=weight,
                 bbox={"facecolor": HIGHLIGHT, "edgecolor": "none",
                       "boxstyle": "round,pad=0.3"} if highlight else None)
    ax.set_xlabel(f"{'Negative ' if sign < 0 else ''}{label} (more diverse →)", fontsize=10)
    if show_ylabel:
        safety_label = ("Overall" if test_dataset == "Avg"
                        else TEST_DATASETS[test_dataset])
        ax.set_ylabel(f"{safety_label} harmful score (%) ↓", fontsize=11)
    ax.margins(x=0.18)
    if len(y):
        span = max(float(np.ptp(y)), 1.0)
        # Reserve room for the statistics above every annotated point.
        ax.set_ylim(max(0, y.min() - span * 0.15),
                    y.max() + span * (0.18 + 0.10 * len(summary)))
    style_axis(ax)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def plotted_methods(methods):
    """Keep the figure methods, falling back to all if --corr selected only others."""
    return [method for method in methods if method in PLOT_METHODS] or list(methods)


def plot_scatter(scores, correlations, title, context, path, group, test_dataset="Avg"):
    """Draw one panel per metric of a lexical, semantic or structural group."""
    correlations = correlations.loc[
        correlations["method"].isin(plotted_methods(correlations["method"].unique()))
    ]
    name, color = GROUPS[group]
    metrics = [metric for metric in FIGURE_METRICS if metric.startswith(f"{group}_")]
    columns = min(len(metrics), GRID_COLUMNS)
    rows = -(-len(metrics) // columns)
    width = max(columns * 4.6, 10)
    # Shade the two metrics whose mean coefficient is most negative (more diversity,
    # less harm) and bold the lowest.
    group_correlations = correlations.loc[correlations["metric"].isin(metrics)]
    ranked = group_correlations.groupby("metric")["coefficient"].mean().dropna().sort_values()
    top_two = list(ranked.index[:2])
    fig = plt.figure(figsize=(width, 1.8 + 4.3 * rows), layout="constrained")
    for index, ax in enumerate(fig.subplots(rows, columns, squeeze=False).ravel()):
        if index < len(metrics):
            plot_scatter_panel(ax, scores, correlations, metrics[index], test_dataset,
                               show_ylabel=index % columns == 0,
                               bold=metrics[index] in top_two[:1],
                               highlight=metrics[index] in top_two)
        else:
            ax.set_visible(False)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, markersize=8,
               label="Subcategory"),
        Line2D([], [], color=color, alpha=0.65, linewidth=1.5,
               label="Least-squares fit on correlated subcategories"),
    ]
    if (~correlation_mask(scores)).any():
        handles.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="none",
                              markeredgecolor="0.3", markersize=8,
                              label="Excluded from correlation (benign or --exclude)"))
    fig.legend(handles=handles, loc="outside lower center",
               ncols=len(handles) if width > 12 else 1, fontsize=11, frameon=False)
    counts = group_correlations["n_subcategories"]
    coverage = (f"Correlations across {counts.iloc[0]} subcategories"
                if counts.nunique() == 1 else
                f"Correlations across {counts.min()}–{counts.max()} subcategories")
    methods = "/".join(method.title() for method in group_correlations["method"].unique())
    coverage += f"; shaded = two lowest mean {methods} correlations, bold = lowest"
    if test_dataset != "Avg":
        context += f"; test: {TEST_DATASETS[test_dataset]}"
    # Wrap the run details so narrow figures (e.g. two lexical panels) keep them visible.
    details = f"{textwrap.fill(context, width=int(width * 7))}\n{coverage}"
    fig.suptitle(f"{title}: {name.lower()} diversity vs. post-training harmful score\n{details}",
                 fontsize=16)
    save_figure(fig, path)


def plot_correlations(table, metric, method, context, path):
    scores = table.loc[(table["metric"] == metric) & (table["method"] == method)]
    x = np.arange(len(scores), dtype=float)
    colors = ["#999999" if model == "Average" else METRICS[metric][2]
              for model in scores["model"]]
    fig, ax = plt.subplots(figsize=(max(12, len(scores) * 1.4), 7))
    bars = ax.bar(x, scores["coefficient"], 0.6, color=colors, zorder=3)
    for bar, row in zip(bars, scores.itertuples()):
        value = row.coefficient
        ax.annotate(f"{value:+.3f}" if np.isfinite(value) else "undefined",
                    (bar.get_x() + bar.get_width() / 2, value if np.isfinite(value) else 0),
                    xytext=(0, 5 if not np.isfinite(value) or value >= 0 else -5),
                    textcoords="offset points", ha="center",
                    va="bottom" if not np.isfinite(value) or value >= 0 else "top", fontsize=11)
    labels = scores["model"].tolist()
    ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.axhline(0, color="0.3", linewidth=1)
    if "Average" in scores["model"].values:
        ax.axvline(len(scores) - 1.5, color="0.65", linestyle="--", linewidth=1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.set_ylabel(f"{method.title()} coefficient", fontsize=18)
    label, sign, _ = METRICS[metric]
    ax.set_title(f"Diversity ({'negative ' if sign < 0 else ''}{label}) vs. overall harmful score",
                 fontsize=23, pad=20)
    counts = scores["n_subcategories"]
    coverage = (f"Correlations across {counts.iloc[0]} subcategories"
                if counts.nunique() == 1 else
                f"Correlations across {counts.min()}–{counts.max()} subcategories (varies by model)")
    ax.set_xlabel(f"{context}\n{coverage}; outliers excluded", fontsize=14, labelpad=12)
    style_axis(ax)
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="?", help="Diversity CSV path")
    parser.add_argument("--csv", type=Path, dest="csv_option", help="Alternative to positional CSV")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        help="One folder for figures and CSVs; default: figs/ and csv/ of the "
                             "input's embedding-model folder")
    parser.add_argument("--suffix",
                        help="Safety result folder suffix, e.g. 1800_wildguardmix")
    parser.add_argument("--dataset", help="Training dataset; inferred from CSV filename")
    parser.add_argument("--embedder", default="bge-m3",
                        help="Embedding-model folder used when no CSV is given (default: bge-m3)")
    parser.add_argument("--train-size", type=int,
                        help="Safety-run training size; defaults to CSV scored_on")
    parser.add_argument("--corr", choices=[*CORRELATIONS, "all"], default="all")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Additional subcategories to exclude from correlations, "
                             "while keeping them plotted and in CSVs; benign is always excluded")
    parser.add_argument("--all-plots", action="store_true",
                        help="Also draw per-metric bar charts, per-model scatter plots, "
                             "correlation bar charts and per-test-dataset average plots")
    args = parser.parse_args()
    if args.suffix is not None:
        args.suffix = args.suffix.lstrip("_")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.suffix):
            parser.error("--suffix must be a nonempty folder-name suffix without path separators")
    if args.csv and args.csv_option:
        parser.error("Supply the CSV either positionally or with --csv")
    csv_path = args.csv or args.csv_option
    if csv_path is None:
        dataset = (args.dataset or "wildguardmix").removesuffix("_train")
        csv_path = args.results_dir / "diversity" / args.embedder / "csv" / (
            f"{dataset}_train_diversity_{args.train_size or 800}_1.0.csv"
        )
    try:
        diversity = load_diversity(csv_path)
        dataset = args.dataset or csv_path.stem.split("_diversity")[0]
        dataset = dataset.removesuffix("_train").removesuffix("_test")
        size = args.train_size
        if args.suffix and re.match(r"^[1-9][0-9]*_", args.suffix):
            suffix_size = int(args.suffix.split("_", 1)[0])
            if size is not None and size != suffix_size:
                raise ValueError("--train-size must match the training size in --suffix")
            size = suffix_size
        if size is None:
            sizes = pd.to_numeric(diversity["scored_on"], errors="raise").unique()
            if len(sizes) != 1 or not np.isfinite(sizes[0]) or sizes[0] != int(sizes[0]):
                raise ValueError("CSV must have a single scored_on size; specify --train-size")
            size = int(sizes[0])
        if size <= 0:
            raise ValueError("--train-size must be positive")
        safety_by_test = {
            test: load_safety(args.results_dir, dataset, size,
                              set(diversity["abbr"]) - {"overall"}, args.suffix,
                              test_dataset=test)
            for test in ["Avg", *TEST_DATASETS]
        }
        safety = safety_by_test["Avg"]
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    # results/diversity/<model>/csv/<CSV> -> figures in <model>/figs, CSVs in <model>/csv.
    grouped = csv_path.parent.name == "csv"
    embedder = csv_path.parent.parent.name if grouped else None
    fig_dir = args.output_dir or (csv_path.parent.parent if grouped else csv_path.parent) / "figs"
    csv_dir = args.output_dir or csv_path.parent
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    context_parts = [dataset]
    if embedder:
        context_parts.append(f"embeddings: {embedder}")
    if args.suffix:
        context_parts.append(f"results: {args.suffix}")
    if "scored_on" in diversity and diversity["scored_on"].nunique() == 1:
        context_parts.append(f"{int(diversity['scored_on'].iloc[0]):,} diversity examples")
    if {"subcategory_count", "scored_on"} <= set(diversity):
        fractions = diversity["subcategory_count"] / diversity["scored_on"]
        if fractions.nunique() == 1:
            context_parts.append(f"harmful fraction {fractions.iloc[0]:g}")
    context = ", ".join(context_parts)
    if args.all_plots:
        for metric in FIGURE_METRICS:
            plot_diversity(diversity, metric, context,
                           fig_dir / f"{dataset}_diversity_{metric.rstrip('↑↓')}.png")
    if safety.empty:
        print(f"No {dataset} safety runs with training size {size}; saved diversity plots only. "
              "Check --suffix, --results-dir, or --train-size for the intended runs.")
        return

    pairs = join_scores(diversity, safety)
    excluded_abbrs = {"benign", *(abbr.lower() for abbr in args.exclude)}
    pairs["included_in_correlation"] = ~pairs["abbr"].str.lower().isin(excluded_abbrs)
    print(f"Matched {len(safety)} runs across {safety['model'].nunique()} models "
          f"and {safety['abbr'].nunique()} subcategories")
    methods = list(CORRELATIONS) if args.corr == "all" else [args.corr]
    table = correlation_table(pairs, methods)
    table.to_csv(csv_dir / f"{dataset}_diversity_safety_correlations.csv", index=False)
    context += f"; safety training size {size}"
    for model, scores in pairs.groupby("model", sort=False):
        if model != "Average" and not args.all_plots:
            continue
        title = model if model != "Average" else f"Average across {safety['model'].nunique()} models"
        for group in GROUPS:
            plot_scatter(scores, table.loc[table["model"] == model], title, context,
                         fig_dir / f"{dataset}_{model.lower()}_{group}_diversity_vs_harmful.png",
                         group)
    print(f"Saved figures in {fig_dir} and correlations in {csv_dir}")
    if not args.all_plots:
        return

    pairs.to_csv(csv_dir / f"{dataset}_diversity_safety_scores.csv", index=False)
    for metric in FIGURE_METRICS:
        for method in plotted_methods(methods):
            plot_correlations(
                table, metric, method, context,
                fig_dir / f"{dataset}_corr_{metric.rstrip('↑↓')}_vs_harmful_{method}.png",
            )
    test_pairs, test_correlations = [], []
    for test in TEST_DATASETS:
        if safety_by_test[test].empty:
            continue
        joined = join_scores(diversity, safety_by_test[test])
        average = joined.loc[joined["model"] == "Average"].copy()
        if average.empty:
            continue
        average["included_in_correlation"] = ~average["abbr"].str.lower().isin(excluded_abbrs)
        correlations = correlation_table(average, methods)
        for group in GROUPS:
            plot_scatter(
                average, correlations,
                f"Average across {safety_by_test[test]['model'].nunique()} models",
                context, fig_dir / f"{dataset}_average_{group}_diversity_vs_harmful_{test}.png",
                group, test_dataset=test,
            )
        test_pairs.append(average.assign(test_dataset=test))
        test_correlations.append(correlations.assign(test_dataset=test))
    if test_pairs:
        pd.concat(test_pairs, ignore_index=True).to_csv(
            csv_dir / f"{dataset}_average_diversity_safety_scores_by_test.csv", index=False,
        )
        pd.concat(test_correlations, ignore_index=True).to_csv(
            csv_dir / f"{dataset}_average_diversity_safety_correlations_by_test.csv", index=False,
        )


if __name__ == "__main__":
    main()

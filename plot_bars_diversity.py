"""Plot a diversity CSV and its relationship with post-training safety.

Example (run from any directory):
    python sa_data/plot_bars_diversity.py \
        results/diversity/wildguardmix_train_diversity_800_1.0.csv \
        --suffix 800_vanilla_benign --corr all

The three metrics are Self-BLEU, POS n-gram diversity, and Vendi score.
Bar charts display Self-BLEU and POS diversity as percentages and Vendi in
its original units, with one decimal place and a zoomed linear y-axis.
CSV exports retain the original metric units.
Safety uses the precomputed Avg harmful_score_pct (lower is safer), matching
plot_bars_overall.py. Correlations use negative Self-BLEU so larger always
means more diverse. Average correlations are computed against equally weighted
model-mean harmful scores on subcategories available for every model; they
are not averages of correlation coefficients. The pooled overall diversity
reference is excluded from correlations. The benign subcategory and any --exclude
subcategories are plotted but excluded from correlation coefficients and fitted
lines. CSV exports contain plotted pairs
and coefficients, including sample counts and p-values.
Run names encode dataset, subcategory, and size, but not the mixture fraction;
use --suffix to select grouped result folders for the intended configuration.
"""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
# Column: (display label, direction of increasing diversity, color).
METRICS = {
    "self_bleu": ("Self-BLEU", -1, "#D55E00"),
    "pos_ngram_diversity": ("POS n-gram diversity", 1, "#0072B2"),
    "vendi_score": ("Vendi score", 1, "#009E73"),
}
CORRELATIONS = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}


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


def load_safety(results_dir, dataset, train_size, abbreviations, suffix=None):
    """Read grouped results matching a suffix, or legacy flat runs by size."""
    pattern = re.compile(rf"^(.+)_{re.escape(dataset)}_(.+)_{train_size}$")
    group_suffix = "_" + (suffix or f"{train_size}_vanilla_benign")
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
        scores = data.loc[data["dataset"] == "Avg", "harmful_score_pct"]
        if len(scores) != 1:
            raise ValueError(f"{path}: expected exactly one Avg row")
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
                    "model": model, "metric": metric, "method": method,
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
    # Display bounded metrics as percentages; Vendi remains in its native units.
    scale = 100 if metric in {"self_bleu", "pos_ngram_diversity"} else 1
    values = ordered[metric] * scale
    bars = ax.bar(x, values, 0.65, color=colors, zorder=3)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, ordered["abbr"], rotation=45, ha="right", rotation_mode="anchor")
    if len(reference) and len(subs):
        ax.axvline((x[0] + x[1]) / 2, color="0.65", linestyle="--", linewidth=1)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.5)
    ax.set_ylim(max(0, low - span * 0.15), high + span * 0.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel("Score (%)" if scale == 100 else "Score", fontsize=18)
    ax.set_xlabel(f"Training subcategory ({context})\nIncreasing diversity from left to right",
                  fontsize=16, labelpad=12)
    ax.set_title(f"Diversity measured by {label} ({'↑' if sign > 0 else '↓'})",
                 pad=20, fontsize=23)
    style_axis(ax)
    save_figure(fig, path)


def plot_scatter(scores, correlations, title, context, path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (metric, (label, sign, color)) in zip(axes, METRICS.items()):
        x, y, abbreviations = finite_pairs(scores, metric)
        excluded_abbrs = set(scores.loc[~correlation_mask(scores), "abbr"])
        excluded = np.array([abbr in excluded_abbrs for abbr in abbreviations], dtype=bool)
        ax.scatter(x[~excluded], y[~excluded], color=color, s=48, zorder=3)
        if excluded.any():
            ax.scatter(x[excluded], y[excluded], facecolors="none", edgecolors="0.3",
                       marker="D", s=70, zorder=4, label="excluded from correlation (outlier)")
            ax.legend(loc="lower right", fontsize=9)
        for xi, yi, abbr in zip(x, y, abbreviations):
            ax.annotate(abbr, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
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
        ax.text(0.03, 0.97, "\n".join(summary), transform=ax.transAxes, va="top",
                fontsize=11, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_title(label, fontsize=20, pad=12)
        ax.set_xlabel(
            f"{'Negative ' if sign < 0 else ''}{label} (more diverse →)\n"
            f"Correlation across {len(corr_x)} subcategories (outliers excluded)", fontsize=14,
        )
        ax.set_ylabel("Overall harmful score (%) ↓", fontsize=14)
        ax.margins(x=0.18)
        if len(y):
            span = max(float(np.ptp(y)), 1.0)
            # Reserve room for the statistics above every annotated point.
            ax.set_ylim(max(0, y.min() - span * 0.15),
                        y.max() + span * (0.18 + 0.10 * len(stats)))
        style_axis(ax)
    fig.suptitle(f"{title}: diversity vs. post-training harmful score\n{context}", fontsize=21)
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
                        help="Default: ROOT/figs/diversity/<input CSV stem>; --suffix is appended")
    parser.add_argument("--suffix",
                        help="Result folder suffix, e.g. 800_vanilla_benign; also appended to output folder")
    parser.add_argument("--dataset", help="Training dataset; inferred from CSV filename")
    parser.add_argument("--train-size", type=int,
                        help="Safety-run training size; defaults to CSV scored_on")
    parser.add_argument("--corr", choices=[*CORRELATIONS, "all"], default="pearson")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Additional subcategories to exclude from correlations, "
                             "while keeping them plotted and in CSVs; benign is always excluded")
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
        csv_path = args.results_dir / "diversity" / (
            f"{dataset}_train_diversity_{args.train_size or 800}_1.0.csv"
        )
    try:
        diversity = load_diversity(csv_path)
        dataset = args.dataset or csv_path.stem.split("_diversity")[0]
        dataset = dataset.removesuffix("_train")
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
        safety = load_safety(args.results_dir, dataset, size,
                             set(diversity["abbr"]) - {"overall"}, args.suffix)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    output_dir = args.output_dir or ROOT / "figs" / "diversity" / csv_path.stem
    if args.suffix:
        output_dir = output_dir.with_name(f"{output_dir.name}_{args.suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    context_parts = [dataset]
    if args.suffix:
        context_parts.append(f"results: {args.suffix}")
    if "scored_on" in diversity and diversity["scored_on"].nunique() == 1:
        context_parts.append(f"{int(diversity['scored_on'].iloc[0]):,} diversity examples")
    if {"subcategory_count", "scored_on"} <= set(diversity):
        fractions = diversity["subcategory_count"] / diversity["scored_on"]
        if fractions.nunique() == 1:
            context_parts.append(f"harmful fraction {fractions.iloc[0]:g}")
    context = ", ".join(context_parts)
    for metric in METRICS:
        plot_diversity(diversity, metric, context, output_dir / f"diversity_{metric}.png")
    diversity.to_csv(output_dir / "diversity_scores.csv", index=False)
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
    pairs.to_csv(output_dir / "diversity_safety_scores.csv", index=False)
    table.to_csv(output_dir / "diversity_safety_correlations.csv", index=False)
    context += f"; safety training size {size}"
    for model, scores in pairs.groupby("model", sort=False):
        title = model if model != "Average" else f"Average across {safety['model'].nunique()} models"
        plot_scatter(scores, table.loc[table["model"] == model], title, context,
                     output_dir / f"{model.lower()}_diversity_vs_harmful.png")
    for metric in METRICS:
        for method in methods:
            plot_correlations(table, metric, method, context,
                              output_dir / f"corr_{metric}_vs_harmful_{method}.png")
    print(f"Saved score and correlation CSVs in {output_dir}")


if __name__ == "__main__":
    main()

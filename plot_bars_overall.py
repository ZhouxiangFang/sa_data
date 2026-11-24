"""Bar charts of the harmful, overrefusal and IF scores of the starting models and the runs.

    python plot_bars_overall.py --seeds 43 86 129
    python plot_bars_overall.py --results_dir ../ablations/refusal --seeds 43

For each model and their average: the official instruct model and the git20k starting model
(from --baseline_dir, evaluated by range/run.sh's base stage), then every trained group (from
<results_dir>/seed<S>/), the random and evenly baselines first and the subcategories by harmful
score, most harmful first (--ascending: all groups ranked together, least harmful first). Bars:
harmful (the Avg row: mean over the five harm test sets), overrefusal (mean of XSTest and
OR-Bench-Hard) and IF (mean IFBench / IFEval prompt-level loose accuracy). Several seeds are
averaged; the average gives each model equal weight. The same figures per harm test set go to
<test>/ subfolders.
Writes to <results_dir>/figs/seed<S>/ (several seeds: seeds<S1>-<S2>.../): <model>.png,
average.png, <test>/*.png and scores.csv (every plotted number, and each seed's).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval import result_dirs
from helper import (BASELINE_ABBRS, DATASET, DEFAULT_NUM_TRAIN, DEFAULT_SEED, GIT20K_MODELS,
                    ROOT, checkpoint_name, safety_test_datasets, seed_dir)

# Column: (legend label, color: categorical slots 2, 1, 3 of the reference palette).
BARS = {
    "harmful": ("Harmful ↓", "#eb6834"),
    "overrefusal": ("Overrefusal (XSTest, OR-Bench-Hard) ↓", "#2a78d6"),
    "if": ("IF prompt loose (IFBench, IFEval) ↑", "#1baf7a"),
}
INSTRUCT, START = "official instruct", "git20k start"


def run_scores(output_dir, name):
    """Scores (%) of one evaluated model: harmful (Avg and per test set), overrefusal, IF."""
    safety_dir, if_dir, prefix = result_dirs(output_dir, name)
    harmful = pd.read_csv(Path(safety_dir) / f"{prefix}_overall.csv").set_index("dataset")
    overrefusal = pd.read_csv(Path(safety_dir) / f"{prefix}_overrefusal.csv").set_index("dataset")
    loose = pd.read_csv(Path(if_dir) / f"{prefix}_if_summary.csv").set_index("benchmark")
    scores = {"harmful": harmful.loc["Avg", "harmful_score_pct"],
              **{test: harmful.loc[test, "harmful_score_pct"] for test in safety_test_datasets},
              "overrefusal": overrefusal.loc["Avg", "overrefusal_score_pct"],
              "ifbench": 100 * loose.loc["IFBench", "prompt_loose"],
              "ifeval": 100 * loose.loc["IFEval", "prompt_loose"]}
    return {**scores, "if": (scores["ifbench"] + scores["ifeval"]) / 2}


def trained_groups(results_dir, model, num_train, seed):
    """Groups with an overall CSV in <results_dir>/seed<seed>/<model>_<num_train>_wildguardmix/."""
    run = Path(results_dir) / seed_dir(seed) / f"{model}_{num_train}_{DATASET}"
    prefix = f"{DATASET}_"
    return {path.parent.name.removeprefix(prefix)
            for path in run.glob(f"{prefix}*/{prefix}*_overall.csv")}


def load_scores(results_dir, baseline_dir, models, num_train, seeds):
    """One row per model, version (official instruct, git20k start or group) and seed."""
    found = [trained_groups(results_dir, model, num_train, seed)
             for model in models for seed in seeds]
    groups = set.intersection(*found)
    if set.union(*found) - groups:
        print(f"Left out, not evaluated for every model and seed: "
              f"{' '.join(sorted(set.union(*found) - groups))}")
    if not groups:
        raise SystemExit(f"No group evaluated for every model and seed in {results_dir}")
    rows = []
    for model in models:
        base = model.removesuffix("_git20k")
        baselines = {INSTRUCT: f"{base}-ins", START: model}
        for version, name in baselines.items():
            rows += [{"model": base, "version": version, "seed": seed,
                      **run_scores(baseline_dir, name)} for seed in seeds]
        for seed in seeds:
            for abbr in groups:
                name = checkpoint_name(model, num_train, abbr, seed)
                rows.append({"model": base, "version": abbr, "seed": seed,
                             **run_scores(results_dir, name)})
    return pd.DataFrame(rows)


def plot(scores, harmful, title, path, ascending=False):
    """scores: one row per version (index). harmful: the column the harmful bars show."""
    trained = scores.drop(index=[INSTRUCT, START])
    if ascending:
        groups = [[INSTRUCT, START], list(trained.sort_values(harmful).index)]
    else:
        subcategories = trained.drop(index=list(BASELINE_ABBRS))
        groups = [[INSTRUCT, START], list(BASELINE_ABBRS),
                  list(subcategories.sort_values(harmful, ascending=False).index)]
    versions = [version for group in groups for version in group]
    scores = scores.loc[versions]
    x = np.arange(len(versions), dtype=float)
    boundaries = np.cumsum([len(group) for group in groups])[:-1]
    for boundary in boundaries:
        x[boundary:] += 0.6  # a gap between the groups
    fig, ax = plt.subplots(figsize=(max(12, len(versions) * 1.0), 6.5))
    for (column, (label, color)), shift in zip(BARS.items(), (-0.29, 0, 0.29)):
        values = scores[harmful if column == "harmful" else column]
        if column == "harmful" and harmful != "harmful":
            label = f"{harmful} harmful ↓"
        bars = ax.bar(x + shift, values, 0.27, color=color, label=label, zorder=3,
                      alpha=1 if column == "harmful" else 0.35)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8 if column == "harmful" else 6.5,
                     color="#1f1f1f" if column == "harmful" else "#5f6368")
    for boundary in boundaries:
        ax.axvline((x[boundary - 1] + x[boundary]) / 2, color="#c8c8c8", linewidth=1)
    ax.set_xticks(x, versions, rotation=40, ha="right", rotation_mode="anchor", fontsize=11)
    if ascending:
        for tick, version in zip(ax.get_xticklabels(), versions):
            tick.set_fontweight("bold" if version in BASELINE_ABBRS else "normal")
    ax.set_ylim(0, min(105, np.ceil(scores[[harmful, "overrefusal", "if"]].max().max() / 10)
                       * 10 + 10))
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title(title, fontsize=15, pad=36, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, frameon=False, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--baseline_dir", type=Path, default=ROOT / "results",
                        help="eval.py --output_dir of the instruct and git20k models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--num_train", type=int, default=DEFAULT_NUM_TRAIN)
    parser.add_argument("--models", nargs="+", default=list(GIT20K_MODELS))
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    output_dir = args.output_dir or args.results_dir / "figs" / (
        seed_dir(seeds[0]) if len(seeds) == 1 else "seeds" + "-".join(map(str, seeds)))
    per_seed = load_scores(args.results_dir, args.baseline_dir, args.models, args.num_train,
                           seeds)
    columns = [column for column in per_seed if column not in ("model", "version", "seed")]
    means = per_seed.groupby(["model", "version"], sort=False)[columns].mean()
    average = means.groupby(level="version", sort=False).mean()
    note = f"seed {seeds[0]}" if len(seeds) == 1 else f"mean of seeds {', '.join(map(str, seeds))}"
    tables = {model: means.loc[model] for model in means.index.unique("model")}
    tables[f"Average of {len(tables)} models"] = average
    for test in ["harmful", *safety_test_datasets]:
        folder = output_dir if test == "harmful" else output_dir / test
        folder.mkdir(parents=True, exist_ok=True)
        for title, scores in tables.items():
            name = "average" if title.startswith("Average") else title
            heading = f"{title} ({note})" + ("" if test == "harmful" else f": {test} test")
            plot(scores, test, heading, folder / f"{name}.png", args.ascending)
    table = pd.concat([means.reset_index(), average.reset_index().assign(model="Average"),
                       per_seed.assign(seed=per_seed["seed"].astype(str))], ignore_index=True)
    table.insert(2, "seed", table.pop("seed").fillna("mean"))
    table.to_csv(output_dir / "scores.csv", index=False, float_format="%.2f")
    print(f"Saved figures and scores.csv in {output_dir}")


if __name__ == "__main__":
    main()

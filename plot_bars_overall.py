"""Bar charts of the harmful, overrefusal and IF scores of the starting models and the runs.

    python plot_bars_overall.py --seeds 43 86 129
    python plot_bars_overall.py --results_dir ../results/correlations/refusal --seeds 43
    python plot_bars_overall.py --results_dir ../results/selection/wildjailbreak --seeds 43

For each model and their average: the official instruct model and the git20k starting model
(from --baseline_dir, evaluated by range/run.sh's base stage), then every trained group of the
setting (<results_dir>/seed<S>/<model>/<group>.csv; stage 2: the selection methods) ranked by
harmful score, least harmful first (the random and evenly baselines in bold). Bars: harmful (the Avg row: mean
over the five harm test sets), overrefusal (mean of XSTest and OR-Bench-Hard) and IF (mean
IFBench / IFEval prompt-level loose accuracy). Several seeds are averaged; the average gives each
model equal weight. The same figures per harm test set go to <test>/ subfolders.
Writes to the setting's figs/ folder (helper.figs_dir, e.g. figs/correlations/default/),
seed<S>/ (several seeds: seeds<S1>-<S2>.../): <model>.png, average.png, <test>/*.png and
scores.csv (every plotted number, and each seed's); and the self-trained vs official instruct
comparison to figs/baselines/instruct_vs_git20k.png and .csv.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval import evaluated, read_scores, result_paths, run_csv
from helper import (BASELINE_ABBRS, BASELINES_DIR, DEFAULT_SEED, GIT20K_MODELS, STAGE1_DIR,
                    figs_dir, safety_test_datasets, seed_dir)

# Column: (legend label, color: categorical slots 2, 1, 3 of the reference palette).
BARS = {
    "harmful": ("Harmful ↓", "#eb6834"),
    "overrefusal": ("Overrefusal (XSTest, OR-Bench-Hard) ↓", "#2a78d6"),
    "if": ("IF prompt loose (IFBench, IFEval) ↑", "#1baf7a"),
}
INSTRUCT, START = "official instruct", "git20k start"


def run_scores(path):
    """Scores (%) of one evaluated model: harmful (Avg and per test set), overrefusal, IF."""
    scores = read_scores(path)
    return {"harmful": scores["harmful", "Avg"],
            **{test: scores["harmful", test] for test in safety_test_datasets},
            "overrefusal": scores["overrefusal", "Avg"],
            "ifbench": scores["if_prompt_loose", "IFBench"],
            "ifeval": scores["if_prompt_loose", "IFEval"],
            "if": scores["if_prompt_loose", "Avg"]}


def trained_groups(results_dir, model, seed):
    """Groups with a complete scores CSV in <results_dir>/seed<seed>/<model>/."""
    return {path.stem for path in (Path(results_dir) / seed_dir(seed) / model).glob("*.csv")
            if evaluated(path)}


def load_scores(results_dir, baseline_dir, models, seeds):
    """One row per model, version (official instruct, git20k start or group) and seed."""
    found = [trained_groups(results_dir, model, seed) for model in models for seed in seeds]
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
            scores = run_scores(result_paths(baseline_dir, name)[0])
            rows += [{"model": base, "version": version, "seed": seed, **scores}
                     for seed in seeds]
        for seed in seeds:
            for abbr in groups:
                rows.append({"model": base, "version": abbr, "seed": seed,
                             **run_scores(run_csv(results_dir, model, abbr, seed))})
    return pd.DataFrame(rows)


def plot(scores, harmful, title, path):
    """scores: one row per version (index). harmful: the column the harmful bars show (the runs
    are ranked by it, least harmful first)."""
    trained = scores.drop(index=[INSTRUCT, START])
    groups = [[INSTRUCT, START], list(trained.sort_values(harmful, kind="stable").index)]
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


def plot_instruct(baseline_dir, models):
    """Self-trained (git20k) vs official instruct models: IFEval and IFBench prompt-level loose
    accuracy per model and their mean, to figs_dir(baseline_dir)/instruct_vs_git20k.png, and
    every IF score with the harmful and overrefusal Avg to the .csv beside it."""
    rows = []
    for model in models:
        base = model.removesuffix("_git20k")
        for version, name in ((INSTRUCT, f"{base}-ins"), ("self-trained (git20k)", model)):
            scores = read_scores(result_paths(baseline_dir, name)[0])
            rows.append({"model": base, "version": version,
                         **{f"{benchmark}_{score}": scores[f"if_{score}", benchmark]
                            for benchmark in ("IFEval", "IFBench")
                            for score in ("prompt_loose", "prompt_strict", "instruction_loose",
                                          "instruction_strict")},
                         "harmful": scores["harmful", "Avg"],
                         "overrefusal": scores["overrefusal", "Avg"]})
    table = pd.DataFrame(rows)
    table = pd.concat([table, table.groupby("version", as_index=False, sort=False)
                       .mean(numeric_only=True).assign(model="Average")], ignore_index=True)
    output_dir = figs_dir(baseline_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "instruct_vs_git20k.csv", index=False, float_format="%.2f")
    names = list(dict.fromkeys(table["model"]))
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, benchmark in zip(axes, ("IFEval", "IFBench")):
        for shift, (version, color) in zip((-0.2, 0.2), ((INSTRUCT, "#2a78d6"),
                                                         ("self-trained (git20k)", "#eb6834"))):
            values = table[table["version"] == version].set_index("model").loc[
                names, f"{benchmark}_prompt_loose"]
            bars = ax.bar(x + shift, values, 0.38, color=color, label=version, zorder=3)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color="#1f1f1f")
        ax.set_xticks(x, names, fontsize=10)
        ax.set_title(f"{benchmark} prompt-level loose accuracy (%)", fontsize=11, loc="left")
        ax.grid(axis="y", color="#e6e6e6")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "instruct_vs_git20k.png", dpi=180)
    plt.close(fig)
    print(f"Saved {output_dir / 'instruct_vs_git20k.png'} and .csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, default=STAGE1_DIR,
                        help="The setting's scores folder (default: results/correlations/default)")
    parser.add_argument("--baseline_dir", type=Path, default=BASELINES_DIR,
                        help="eval.py --output_dir of the instruct and git20k models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--models", nargs="+", default=list(GIT20K_MODELS))
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    output_dir = args.output_dir or figs_dir(args.results_dir) / (
        seed_dir(seeds[0]) if len(seeds) == 1 else "seeds" + "-".join(map(str, seeds)))
    plot_instruct(args.baseline_dir, args.models)
    per_seed = load_scores(args.results_dir, args.baseline_dir, args.models, seeds)
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
            plot(scores, test, heading, folder / f"{name}.png")
    table = pd.concat([means.reset_index(), average.reset_index().assign(model="Average"),
                       per_seed.assign(seed=per_seed["seed"].astype(str))], ignore_index=True)
    table.insert(2, "seed", table.pop("seed").fillna("mean"))
    table.to_csv(output_dir / "scores.csv", index=False, float_format="%.2f")
    print(f"Saved figures and scores.csv in {output_dir}")


if __name__ == "__main__":
    main()

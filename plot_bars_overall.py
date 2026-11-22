"""Plot harmful, overrefusal and prompt-loose IF scores per model and their mean.

Run from any directory: python sa_data/plot_bars_overall.py
Select a result configuration with --suffix 800_wildguardmix and a training
seed with --seed (default: 43; several seeds, e.g. --seed 43 86 129, are
averaged per model and run, and their figures default to
figs/summary/<baseline>_<suffix>): trained runs are read from <results>/seed<seed>/ (IF: <results>/seed<seed>/if_eval/), while the official and
self-trained instruct baselines, always included, stay in <results>/ and
<results>/if_eval/. --models selects the models (default: llama3.1 mistral olmo2
qwen2.5). Figures default to figs/seed<seed>/<baseline>_<size>_<dataset> (for
example, figs/seed42/git20k_800_wildguardmix), inferred from the loaded runs, or
figs/seed<seed>/<baseline>_<suffix> when selected.
Use --output-dir to choose an explicit folder.
Runs trained on the benign subcategory are left out (EXCLUDED_ABBRS): they
replace the harmful prompts with benign ones, so they are not safety training.
Harmful scores use the precomputed Avg row (a macro-average over safety
datasets); overrefusal scores use the Avg row of <run>_overrefusal.csv (a
macro-average over XSTest and OR-Bench-Hard); IFBench and IFEval prompt_loose
fractions are converted to percent and averaged equally. Official and
self-trained instruct baselines come first, then the random and evenly
training-data baselines, followed by subcategories ranked by descending harmful
score; --ascending instead ranks random, evenly and the subcategories together,
least harmful first (instruct baselines stay first), with bold mix labels. Pale blue and gray bars beside the harmful bars show the secondary
overrefusal and IF metrics. The average figure gives each model equal weight.
X labels drop the embedding-model suffix of select_data.py subsets
(semantic_kmeans_qwen3-embedding-0.6b -> semantic_kmeans) unless two would match.
The same figures are drawn per safety test dataset in <output>/<test>/, with
that dataset's harmful score in place of the Avg row.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper import DEFAULT_SEED, insert_seed_dir, seed_dir
from calculate_diversity import EMBEDDING_MODEL_IDS, embedding_name


ROOT = Path(__file__).resolve().parent.parent
# Project layout. results/ holds the safety results of the default 1:1:1
# (harmful : benign : general) runs and the instruct baselines; every other
# training setting is in ablations/<setting>/. Trained runs of each seed are in
# a seed<seed>/ subfolder of these (results/seed42/...), and every folder keeps
# its IF results in its own if_eval/ (results/if_eval/ for the baselines,
# results/seed42/if_eval/ for seed 42's runs). Diversity results are in
# diversity/ (per-seed: diversity/seed42/).
RESULTS_DIR = ROOT / "results"
ABLATIONS_DIR = ROOT / "ablations"
DIVERSITY_DIR = ROOT / "diversity"
MODELS = ("llama3", "llama3.1", "mistral", "olmo2", "qwen2.5", "qwen3")
DEFAULT_MODELS = ("llama3.1", "mistral", "olmo2", "qwen2.5")
METRICS = {
    "harmful_score_pct": ("Overall harmful ↓", "#D55E00"),
    "overrefusal_score_pct": ("Mean XSTest / OR-Bench-Hard overrefusal ↓", "#0072B2"),
    "if_prompt_loose_pct": ("Mean IFBench / IFEval prompt_loose ↑", "#999999"),
}
# Safety test datasets in each run's overall CSV, each plotted in its own folder.
TEST_DATASETS = {
    "aegis": "Aegis",
    "wildguardmix": "WildGuardMix",
    "wildjailbreak": "WildJailbreak",
    "ailuminate": "AILuminate",
    "gretel": "Gretel",
}
# Mixed-subcategory training baselines (helper.BASELINE_ABBRS), shown in this
# order between the instruct models and the ranked subcategories.
BASELINE_ABBRS = ("random", "evenly")
# Training subcategories left out of every figure and score table; benign runs
# have no harmful prompts (align.py).
EXCLUDED_ABBRS = ("benign",)
# Embedding-model suffixes of select_data.py subset names
# (semantic_kmeans_qwen3-embedding-0.6b), left out of the x labels.
EMBEDDER_SUFFIXES = tuple(f"_{embedding_name(model)}" for model in EMBEDDING_MODEL_IDS)


def training_subcategory(version):
    """Training subcategory of a DATASET_SUBCATEGORY_SIZE version label."""
    return version.split("_", 1)[1].rsplit("_", 1)[0]


def short_labels(subcategories):
    """X labels without the embedding-model suffix, unless that makes two equal."""
    short = [
        next((name.removesuffix(suffix) for suffix in EMBEDDER_SUFFIXES
              if name.endswith(suffix)), name)
        for name in subcategories
    ]
    return short if len(set(short)) == len(short) else list(subcategories)


def discover_runs(results_dir, model, suffix=None, seed=DEFAULT_SEED):
    """Map shared x-axis version labels to result directories relative to results_dir.

    The instruct baselines are in results_dir; trained runs in results_dir/seed<seed>.
    """
    runs = {"-ins": f"{model}-ins"}
    # Existing files use _git20k; also accept the user's -20k convention.
    candidates = [f"{model}{suffix}" for suffix in ("_git20k", "-20k", "-git20k")]
    candidates = [name for name in candidates if (results_dir / name).is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"{model}: expected one 20k baseline, found {candidates}")
    runs["-20k"] = candidates[0]
    prefix = candidates[0] + "_"
    seed_results = results_dir / seed_dir(seed)
    paths = sorted(seed_results.iterdir()) if seed_results.is_dir() else []
    for path in paths:
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        if suffix is not None and not path.name.endswith("_" + suffix):
            continue
        version = path.name[len(prefix):]
        if (path / f"{path.name}_overall.csv").is_file():
            selected = {version: f"{seed_dir(seed)}/{path.name}"}
        else:
            # Grouped layout: MODEL_BASELINE_SIZE_CONFIG/DATASET_CATEGORY.
            size = version.split("_", 1)[0]
            if not size.isdigit():
                raise ValueError(f"{path}: expected a training size before the configuration")
            selected = {
                f"{category.name}_{size}": f"{seed_dir(seed)}/{path.name}/{category.name}"
                for category in sorted(path.iterdir())
                if category.is_dir()
                and (category / f"{category.name}_overall.csv").is_file()
            }
        for label, run in selected.items():
            if training_subcategory(label) in EXCLUDED_ABBRS:
                continue
            if label in runs:
                raise ValueError(
                    f"{model}: multiple runs for {label!r}; select one with --suffix"
                )
            runs[label] = run
    if len(runs) == 2:
        selection = f" matching suffix {suffix!r}" if suffix is not None else ""
        raise ValueError(
            f"{model}: no subcategory runs{selection} found under {seed_results}"
        )
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


def load_scores(results_dir, if_eval_dir, models, suffix=None, seed=DEFAULT_SEED):
    """Scores of the instruct baselines and the trained runs of one seed.

    if_eval_dir holds the baselines' IF summaries; a trained run's summary is in
    its seed folder (results/if_eval -> results/seed42/if_eval).
    """
    rows = []
    versions = None
    seed_prefix = seed_dir(seed) + "/"
    for model in models:
        runs = discover_runs(results_dir, model, suffix, seed)
        if versions is None:
            versions = list(runs)
        elif set(runs) != set(versions):
            raise ValueError(
                f"{model}: versions differ from {models[0]}; "
                f"missing={sorted(set(versions) - set(runs))}, "
                f"extra={sorted(set(runs) - set(versions))}"
            )
        for version, run in runs.items():
            overall = results_dir / run / f"{Path(run).name}_overall.csv"
            overrefusal = results_dir / run / f"{Path(run).name}_overrefusal.csv"
            summary = (insert_seed_dir(if_eval_dir, seed)
                       / f"{run.removeprefix(seed_prefix)}_if_summary.csv"
                       if run.startswith(seed_prefix)
                       else if_eval_dir / f"{run}_if_summary.csv")
            rows.append({
                "model": model,
                "version": version,
                "run": run,
                "harmful_score_pct": read_score(
                    overall, "dataset", "Avg", "harmful_score_pct"
                ),
                **{
                    f"{test}_harmful_score_pct": read_score(
                        overall, "dataset", test, "harmful_score_pct"
                    )
                    for test in TEST_DATASETS
                },
                "overrefusal_score_pct": read_score(
                    overrefusal, "dataset", "Avg", "overrefusal_score_pct"
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


def load_seed_means(results_dir, if_eval_dir, models, suffix, seeds):
    """Scores averaged over seeds, the versions, and every seed's scores.

    The instruct baselines do not depend on the seed, so their mean is their score.
    """
    frames = []
    versions = None
    for seed in seeds:
        data, seed_versions = load_scores(results_dir, if_eval_dir, models, suffix, seed)
        if versions is None:
            versions = seed_versions
        elif set(seed_versions) != set(versions):
            raise ValueError(
                f"seed {seed}: versions differ from seed {seeds[0]}; "
                f"missing={sorted(set(versions) - set(seed_versions))}, "
                f"extra={sorted(set(seed_versions) - set(versions))}"
            )
        frames.append(data.assign(seed=seed))
    per_seed = pd.concat(frames, ignore_index=True)
    score_columns = [column for column in per_seed if column.endswith("_pct")]
    first_seed = seed_dir(seeds[0]) + "/"
    mean = (
        per_seed.groupby(["model", "version"], sort=False)
        .agg(run=("run", "first"), **{column: (column, "mean") for column in score_columns})
        .reset_index()
    )
    mean["run"] = mean["run"].str.removeprefix(first_seed)
    return mean, versions, per_seed


def default_output_dir(data, versions, suffix=None, seed=DEFAULT_SEED):
    """Infer the figure folder from seed, baseline, training size, and dataset.

    seed may be a list of seeds; several seeds go to figs/summary/.
    """
    baselines = {
        row.run[len(row.model):].lstrip("_-")
        for row in data.loc[data["version"] == "-20k"].itertuples()
    }
    sizes = {
        int(version.rsplit("_", 1)[1])
        for version in versions if version not in ("-ins", "-20k")
    }
    datasets = {
        version.split("_", 1)[0]
        for version in versions if version not in ("-ins", "-20k")
    }
    if len(baselines) != 1 or len(sizes) != 1 or len(datasets) != 1:
        raise ValueError(
            "Cannot infer one output folder from mixed baselines or training "
            "sizes or datasets; specify --output-dir"
        )
    output_suffix = suffix or f"{next(iter(sizes))}_{next(iter(datasets))}"
    seeds = seed if isinstance(seed, (list, tuple)) else [seed]
    folder = seed_dir(seeds[0]) if len(seeds) == 1 else "summary"
    return ROOT / "figs" / folder / f"{next(iter(baselines))}_{output_suffix}"


def plot_bars(scores, versions, title, output_path, harmful_column="harmful_score_pct",
              harmful_label=METRICS["harmful_score_pct"][0], ascending=False):
    """ascending ranks the mixes with the subcategories, least harmful first."""
    baselines = ["-ins", "-20k"]
    trained = [version for version in versions if version not in baselines]
    mixes = [] if ascending else sorted(
        (version for version in trained if training_subcategory(version) in BASELINE_ABBRS),
        key=lambda version: BASELINE_ABBRS.index(training_subcategory(version)),
    )
    subcategories = scores.reindex([
        version for version in trained if version not in mixes
    ]).sort_values(
        harmful_column, ascending=ascending, kind="stable"
    )
    groups = [group for group in (baselines, mixes, subcategories.index.tolist()) if group]
    versions = [version for group in groups for version in group]
    scores = scores.reindex(versions)
    x = np.arange(len(versions), dtype=float)
    # Separate the groups with a gap; a dashed line marks each one below.
    boundaries = np.cumsum([len(group) for group in groups])[:-1]
    for boundary in boundaries:
        x[boundary:] += 0.6
    width = 0.27
    offset = 0.29
    fig, ax = plt.subplots(figsize=(max(12, len(versions) * 1.05), 7))
    label, color = METRICS["harmful_score_pct"]
    bars = ax.bar(
        x - offset, scores[harmful_column], width, label=harmful_label, color=color,
        zorder=3,
    )
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    secondary = []
    for column, shift in (("overrefusal_score_pct", 0), ("if_prompt_loose_pct", offset)):
        label, color = METRICS[column]
        handle = ax.bar(
            x + shift, scores[column], width, label=label, color=color,
            alpha=0.25, edgecolor="none", zorder=2,
        )
        ax.bar_label(handle, fmt="%.1f", padding=4, fontsize=7, color="0.5")
        secondary.append(handle)

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
    labels = short_labels(labels)
    training_info = "; ".join(
        f"{dataset}, {size:,} examples per subcategory"
        for dataset, size in sorted(training_sets)
    )
    ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
    if ascending:
        # Bold the random/evenly mixes, which are ranked among the subcategories.
        for tick, version in zip(ax.get_xticklabels(), versions):
            if version not in baselines and training_subcategory(version) in BASELINE_ABBRS:
                tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=14)
    for boundary in boundaries:
        ax.axvline((x[boundary - 1] + x[boundary]) / 2, color="0.65",
                   linestyle="--", linewidth=1)
    plotted = [harmful_column, "overrefusal_score_pct", "if_prompt_loose_pct"]
    ymax = min(105, np.ceil(scores[plotted].to_numpy().max() / 10) * 10 + 10)
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
        handles=[bars, *secondary], loc="lower center",
        bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=13,
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
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--suffix",
        help="Result folder suffix, e.g. 800_wildguardmix (always includes both instruct baselines)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[DEFAULT_SEED],
        help="Training seed(s) of the runs (results_dir/seed<seed>; default: %(default)s). "
             "Several seeds are averaged per model and run; their figures default to "
             "ROOT/figs/summary/ and their per-seed scores go to seed_scores.csv",
    )
    parser.add_argument(
        "--if-eval-dir", type=Path,
        help="Baselines' IF summary directory (default: RESULTS_DIR/if_eval); trained "
             "runs' summaries are in its seed folder (RESULTS_DIR/seed<seed>/if_eval)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output folder (default: ROOT/figs/seed<seed>/<baseline>_<suffix or size_dataset>)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS),
        help="Models to plot and average (default: %(default)s)",
    )
    parser.add_argument(
        "--ascending", action="store_true",
        help="Rank random, evenly and the subcategories together, least harmful first "
             "(default: mixes first, then subcategories most harmful first)",
    )
    args = parser.parse_args()
    if args.suffix is not None:
        args.suffix = args.suffix.lstrip("_")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.suffix):
            parser.error("--suffix must be a nonempty folder-name suffix without path separators")
    seeds = list(dict.fromkeys(args.seed))
    if_eval_dir = args.if_eval_dir or args.results_dir / "if_eval"
    try:
        if len(seeds) == 1:
            data, versions = load_scores(
                args.results_dir, if_eval_dir, args.models, args.suffix, seeds[0],
            )
            per_seed = None
        else:
            data, versions, per_seed = load_seed_means(
                args.results_dir, if_eval_dir, args.models, args.suffix, seeds,
            )
        output_dir = args.output_dir or default_output_dir(data, versions, args.suffix,
                                                           seeds)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    # Several seeds: say so in every title.
    seed_note = "" if len(seeds) == 1 else (
        f", mean of seeds {', '.join(map(str, seeds))}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        scores = data.loc[data["model"] == model].set_index("version")
        plot_bars(scores, versions, model + seed_note, output_dir / f"{model}_overall.png",
                  ascending=args.ascending)

    score_columns = [column for column in data if column.endswith("_pct")]
    average = data.groupby("version")[score_columns].mean().reindex(versions)
    average_title = f"Average across {len(args.models)} models (equal weight){seed_note}"
    plot_bars(average, versions, average_title, output_dir / "average_overall.png",
              ascending=args.ascending)
    for test, name in TEST_DATASETS.items():
        test_dir = output_dir / test
        test_dir.mkdir(exist_ok=True)
        column = f"{test}_harmful_score_pct"
        for model in args.models:
            scores = data.loc[data["model"] == model].set_index("version")
            plot_bars(scores, versions, f"{model}: {name} test{seed_note}",
                      test_dir / f"{model}_{test}.png", column, f"{name} harmful ↓",
                      args.ascending)
        plot_bars(average, versions, f"{average_title}: {name} test",
                  test_dir / f"average_{test}.png", column, f"{name} harmful ↓",
                  args.ascending)
    # Export the exact plotted numbers, including the model means.
    average = average.reset_index().assign(model="Average", run="")
    table_path = output_dir / "overall_scores.csv"
    pd.concat([data, average], ignore_index=True).to_csv(
        table_path, index=False, float_format="%.2f"
    )
    print(f"Saved {table_path}")
    if per_seed is not None:
        seed_path = output_dir / "seed_scores.csv"
        per_seed.to_csv(seed_path, index=False, float_format="%.2f")
        print(f"Saved {seed_path}")


if __name__ == "__main__":
    main()

import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGS_DIR = os.path.join(ROOT, "figs", "heatmap")
MODELS = ("llama3", "llama3.1", "mistral", "olmo2", "qwen2.5", "qwen3")
os.makedirs(FIGS_DIR, exist_ok=True)

_BASE_FONT = plt.rcParams["font.size"]
LABEL_FONTSIZE = _BASE_FONT * 2   # x/y axis labels and colorbar label, doubled
TITLE_FONTSIZE = _BASE_FONT * 2   # title, doubled
TICK_FONTSIZE = _BASE_FONT + 1    # tick labels, enlarged by 1
plt.rcParams.update({
    "axes.labelsize": LABEL_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
})

SHORT_TO_FULL = {
    "benign": "benign",
    "material_harm": "causing_material_harm_by_disseminating_misinformation",
    "copyright": "copyright_violations",
    "cyberattack": "cyberattack",
    "unethical": "defamation_encouraging_unethical_or_unsafe_actions",
    "misinfo": "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
    "fraud": "fraud_assisting_illegal_activities",
    "mental": "mental_health_over-reliance_crisis",
    "others": "others",
    "private": "private_information_individual",
    "sensitive": "sensitive_information_organization_government",
    "sexual": "sexual_content",
    "stereo": "social_stereotypes_and_unfair_discrimination",
    "toxic": "toxic_language_hate_speech",
    "violence": "violence_and_physical_harm",
}
FULL_TO_SHORT = {v: k for k, v in SHORT_TO_FULL.items()}
ORDERED_SHORTS = list(SHORT_TO_FULL.keys())


def find_dataset_csv(dirpath, dataset):
    matches = glob.glob(os.path.join(dirpath, f"*_{dataset}.csv"))
    return matches[0] if matches else None


def find_overall_csv(dirpath):
    matches = glob.glob(os.path.join(dirpath, "*_overall.csv"))
    return matches[0] if matches else None


def overall_avg_harmful(dirpath):
    csv = find_overall_csv(dirpath)
    if csv is None:
        return np.nan
    df = pd.read_csv(csv)
    return float(df["harmful_score_pct"].astype(float).mean())


def load_scores(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["subcategory"], df["harmful_score_pct"].astype(float)))


def load_counts(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["subcategory"], df["count"].astype(int)))


def train_result_dir(n, model, dataset, short, suffix=None):
    if suffix is not None:
        return os.path.join(RESULTS_DIR, f"{model}_{suffix}", f"{dataset}_{short}")
    return os.path.join(RESULTS_DIR, f"{model}_{dataset}_{short}_{n}")


def collect_train_shortnames(n, model, dataset, suffix=None):
    if suffix is not None:
        return [short for short in ORDERED_SHORTS
                if os.path.isdir(train_result_dir(n, model, dataset, short, suffix))]
    prefix = f"{model}_{dataset}_"
    suffix = f"_{n}"
    shorts = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        if d.startswith(prefix) and d.endswith(suffix):
            shorts.append(d[len(prefix):-len(suffix)])
    return shorts


def build_matrices(n, init_scores, model, dataset, suffix=None):
    available = set(collect_train_shortnames(n, model, dataset, suffix))
    train_shorts = [s for s in ORDERED_SHORTS if s in available]
    test_shorts = train_shorts
    abs_matrix = np.full((len(test_shorts), len(train_shorts)), np.nan)
    diff_matrix = np.full((len(test_shorts), len(train_shorts)), np.nan)
    for j, short in enumerate(train_shorts):
        d = train_result_dir(n, model, dataset, short, suffix)
        csv = find_dataset_csv(d, dataset)
        if csv is None:
            continue
        scores = load_scores(csv)
        for i, test_short in enumerate(test_shorts):
            full = SHORT_TO_FULL[test_short]
            if full in scores:
                abs_matrix[i, j] = scores[full]
                if full in init_scores:
                    diff_matrix[i, j] = scores[full] - init_scores[full]
    return train_shorts, test_shorts, abs_matrix, diff_matrix


VMIN_DIFF = -35.0
VMAX_DIFF = 15.0
VMIN_ABS = 0.0
VMAX_ABS = 35.0


def _draw_heatmap(ax, matrix, train_shorts, test_shorts, im_kwargs, fmt, text_thresh, dark_high,
                  test_counts=None, n=None):
    im = ax.imshow(matrix, aspect="equal", **im_kwargs)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                if dark_high:
                    color = "white" if v >= text_thresh else "black"
                else:
                    color = "white" if abs(v) >= text_thresh else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7, color=color)

    train_index = {s: j for j, s in enumerate(train_shorts)}
    for i, test_short in enumerate(test_shorts):
        j = train_index.get(test_short)
        if j is not None:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", linewidth=1.8))

    ax.set_xticks(range(len(train_shorts)))
    ax.set_xticklabels(train_shorts, rotation=45, ha="right")
    # Nudge the rotated x-tick labels slightly to the right.
    offset = ScaledTranslation(0.15, 0, ax.figure.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.set_yticks(range(len(test_shorts)))
    if test_counts is not None:
        ylabels = [f"{s} ({test_counts.get(SHORT_TO_FULL[s], '?')})" for s in test_shorts]
    else:
        ylabels = test_shorts
    ax.set_yticklabels(ylabels)
    xlabel = "training data subcategory"
    if n is not None:
        xlabel += f" ({n} training examples per subcategory)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test data subcategory" + (" (test size in parentheses)" if test_counts is not None else ""))
    return im


def plot_diff_heatmap(matrix, train_shorts, test_shorts, n, model, dataset, out_path, test_counts=None):
    cell = 0.55
    grid = cell * max(len(train_shorts), len(test_shorts))
    fig, ax = plt.subplots(figsize=(grid + 4, grid + 1.5))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=VMIN_DIFF, vmax=VMAX_DIFF)
    text_thresh = max(abs(VMIN_DIFF), abs(VMAX_DIFF)) * 0.6
    im = _draw_heatmap(ax, matrix, train_shorts, test_shorts,
                       dict(cmap="bwr", norm=norm), "{:+.1f}", text_thresh,
                       dark_high=False, test_counts=test_counts, n=n)
    ax.set_title(f"{model}: {dataset} harmful score (%,↓) changes after training")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Δ harmful score (%)", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_abs_heatmap(matrix, train_shorts, test_shorts, n, model, dataset, out_path,
                     init_scores=None, test_counts=None, suffix=None, ins_scores=None,
                     overall_avgs=None):
    # Show the official instruct and pre-training baselines before trained runs.
    baseline_labels = []
    baseline_columns = []
    for label, scores in (("ins", ins_scores), ("git", init_scores)):
        if scores is not None:
            baseline_labels.append(label)
            baseline_columns.append(np.array([
                scores.get(SHORT_TO_FULL[s], np.nan) for s in test_shorts
            ]).reshape(-1, 1))
    col_offset = len(baseline_labels)
    if baseline_columns:
        matrix = np.hstack([*baseline_columns, matrix])
        train_shorts = baseline_labels + list(train_shorts)

    cell = 0.55
    grid = cell * max(len(train_shorts), len(test_shorts))
    fig, ax = plt.subplots(figsize=(grid + 4, grid + 1.5))
    norm = Normalize(vmin=VMIN_ABS, vmax=VMAX_ABS)
    text_thresh = VMAX_ABS * 0.6
    im = _draw_heatmap(ax, matrix, train_shorts, test_shorts,
                       dict(cmap="Reds", norm=norm), "{:.1f}", text_thresh,
                       dark_high=True, test_counts=test_counts, n=n)

    # Separator between the baseline columns and the trained columns.
    if col_offset:
        ax.axvline(x=col_offset - 0.5, color="black", linewidth=1.5)

    if overall_avgs is None:
        overall_avgs = np.array([
            overall_avg_harmful(train_result_dir(n, model, dataset, s, suffix))
            for s in train_shorts[col_offset:]
        ])
    finite_idx = [j for j, v in enumerate(overall_avgs) if np.isfinite(v)]
    if finite_idx:
        top_k = 5
        top_idx = sorted(finite_idx, key=lambda j: overall_avgs[j])[:top_k]
        labels = ax.get_xticklabels()
        for j in top_idx:
            labels[j + col_offset].set_fontweight("bold")
        ranked = [f"{train_shorts[j + col_offset]} ({overall_avgs[j]:.2f})" for j in top_idx]
        ax.set_title(f"{model}: {dataset} harmful score (%,↓) comparisons")
    else:
        ax.set_title(f"{model}: {dataset} harmful score (%,↓) comparisons")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("harmful score (%,↓)", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_average_heatmaps(runs, n, dataset, suffix, include_abs):
    # Align by category, using only categories shared by every model.
    train = [s for s in ORDERED_SHORTS if all(s in run["train"] for run in runs)]
    test = [s for s in ORDERED_SHORTS if all(s in run["test"] for run in runs)]
    if not train or not test:
        raise ValueError("No shared subcategories available for the average heatmap")

    def mean_matrix(key):
        aligned = [run[key][np.ix_([run["test"].index(s) for s in test],
                                  [run["train"].index(s) for s in train])]
                   for run in runs]
        # Preserve missing cells rather than averaging a subset of models.
        return np.mean(np.stack(aligned), axis=0)

    def mean_scores(key):
        return {SHORT_TO_FULL[s]: float(np.mean([
            run[key].get(SHORT_TO_FULL[s], np.nan) for run in runs
        ])) for s in test}

    counts = runs[0]["counts"]
    if any(any(run["counts"].get(SHORT_TO_FULL[s]) != counts.get(SHORT_TO_FULL[s])
               for s in test) for run in runs):
        counts = None
    title = f"Average of {len(runs)} models"
    output_tag = suffix or str(n)
    prefix = os.path.join(FIGS_DIR, f"heatmap_average_{dataset}_{output_tag}")
    plot_diff_heatmap(mean_matrix("diff"), train, test, n, title, dataset,
                      prefix + "_diff.png", test_counts=counts)
    if include_abs:
        overall = np.mean([
            [overall_avg_harmful(train_result_dir(n, run["model"], dataset, s, suffix))
             for s in train] for run in runs
        ], axis=0)
        plot_abs_heatmap(mean_matrix("abs"), train, test, n, title, dataset,
                         prefix + "_abs.png", init_scores=mean_scores("git"),
                         ins_scores=mean_scores("ins"), test_counts=counts,
                         overall_avgs=overall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Model name prefix used in results directories (default: all six models).")
    parser.add_argument("--dataset", type=str, default="wildguardmix",
                        help="Dataset name used in results directories/CSVs.")
    parser.add_argument("--n", type=int, nargs="+",
                        help="Number of training examples (default: suffix size, or 400).")
    parser.add_argument("--suffix", default="1800_wildguardmix",
                        help="Grouped result folder suffix (default: 1800_wildguardmix).")
    parser.add_argument("--abs", action=argparse.BooleanOptionalAction, default=True,
                        help="Plot the absolute heatmap alongside the difference (default: enabled).")
    args = parser.parse_args()
    models = [args.model] if args.model else MODELS
    dataset = args.dataset
    if args.suffix is not None:
        args.suffix = args.suffix.lstrip("_")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.suffix):
            parser.error("--suffix must be a nonempty folder-name suffix without path separators")
        if re.match(r"^[1-9][0-9]*_", args.suffix):
            size = int(args.suffix.split("_", 1)[0])
            if args.n is not None and args.n != [size]:
                parser.error("--n must match the training size in --suffix")
            args.n = [size]
    if args.n is None:
        args.n = [400]

    runs_by_n = {n: [] for n in args.n}
    for model in models:
        # Accept an explicit baseline model or infer the same 20k baseline
        # aliases used by plot_bars_overall.py.
        if args.suffix is not None and not os.path.isdir(os.path.join(RESULTS_DIR, f"{model}_{args.suffix}")):
            candidates = [model + baseline for baseline in ("_git20k", "-20k", "-git20k")
                          if os.path.isdir(os.path.join(RESULTS_DIR, f"{model}{baseline}_{args.suffix}"))]
            if len(candidates) != 1:
                parser.error(f"Expected one result group for {model} with suffix {args.suffix!r}; found {candidates}")
            model = candidates[0]

        init_csv = os.path.join(RESULTS_DIR, model, f"{model}_{dataset}.csv")
        init_scores = load_scores(init_csv)
        test_counts = load_counts(init_csv)
        ins_scores = None
        if args.abs:
            base_model = re.sub(r"(?:_git20k|-20k|-git20k)$", "", model)
            ins_model = f"{base_model}-ins"
            ins_csv = os.path.join(RESULTS_DIR, ins_model, f"{ins_model}_{dataset}.csv")
            ins_scores = load_scores(ins_csv)

        for n in args.n:
            train_shorts, test_shorts, abs_mat, diff_mat = build_matrices(n, init_scores, model, dataset, args.suffix)
            if not train_shorts:
                parser.error(f"No subcategory results found for {model}, {dataset}, n={n}, suffix={args.suffix!r}")
            runs_by_n[n].append({
                "model": model, "train": train_shorts, "test": test_shorts,
                "abs": abs_mat, "diff": diff_mat, "git": init_scores,
                "ins": ins_scores, "counts": test_counts,
            })
            output_tag = args.suffix or str(n)
            plot_diff_heatmap(diff_mat, train_shorts, test_shorts, n, model, dataset,
                              os.path.join(FIGS_DIR, f"heatmap_{model}_{dataset}_{output_tag}_diff.png"),
                              test_counts=test_counts)
            if args.abs:
                plot_abs_heatmap(abs_mat, train_shorts, test_shorts, n, model, dataset,
                                 os.path.join(FIGS_DIR, f"heatmap_{model}_{dataset}_{output_tag}_abs.png"),
                                 init_scores=init_scores, test_counts=test_counts, suffix=args.suffix,
                                 ins_scores=ins_scores)

    if len(models) > 1:
        for n, runs in runs_by_n.items():
            plot_average_heatmaps(runs, n, dataset, args.suffix, args.abs)


if __name__ == "__main__":
    main()

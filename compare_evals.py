"""Compare eval.py outputs of the same models in two result roots (batch-invariance check).

python compare_evals.py ROOT_A ROOT_B MODEL... [--limit N]

MODEL is a result name as eval.py derives it: a model shortcut or a checkpoint folder name.

For each model: share of identical IF responses and safety responses (per test set), share
of identical WildGuard labels, and the score differences. With --limit, only the first N
prompts of every benchmark / test set are compared (ROOT_B ran with eval.py --limit N).
"""
import argparse
from pathlib import Path

import pandas as pd

from eval import result_dirs


def folders(root, model):
    """(safety folder, IF folder, file prefix) of MODEL (a model or checkpoint name) under root."""
    safety_dir, if_dir, prefix = result_dirs(str(root), model)
    return Path(safety_dir), Path(if_dir), prefix


def if_responses(root, model):
    _, if_dir, prefix = folders(root, model)
    frame = pd.read_csv(if_dir / f"{prefix}_if_responses.csv", keep_default_na=False)
    return {source: rows.reset_index(drop=True) for source, rows in frame.groupby("prompt_source")}


def safety_responses(root, model):
    folder, _, prefix = folders(root, model)
    out = {}
    for path in sorted(folder.glob(f"{prefix}_*_responses.csv")):
        name = path.name[len(prefix) + 1:-len("_responses.csv")]
        out[name] = pd.read_csv(path, keep_default_na=False)
    return out


def scores(root, model):
    folder, if_dir, prefix = folders(root, model)
    rows = {}
    for kind in ("overall", "overrefusal"):
        path = folder / f"{prefix}_{kind}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            col = [c for c in frame.columns if c.endswith("_score_pct")][0]
            rows.update({f"{kind}:{d}": v for d, v in zip(frame["dataset"], frame[col])})
    if_path = if_dir / f"{prefix}_if_summary.csv"
    if if_path.exists():
        frame = pd.read_csv(if_path)
        rows.update({f"IF:{b}": v * 100 for b, v in zip(frame["benchmark"], frame["prompt_loose"])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_a")
    parser.add_argument("root_b")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for model in args.models:
        print(f"\n=== {model}: {args.root_a} vs {args.root_b}")
        a_if, b_if = if_responses(args.root_a, model), if_responses(args.root_b, model)
        for source in sorted(a_if):
            a, b = a_if[source], b_if[source]
            n = min(len(a), len(b)) if args.limit else len(a)
            assert list(a["prompt"][:n]) == list(b["prompt"][:n]), source
            same = (a["responses"][:n] == b["responses"][:n]).mean()
            print(f"  IF {source:10} n={n:5} identical responses {same:7.2%}")
        a_s, b_s = safety_responses(args.root_a, model), safety_responses(args.root_b, model)
        for name in sorted(a_s):
            a, b = a_s[name], b_s[name]
            n = min(len(a), len(b)) if args.limit else len(a)
            assert list(a["prompt"][:n]) == list(b["prompt"][:n]), name
            same = (a["generated_response"][:n] == b["generated_response"][:n]).mean()
            label = "is_refusal" if "is_refusal" in a else "is_unsafe"
            # Numerically: a column with missing labels is read as float (0.0), a full one as int (0)
            labels_a, labels_b = (pd.to_numeric(x[label][:n]).fillna(-1).to_numpy() for x in (a, b))
            same_label = (labels_a == labels_b).mean()
            print(f"  {name:22} n={n:5} identical responses {same:7.2%}  labels {same_label:7.2%}")
        if not args.limit:
            sa, sb = scores(args.root_a, model), scores(args.root_b, model)
            # A score both runs leave undefined (NaN: no harmful prompt within --limit) is equal
            diffs = {k: round(sb[k] - sa[k], 3) for k in sa
                     if k in sb and not (pd.isna(sa[k]) and pd.isna(sb[k]))}
            print(f"  score differences (B - A): max |diff| {max(map(abs, diffs.values()), default=0)}")
            for key, diff in diffs.items():
                if diff:
                    print(f"    {key}: {sa[key]:.3f} -> {sb[key]:.3f}")


if __name__ == "__main__":
    main()

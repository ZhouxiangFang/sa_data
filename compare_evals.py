"""Compare eval.py outputs of the same models in two result roots (batch-invariance check).

python compare_evals.py ROOT_A ROOT_B MODEL... [--limit N]

ROOT_A/B are eval.py --output_dir folders; MODEL is a result name as eval.py derives it: a model
shortcut or a checkpoint folder name (responses: eval.result_paths).

For each model: share of identical IF responses and safety responses (per test set), share
of identical WildGuard labels, and the score differences. With --limit, only the first N
prompts of every benchmark / test set are compared (ROOT_B ran with eval.py --limit N).
"""
import argparse

import pandas as pd

from eval import read_scores, result_paths
from eval_safe import SCORE_LABELS, metric


def responses(root, model):
    """{name: responses table} of a model: IF per benchmark, then every safety test set."""
    _, folder = result_paths(root, model)
    tables = pd.read_csv(folder / "if_responses.csv", keep_default_na=False)
    out = {f"IF {benchmark}": rows.reset_index(drop=True)
           for benchmark, rows in tables.groupby("benchmark", sort=False)}
    for path in sorted(folder.glob("*_responses.csv")):
        if path.name != "if_responses.csv":
            out[path.name.removesuffix("_responses.csv")] = pd.read_csv(path, keep_default_na=False)
    return out


def scores(root, model):
    path, _ = result_paths(root, model)
    return ({f"{metric}:{test}": value for (metric, test), value in read_scores(path).items()}
            if path.exists() else {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_a")
    parser.add_argument("root_b")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for model in args.models:
        print(f"\n=== {model}: {args.root_a} vs {args.root_b}")
        tables_a, tables_b = responses(args.root_a, model), responses(args.root_b, model)
        for name, a in tables_a.items():
            b = tables_b[name]
            n = min(len(a), len(b)) if args.limit else len(a)
            a, b = a.iloc[:n], b.iloc[:n]
            assert a["prompt"].tolist() == b["prompt"].tolist(), name
            line = (f"  {name:22} n={n:5} identical responses "
                    f"{(a['generated_response'] == b['generated_response']).mean():7.2%}")
            if not name.startswith("IF "):  # the label the score counts ('' if unparsed)
                label = SCORE_LABELS[metric(name)]
                line += f"  labels {(a[label] == b[label]).mean():7.2%}"
            print(line)
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

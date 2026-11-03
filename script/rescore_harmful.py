"""Re-score saved safety evaluations so harmful scores count only harmful prompts.

eval_safe.py generates and judges every prompt of the five safety test sets,
but its harmful scores now count only harmful prompts: WildGuardMix, Aegis and
WildJailbreak also contain benign (and, in WildGuardMix, unlabeled) prompts,
whose responses are almost never harmful and so lowered the scores. This
script updates results evaluated before that change without regenerating
responses or re-running the guard. It reuses each response's saved guard label
(is_unsafe) in <prefix>_<test>_responses.csv, adds the prompt_type column that
eval_safe.py now writes (harmful / benign / unknown, from the test set's own
prompt labels), and recomputes <prefix>_overall.csv and the per-subcategory
<prefix>_<test>.csv with eval_safe.save_results. Overrefusal files are not
touched. Overall rows of a test set without a responses file are kept as they
were and reported.

Example (run from any directory):
    python sa_data/script/rescore_harmful.py                 # results/ and ablations/*
    python sa_data/script/rescore_harmful.py --roots ablations/h0.2_mix
    python sa_data/script/rescore_harmful.py --all-prompts   # the former all-prompt scores
"""

import argparse
import contextlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# sa_data/ holds the shared modules this script imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_safe
from helper import load_safety_dataset, prompt_types, safety_test_datasets
from plot_bars_overall import ABLATIONS_DIR, RESULTS_DIR

# Read responses without turning strings such as "NA" or "None" into NaN.
READ_OPTIONS = {"keep_default_na": False, "na_values": [""]}


def normalize(text):
    return " ".join(str(text).split())


def reference_types(dataset):
    """Prompts and prompt types of a test set, in eval_safe.py's order."""
    data = load_safety_dataset(dataset, "test").reset_index(drop=True)
    return data["prompt"].astype(str).tolist(), prompt_types(data, dataset).tolist()


def align_types(responses, prompts, types, path):
    """prompt_type for each saved response, matched by position, else by prompt."""
    # An empty prompt is saved as an empty field, which reads back as NaN.
    saved = responses["prompt"].fillna("").astype(str)
    if len(saved) == len(prompts):
        same = [a == b or normalize(a) == normalize(b) for a, b in zip(saved, prompts)]
        if all(same):
            return list(types)
    by_prompt = {}
    for prompt, kind in zip(prompts, types):
        by_prompt.setdefault(normalize(prompt), set()).add(kind)
    matched = [by_prompt.get(normalize(prompt), set()) for prompt in saved]
    if any(len(kinds) != 1 for kinds in matched):
        missing = sum(len(kinds) == 0 for kinds in matched)
        ambiguous = sum(len(kinds) > 1 for kinds in matched)
        raise ValueError(f"{path}: {missing} prompts not in the test set, "
                         f"{ambiguous} with conflicting prompt labels")
    return [next(iter(kinds)) for kinds in matched]


def rescore_run(overall_path, references, all_prompts):
    """Re-score one run; return (overall path, kept datasets, before, after)."""
    prefix = overall_path.name.removesuffix("_overall.csv")
    previous = pd.read_csv(overall_path)
    order = [name for name in previous["dataset"] if name in safety_test_datasets]
    order += [name for name in safety_test_datasets if name not in order]
    datasets = {}
    for name in order:
        path = overall_path.parent / f"{prefix}_{name}_responses.csv"
        if not path.exists():
            continue
        responses = pd.read_csv(path, **READ_OPTIONS)
        responses["prompt_type"] = align_types(responses, *references[name], path)
        # eval_safe.save_results rebuilds abbr from sub_abbrs exactly as saved.
        responses["sub_abbrs"] = responses["abbr"].astype(str).str.split(", ")
        datasets[name] = responses
    kept = [name for name in previous["dataset"] if name not in (*datasets, "Avg")]
    before = previous.set_index("dataset")["harmful_score_pct"].astype(float)
    if datasets:
        with contextlib.redirect_stdout(io.StringIO()):
            eval_safe.save_results(datasets, overall_path.parent, prefix, merge=True,
                                   all_prompts=all_prompts)
    after = pd.read_csv(overall_path).set_index("dataset")["harmful_score_pct"].astype(float)
    return overall_path, kept, before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path,
                        help="Result roots to search recursively "
                             "(default: results/ and every ablations/<setting>/)")
    parser.add_argument("--all-prompts", action="store_true",
                        help="Score every prompt, as before (default: harmful prompts only)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    roots = args.roots or [RESULTS_DIR, *sorted(path for path in ABLATIONS_DIR.iterdir()
                                                if path.is_dir())]
    runs = sorted(path for root in roots for path in root.resolve().rglob("*_overall.csv")
                  if "if_eval" not in path.parts)
    if not runs:
        parser.error(f"No *_overall.csv files under {', '.join(map(str, roots))}")
    print(f"Re-scoring {len(runs)} runs under {', '.join(map(str, roots))} "
          f"({'all prompts' if args.all_prompts else 'harmful prompts only'})")
    references = {}
    for name in safety_test_datasets:
        with contextlib.redirect_stderr(io.StringIO()):
            references[name] = reference_types(name)
        counts = pd.Series(references[name][1]).value_counts().to_dict()
        print(f"  {name}: {len(references[name][0])} prompts {counts}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(rescore_run, path, references, args.all_prompts)
                   for path in runs]
        for future in futures:
            path, kept, before, after = future.result()
            if kept:
                print(f"Warning: {path} has no responses for {kept}; kept their old rows")
            for dataset in after.index:
                rows.append({"run": str(path), "dataset": dataset,
                             "before": before.get(dataset), "after": after[dataset]})
    table = pd.DataFrame(rows)
    summary = table.groupby("dataset", sort=False)[["before", "after"]].mean().round(2)
    print(f"\nMean harmful score over {len(runs)} runs (before -> after):")
    print(summary.to_string())


if __name__ == "__main__":
    main()

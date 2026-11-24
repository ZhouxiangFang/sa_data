"""List one seed's training jobs (model x training group) and which are left to do.

    python train_jobs.py --seed 43 --ckpt_dir /scratch/zf28/ckpts/seed43/default \
        --results_dir ../results/correlations/default [-- align.py options]

One tab-separated line per job: model, group, checkpoint, status:
  train   still to train
  saved   checkpoint saved, not yet evaluated
  done    evaluated: eval.py's scores CSV in --results_dir is complete (the checkpoint may be
          gone)
  stale   a checkpoint or scores exist from other training data or settings: their fingerprint
          (helper.run_fingerprint) is not this run's
The run's training settings (helper.TRAINING_SETTINGS) are read from the align.py options after
--; the others there are ignored. Groups: every WildGuardMix subcategory with enough eligible
rows for the harmful third of --num_train (helper.training_groups), then the random and evenly
baselines; with --selected DATASET (stage 2), every subset select_data.py saved for the seed.
"""

import argparse
import sys
from pathlib import Path

from eval import evaluated, result_paths
from helper import (BASELINE_ABBRS, DATASET, DEFAULT_NUM_TRAIN, DEFAULT_SEED, GIT20K_MODELS,
                    SELECTION_DATASETS, add_pool_argument, add_training_arguments,
                    checkpoint_fingerprint, checkpoint_name, load_train, mixture_counts,
                    run_files, run_fingerprint, selection_methods, training_groups)


def status(checkpoint, scores, fingerprint):
    """train, saved, done or stale (see the module doc); scores: the scores CSV, or None."""
    # align.py writes the tokenizer last, so it marks a complete save.
    saved = all((checkpoint / name).is_file() for name in ("config.json", "tokenizer_config.json"))
    if fingerprint and scores and evaluated(scores, fingerprint):
        return "done"
    if fingerprint and saved and checkpoint_fingerprint(checkpoint) == fingerprint:
        return "saved"
    return "stale" if saved or (scores and scores.exists()) else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(GIT20K_MODELS))
    parser.add_argument("--abbrs", nargs="+", help="Only these groups")
    parser.add_argument("--selected", choices=SELECTION_DATASETS,
                        help="Stage 2: the subsets selected from this dataset")
    parser.add_argument("--mixed_diversity", action="store_true",
                        help="With --selected: the mixed-diversity ablation's subsets")
    parser.add_argument("--num_train", type=int, default=DEFAULT_NUM_TRAIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="align.py --output_dir")
    parser.add_argument("--results_dir", type=Path,
                        help="eval.py --output_dir; jobs evaluated there are done")
    add_pool_argument(parser)
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    training = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_training_arguments(training)
    settings = vars(training.parse_known_args(argv[split + 1:])[0])
    harmful_count = mixture_counts(args.num_train)[0]
    if args.selected:
        groups = selection_methods(args.seed, args.selected, harmful_count,
                                   args.response_label_filter, args.mixed_diversity)
    else:
        groups = training_groups(load_train(), harmful_count, args.response_label_filter)
        groups += list(BASELINE_ABBRS)
    if args.abbrs:
        unknown = sorted(set(args.abbrs) - set(groups))
        if unknown:
            sys.exit(f"Not a training group with {harmful_count:,} examples: {' '.join(unknown)}"
                     f" (groups: {' '.join(groups)})")
        groups = [abbr for abbr in groups if abbr in args.abbrs]
    files = {abbr: run_files(abbr, args.seed, args.num_train, args.response_label_filter,
                             settings["max_length"], args.selected, args.mixed_diversity)
             for abbr in groups}
    fingerprints = {(model, abbr): run_fingerprint(model, args.seed, files[abbr], settings)
                    for model in args.models for abbr in groups}
    for model in args.models:
        for abbr in groups:
            checkpoint = args.ckpt_dir / checkpoint_name(model, args.num_train, abbr, args.seed,
                                                         args.selected or DATASET)
            scores = args.results_dir and result_paths(args.results_dir, checkpoint.name)[0]
            print(model, abbr, checkpoint, status(checkpoint, scores, fingerprints[model, abbr]),
                  sep="\t")


if __name__ == "__main__":
    main()

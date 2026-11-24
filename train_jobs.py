"""List one seed's training jobs (model x training group) and which are left to do.

    python train_jobs.py --seed 43 --ckpt_dir /scratch/zf28/ckpts/seed43/default \
        --results_dir ../results

One tab-separated line per job: model, group, checkpoint, status:
  train   still to train
  saved   checkpoint saved, not yet evaluated
  done    evaluated: eval.py's summaries are in --results_dir (the checkpoint may be gone)
Groups: every WildGuardMix subcategory with enough eligible rows for the harmful third of
--num_train (helper.training_groups), then the random and evenly baselines.
"""

import argparse
import os
import sys
from pathlib import Path

from helper import (BASELINE_ABBRS, DEFAULT_NUM_TRAIN, DEFAULT_SEED, GIT20K_MODELS,
                    add_pool_argument, checkpoint_name, load_train, mixture_counts,
                    training_groups)


def status(checkpoint, results_dir):
    if results_dir is not None:
        from eval import result_files

        if all(os.path.exists(path) for path in result_files(results_dir, checkpoint.name)):
            return "done"
    # align.py writes the tokenizer last, so it marks a complete save.
    if (checkpoint / "config.json").is_file() and (checkpoint / "tokenizer_config.json").is_file():
        return "saved"
    return "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(GIT20K_MODELS))
    parser.add_argument("--abbrs", nargs="+", help="Only these groups")
    parser.add_argument("--num_train", type=int, default=DEFAULT_NUM_TRAIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="align.py --output_dir")
    parser.add_argument("--results_dir", type=Path,
                        help="eval.py --output_dir; jobs evaluated there are done")
    add_pool_argument(parser)
    args = parser.parse_args()
    harmful_count = mixture_counts(args.num_train)[0]
    groups = training_groups(load_train(), harmful_count, args.response_label_filter)
    groups += list(BASELINE_ABBRS)
    if args.abbrs:
        unknown = sorted(set(args.abbrs) - set(groups))
        if unknown:
            sys.exit(f"Not a training group with {harmful_count:,} examples: {' '.join(unknown)}"
                     f" (groups: {' '.join(groups)})")
        groups = [abbr for abbr in groups if abbr in args.abbrs]
    for model in args.models:
        for abbr in groups:
            checkpoint = args.ckpt_dir / checkpoint_name(model, args.num_train, abbr, args.seed)
            print(model, abbr, checkpoint, status(checkpoint, args.results_dir), sep="\t")


if __name__ == "__main__":
    main()

"""List one run's training jobs (model x dataset x group) and which are left to do.

    python train_jobs.py --seed 43 --ckpt_dir /scratch/zf28/ckpts/seed43/default \
        --results_dir ../results

Prints one tab-separated line per job: model, dataset, group, checkpoint path and
status, which is
  train   still to train
  saved   checkpoint saved (config and tokenizer written), not yet evaluated
  done    evaluated: eval.py's summaries are in --results_dir (the checkpoint may be gone)
Groups default to every non-benign subcategory with enough examples for the
harmful part (or, with --group_by tactic, every such Gretel tactic), then the random
and evenly baselines. With --group_by selection the groups are the subsets
select_data.py saved for the seed (data/seed<seed>/<dataset>/<name>_<harmful count>.csv).
--response_filter and --dedup are align.py's pool options: group sizes count the rows
that pass them, and selected subsets come from data/seed<seed>/<dataset>_<tag>/.
Launchers (chili/train.sh, a Slurm array) train the "train" jobs with align.py and the
same settings.
"""

import argparse
import os
import sys
from pathlib import Path

from helper import (
    BASELINE_ABBRS, DEFAULT_POOL, DEFAULT_SEED, GIT20K_MODELS, GROUP_BYS, TRAIN_DATASETS,
    add_pool_arguments, checkpoint_name, group_sizes, mixture_counts, resolve_pool_options,
    selection_names,
)

DATASETS = TRAIN_DATASETS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(GIT20K_MODELS),
                        help="Starting models, as align.py --model (default: the git20k models)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        help="Default: aegis wildguardmix gretel (gretel with --group_by tactic)")
    parser.add_argument("--abbrs", nargs="+", help="Only these groups (may include benign)")
    parser.add_argument("--group_by", choices=GROUP_BYS, default="subcategory")
    parser.add_argument("--num_train", type=int, default=1800)
    parser.add_argument("--harmful_rate", type=float, default=1 / 3)
    parser.add_argument("--general_rate", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="align.py --output_dir")
    parser.add_argument("--results_dir", type=Path,
                        help="eval.py --output_dir; jobs evaluated there are done")
    add_pool_arguments(parser)
    args = parser.parse_args(argv)
    resolve_pool_options(parser, args, ["gretel"] if args.group_by == "tactic" else DATASETS)
    if args.group_by == "tactic" and args.datasets != ["gretel"]:
        parser.error("--group_by tactic supports only --datasets gretel")
    try:
        args.harmful_count = mixture_counts(args.num_train, args.harmful_rate, args.general_rate)[0]
    except ValueError as error:
        parser.error(str(error))
    return args


def dataset_groups(dataset, harmful_count, group_by, abbrs=None, seed=DEFAULT_SEED,
                   options=DEFAULT_POOL):
    """(groups to train, {group: size} of requested groups too small) of one dataset.

    abbrs (default: every non-benign group and the baselines, or every selected subset
    of the seed) may name groups of other datasets; those are ignored here. Group sizes
    count the rows that pass the pool options.
    """
    if group_by == "selection":
        names = selection_names(seed, dataset, harmful_count, options)
        if not names:
            print(f"No selected subsets of {harmful_count:,} examples for "
                  f"{options.folder(dataset)}, seed {seed}; run select_data.py "
                  "(chili/select_data.sh) first", file=sys.stderr)
        return [name for name in (names if abbrs is None else abbrs) if name in names], {}
    sizes = group_sizes(dataset, group_by, options)
    if abbrs is None:
        abbrs = [abbr for abbr in sizes if abbr != "benign"] + list(BASELINE_ABBRS)
    abbrs = [abbr for abbr in abbrs if abbr in sizes or abbr in BASELINE_ABBRS]
    small = {abbr: sizes[abbr] for abbr in abbrs if sizes.get(abbr, harmful_count) < harmful_count}
    return [abbr for abbr in abbrs if abbr not in small], small


def status(checkpoint, results_dir):
    if results_dir is not None:
        from eval import result_files

        if all(os.path.exists(path) for path in result_files(results_dir, checkpoint.name)):
            return "done"
    # The tokenizer is saved after the weights, so it marks a complete save.
    if (checkpoint / "config.json").is_file() and (checkpoint / "tokenizer_config.json").is_file():
        return "saved"
    return "train"


def main():
    args = parse_args()
    plans = {dataset: dataset_groups(dataset, args.harmful_count, args.group_by, args.abbrs,
                                     args.seed, args.options)
             for dataset in args.datasets}
    known = {abbr for groups, small in plans.values() for abbr in [*groups, *small]}
    unknown = [abbr for abbr in args.abbrs or [] if abbr not in known]
    if unknown:
        sys.exit(f"Unknown {args.group_by} in {' '.join(args.datasets)}: {' '.join(unknown)}")
    for dataset, (groups, small) in plans.items():
        for abbr, size in small.items():
            print(f"Skipping {dataset}/{abbr}: {size:,} examples, {args.harmful_count:,} needed",
                  file=sys.stderr)
        for model in args.models:
            for abbr in groups:
                checkpoint = args.ckpt_dir / checkpoint_name(
                    model, args.num_train, dataset, abbr, args.seed)
                print(model, dataset, abbr, checkpoint, status(checkpoint, args.results_dir),
                      sep="\t")


if __name__ == "__main__":
    main()

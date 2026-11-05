"""Evaluate one model on instruction following and safety, loading it once.

    python eval.py --model /scratch/zf28/ckpts/seed43/default/llama3.1_git20k_1800_aegis_hate_seed43
    python eval.py --model qwen2.5-ins --skip_if --datasets gretel --merge

One vLLM engine generates the IFBench/IFEval responses (eval_if.py) and then,
with its prefix cache cleared so they match a run of their own, the responses
to the safety and overrefusal test sets (eval_safe.py), one greedy generate call
per benchmark or test set. The model is then released and WildGuard judges the
safety responses.

A trained checkpoint <model>_<n>_<dataset>_<abbr>_seed<seed> (align.py) writes to
  <output_dir>/seed<seed>/<model>_<n>_<dataset>/<dataset>_<abbr>/    safety
  <output_dir>/seed<seed>/if_eval/<model>_<n>_<dataset>/             IF
(seed 42 when the name has no seed); any other model to <output_dir>/<model>/ and
<output_dir>/if_eval/. The default output_dir is results/ beside sa_data; use
ablations/<setting> for any training setting other than the default 1:1:1 mixture.
"""

import argparse
import gc
import logging
import os
import re
from pathlib import Path

import eval_if
import eval_safe
from helper import LEGACY_SEED, model_mapping, seed_dir

logging.getLogger("vllm").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_NAME = re.compile(
    r"(?P<model>.+)_(?P<count>[1-9][0-9]*)_(?P<category>(?:wildguardmix|aegis|gretel)_.+?)"
    r"(?:_seed(?P<seed>[1-9][0-9]*))?(?P<variant>_permit)?"
)


def resolve_model(model):
    """(model ID or path, result name) of a model shortcut, path or repo ID."""
    if os.path.isdir(model):
        path = os.path.abspath(model)
        return path, os.path.basename(os.path.normpath(path))
    return model_mapping.get(model.lower(), model), model.replace("/", "_")


def result_dirs(output_dir, name):
    """(safety folder, IF folder, file prefix) of a model's results; see the module doc."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    match = CHECKPOINT_NAME.fullmatch(name)
    if not match:
        return os.path.join(output_dir, name), os.path.join(output_dir, "if_eval"), name
    root = os.path.join(output_dir, seed_dir(match["seed"] or LEGACY_SEED))
    variant = match["variant"] or ""
    run = f"{match['model']}_{match['count']}_{match['category'].split('_', 1)[0]}{variant}"
    return (os.path.join(root, run, match["category"]), os.path.join(root, "if_eval", run),
            match["category"] + variant)


def result_files(output_dir, name):
    """The summary CSVs a full evaluation writes: safety overall and IF summary."""
    safety_dir, if_dir, prefix = result_dirs(output_dir, name)
    return (os.path.join(safety_dir, f"{prefix}_overall.csv"),
            os.path.join(if_dir, f"{prefix}_if_summary.csv"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Checkpoint path, model shortcut (helper.model_mapping) or repo ID.")
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR,
                        help="Result root (default: results/ beside sa_data).")
    parser.add_argument("--datasets", nargs="+", choices=eval_safe.TEST_SETS,
                        default=eval_safe.TEST_SETS,
                        help="Safety and overrefusal test sets (default: all).")
    parser.add_argument("--guard", default="wildguard", help="Guard model name, repo ID or path.")
    parser.add_argument("--skip_if", action="store_true", help="Skip IFBench and IFEval.")
    parser.add_argument("--skip_safety", action="store_true", help="Skip the safety test sets.")
    parser.add_argument("--permit", action="store_true",
                        help="Let the model decide whether to comply with each safety request "
                             "(results get a _permit suffix; IF is skipped, as its prompts "
                             "do not change).")
    parser.add_argument("--score_all_prompts", action="store_true",
                        help="Count every safety test prompt in the harmful score, not only "
                             "harmful ones (benign ones are always generated, judged and saved).")
    parser.add_argument("--merge", action="store_true",
                        help="Keep other test sets' rows in an existing overall/overrefusal CSV "
                             "and recompute its Avg instead of overwriting the file.")
    parser.add_argument("--limit", type=int,
                        help="Use only the first N prompts of every benchmark (quick test).")
    args = parser.parse_args(argv)
    args.skip_if |= args.permit
    if args.skip_if and args.skip_safety:
        parser.error("--skip_if and --skip_safety leave nothing to evaluate")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    model_id, name = resolve_model(args.model)
    guard_id = model_mapping.get(args.guard.lower(), args.guard)
    _, if_dir, if_prefix = result_dirs(args.output_dir, name)
    safety_dir, _, prefix = result_dirs(args.output_dir, name + "_permit" * args.permit)
    print(f"Model: {model_id}")
    if not args.skip_if:
        print(f"IF results: {if_dir}")
    if not args.skip_safety:
        print(f"Safety results: {safety_dir}")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    llm = LLM(model=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not args.skip_if:
        eval_if.run(llm, tokenizer, args.model, model_id, if_dir, if_prefix, limit=args.limit)
        if args.skip_safety:
            return
        llm.reset_prefix_cache()

    datasets = eval_safe.load_test_sets(args.datasets, args.limit)
    eval_safe.generate(llm, tokenizer, datasets, permit=args.permit)
    print("\nReleasing the model for the guard...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    guard = LLM(model=guard_id)
    eval_safe.judge(guard_id, guard, datasets)
    del guard
    eval_safe.save_results(datasets, safety_dir, prefix, merge=args.merge,
                           all_prompts=args.score_all_prompts)


if __name__ == "__main__":
    main()

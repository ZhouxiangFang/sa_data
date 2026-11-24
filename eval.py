"""Evaluate one model on instruction following, safety and overrefusal, loading it once.

    python eval.py --output_dir ../results/correlations/default \
        --model /scratch/zf28/ckpts/seed43/default/llama3.1_git20k_1800_wildguardmix_toxic_seed43
    python eval.py --model qwen2.5_git20k --limit 5

One vLLM engine generates the IFBench/IFEval responses (eval_if.py) and then,
with its prefix cache cleared so they match a run of their own, the responses
to the safety and overrefusal test sets (eval_safe.py), one greedy generate call
per benchmark or test set. The model is then released and WildGuard judges the
safety responses. vLLM runs in batch-invariant mode, so a rerun reproduces the results.
Safety: harmful-response rate on the harmful prompts of five test sets; overrefusal: refusal
rate on XSTest's safe prompts and OR-Bench-Hard-1k; IF: IFBench and IFEval (prompt-level loose
accuracy is the primary metric).

Each model gets one scores CSV holding every score of its evaluation: columns metric, test_set,
n, count, na_responses, score (%) and fingerprint (the checkpoint's, from align.py's run.json;
empty for other models), metrics harmful, overrefusal and if_{prompt,instruction}_{loose,strict},
and for each metric an Avg row (the unweighted mean of its test sets). A model's responses and
per-subcategory breakdowns go to its responses folder:
  checkpoint <model>_<n>_<dataset>_<abbr>_seed<seed> (align.py):
      <output_dir>/seed<seed>/<model>/<abbr>.csv  and  <responses>/seed<seed>/<model>/<abbr>/
  any other model (the instruct and git20k baselines):
      <output_dir>/<model>.csv  and  <responses>/<model>/
<responses> is output_dir's path under /scratch/zf28/results when output_dir is in results/,
else output_dir itself (helper.responses_dir). --output_dir defaults to results/baselines; a
checkpoint needs its setting's folder, results/correlations/<setting> (stage 1) or
results/selection/<setting> (stage 2), as range/common.sh's setting_setup sets it.
"""

import argparse
import gc
import logging
import os
import re
from pathlib import Path

import eval_if
import eval_safe
import pandas as pd

from helper import (BASELINES_DIR, SELECTION_DATASETS, checkpoint_fingerprint, model_mapping,
                    responses_dir, seed_dir)

logging.getLogger("vllm").setLevel(logging.WARNING)
# Greedy decoding alone is not repeatable in vLLM: outputs depend on how requests are
# batched (two runs of one checkpoint shared ~40% of responses; identical weights gave ~16%
# different responses on RANGE). Batch-invariant kernels make each output independent of
# the batch. vLLM 0.11 silently ignored VLLM_BATCH_INVARIANT, so load_llm() refuses a vLLM
# that does not read it; the nlp env (requirements.txt, vLLM 0.23) has one that does.
# VLLM_BATCH_INVARIANT=0 turns this off.
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
BATCH_INVARIANT = os.environ["VLLM_BATCH_INVARIANT"] == "1"
# One of the attention backends vLLM tests batch invariance with on CUDA.
ATTENTION_BACKEND = "FLASH_ATTN"
# check_tokenizer's text: starts with a word and has double spaces, indentation and a tab,
# where converted SentencePiece tokenizers go wrong (transformers 5.20 did on allenai/wildguard,
# which ships only a tokenizer.model: no leading-space marker, double spaces merged).
TOKENIZER_PROBE = "Hello,  i.e. two  spaces\n  indented line\n\tTab and   three spaces.  End"

CHECKPOINT_NAME = re.compile(
    rf"(?P<model>.+)_(?P<count>[1-9][0-9]*)_(?P<dataset>{'|'.join(SELECTION_DATASETS)})"
    r"_(?P<abbr>.+)_seed(?P<seed>[1-9][0-9]*)")
SCORE_COLUMNS = ["metric", "test_set", "n", "count", "na_responses", "score", "fingerprint"]
METRIC_ORDER = ["harmful", "overrefusal",
                *(f"if_{name}" for name in eval_if.SCORE_NAMES)]
# The rows a complete evaluation writes; train_jobs.py and the figures look for them.
SUMMARY_ROWS = [("harmful", "Avg"), ("overrefusal", "Avg"), ("if_prompt_loose", "Avg")]


def resolve_model(model):
    """(model ID or path, result name) of a model shortcut, path or repo ID."""
    if os.path.isdir(model):
        path = os.path.abspath(model)
        return path, os.path.basename(os.path.normpath(path))
    return model_mapping.get(model.lower(), model), model.replace("/", "_")


def run_csv(output_dir, model, abbr, seed):
    """Scores CSV of a trained run."""
    return Path(output_dir) / seed_dir(seed) / model / f"{abbr}.csv"


def result_paths(output_dir, name):
    """(scores CSV, responses folder) of a model; see the module doc."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    match = CHECKPOINT_NAME.fullmatch(name)
    if not match:
        return Path(output_dir) / f"{name}.csv", responses_dir(output_dir) / name
    return (run_csv(output_dir, match["model"], match["abbr"], match["seed"]),
            responses_dir(output_dir) / seed_dir(match["seed"]) / match["model"] / match["abbr"])


def read_table(path):
    """A scores CSV (fingerprints as text: one of digits only would read as a number)."""
    return pd.read_csv(path, dtype={"fingerprint": str})


def read_scores(path):
    """A scores CSV as a Series of scores (%) indexed by (metric, test_set)."""
    return read_table(path).set_index(["metric", "test_set"])["score"]


def evaluated(path, fingerprint=None):
    """Whether a scores CSV has the harmful, overrefusal and IF averages (with fingerprint:
    every row of the checkpoint with that fingerprint)."""
    if not Path(path).is_file():
        return False
    table = read_table(path)
    return (set(SUMMARY_ROWS) <= set(zip(table["metric"], table["test_set"]))
            and (fingerprint is None or (table["fingerprint"] == fingerprint).all()))


def save_scores(path, rows, fingerprint=None):
    """Write a model's score rows with its checkpoint's fingerprint, keeping the earlier rows
    of the same checkpoint for metrics this evaluation skipped (--skip_if, --skip_safety)."""
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS).assign(fingerprint=fingerprint)
    path = Path(path)
    if path.exists():
        earlier = read_table(path)
        keep = (~earlier["metric"].isin(scores["metric"])
                & (earlier["fingerprint"].fillna("") == (fingerprint or "")))
        scores = pd.concat([scores, earlier[keep]], ignore_index=True)
    order = {metric: i for i, metric in enumerate(METRIC_ORDER)}
    scores = scores.sort_values("metric", key=lambda column: column.map(order), kind="stable")
    for column in ("n", "count", "na_responses"):  # IF rows have no na_responses
        scores[column] = scores[column].astype("Int64")
    scores["score"] = scores["score"].map(lambda value: f"{float(value):.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)
    print(f"Saved: {path}")


def release(llm, timeout=120):
    """Stop vLLM's engine and wait until the GPU memory it held is free again.

    vLLM runs the engine in a child process that can still be exiting when `del llm`
    returns, so WildGuard's engine then found 7 of 93 GiB free on an H100 and refused to
    start (range, 2026-11-05). Waits for vLLM's default 90% to be free.
    """
    import time

    import torch

    engine_core = getattr(llm.llm_engine, "engine_core", None)
    if engine_core is not None:
        engine_core.shutdown()
    gc.collect()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        free, total = torch.cuda.mem_get_info()
        if free >= 0.9 * total:
            break
        time.sleep(2)
    torch.cuda.empty_cache()


def sentencepiece_file(model_id):
    """The model's own SentencePiece tokenizer.model (local folder or HF cache), or None."""
    if os.path.isdir(model_id):
        path = os.path.join(model_id, "tokenizer.model")
        return path if os.path.isfile(path) else None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "tokenizer.model")
    return path if isinstance(path, str) else None


def check_tokenizer(tokenizer, model_id):
    """Refuse a tokenizer that splits text unlike the model's own SentencePiece file."""
    path = sentencepiece_file(model_id)
    if path is None:
        return
    import sentencepiece

    expected = sentencepiece.SentencePieceProcessor(model_file=path).encode(TOKENIZER_PROBE)
    got = list(tokenizer.encode(TOKENIZER_PROBE, add_special_tokens=False))
    if got != expected:
        raise SystemExit(
            f"The tokenizer of {model_id} disagrees with its SentencePiece model ({path}) on "
            f"{TOKENIZER_PROBE!r}: {got[:12]} vs {expected[:12]}; check the transformers "
            "version (requirements.txt).")


def load_llm(model_id):
    """A vLLM engine, batch-invariant unless VLLM_BATCH_INVARIANT=0, with a checked tokenizer."""
    import vllm
    import vllm.envs

    # The Hugging Face tokenizer, as in training: vLLM's "auto" switches Mistral-format repos
    # (mistralai/Mistral-7B-Instruct-v0.3) to mistral_common, which tokenizes differently.
    options = {"tokenizer_mode": "hf"}
    if not BATCH_INVARIANT:
        print(f"vLLM {vllm.__version__}: batch invariance OFF, outputs depend on batching")
    elif not getattr(vllm.envs, "VLLM_BATCH_INVARIANT", False):
        raise SystemExit(
            f"vLLM {vllm.__version__} ignores VLLM_BATCH_INVARIANT, so outputs would depend on "
            "batching. Build the env from requirements.txt (ENVIRONMENT.md) "
            "or set VLLM_BATCH_INVARIANT=0 to accept that.")
    else:
        print(f"vLLM {vllm.__version__}: batch-invariant, {ATTENTION_BACKEND} attention")
        options["attention_config"] = {"backend": ATTENTION_BACKEND}
    llm = vllm.LLM(model=model_id, **options)
    check_tokenizer(llm.get_tokenizer(), model_id)
    return llm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Checkpoint path, model shortcut (helper.model_mapping) or repo ID.")
    parser.add_argument("--output_dir", type=Path,
                        help="Scores folder (default: results/baselines; a checkpoint needs "
                             "its setting's folder, see the module doc).")
    parser.add_argument("--datasets", nargs="+", choices=eval_safe.TEST_SETS,
                        default=eval_safe.TEST_SETS,
                        help="Safety and overrefusal test sets (default: all).")
    parser.add_argument("--skip_if", action="store_true", help="Skip IFBench and IFEval.")
    parser.add_argument("--skip_safety", action="store_true",
                        help="Skip the safety and overrefusal test sets.")
    parser.add_argument("--limit", type=int,
                        help="Use only the first N prompts of every benchmark (quick test).")
    args = parser.parse_args(argv)
    if args.skip_if and args.skip_safety:
        parser.error("--skip_if and --skip_safety leave nothing to evaluate")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    model_id, name = resolve_model(args.model)
    if args.output_dir is None:
        if CHECKPOINT_NAME.fullmatch(name):
            raise SystemExit("A checkpoint needs --output_dir: results/correlations/<setting> "
                             "(stage 1) or results/selection/<setting> (stage 2)")
        args.output_dir = BASELINES_DIR
    scores_csv, responses = result_paths(args.output_dir, name)
    fingerprint = checkpoint_fingerprint(model_id) if os.path.isdir(model_id) else None
    print(f"Model: {model_id} (fingerprint {fingerprint})\nScores: {scores_csv}\n"
          f"Responses: {responses}")

    from transformers import AutoTokenizer

    llm = load_llm(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rows = []
    if not args.skip_if:
        rows += eval_if.run(llm, tokenizer, responses, limit=args.limit)
        if args.skip_safety:
            save_scores(scores_csv, rows, fingerprint)
            return
        llm.reset_prefix_cache()

    datasets = eval_safe.load_test_sets(args.datasets, args.limit)
    eval_safe.generate(llm, tokenizer, datasets)
    print("\nReleasing the model for the guard...")
    release(llm)
    del llm

    guard = load_llm(model_mapping["wildguard"])
    eval_safe.judge(guard, datasets)
    del guard
    save_scores(scores_csv, eval_safe.save_results(datasets, responses) + rows, fingerprint)


if __name__ == "__main__":
    main()

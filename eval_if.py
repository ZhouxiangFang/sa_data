"""IFBench and IFEval: generation and scoring (run by eval.py).

Install the evaluator directly from GitHub:
    python -m pip install "ifbench @ git+https://github.com/allenai/IFBench.git"
"""

import json
import os

import pandas as pd

from helper import percentage, with_average

SCORE_NAMES = ["prompt_strict", "instruction_strict", "prompt_loose", "instruction_loose"]
MAX_TOKENS = 4096
# Seeds the two random sources of scoring (see evaluate), so the same responses always get
# the same scores.
SCORING_SEED = 0


def load_benchmarks(limit=None):
    """The official IFBench and IFEval examples (the first `limit` of each)."""
    try:
        from ifbench import data_path
    except ImportError as error:
        raise SystemExit(
            "Install IFBench with:\n"
            "python -m pip install \"ifbench @ git+https://github.com/allenai/IFBench.git\""
        ) from error
    from datasets import load_dataset

    with open(data_path(), encoding="utf-8") as file:
        ifbench = [json.loads(line) for line in file if line.strip()]
    ifeval = list(load_dataset("google/IFEval", split="train"))
    return {"IFBench": ifbench[:limit], "IFEval": ifeval[:limit]}


def format_prompts(tokenizer, examples):
    """Apply the chat template, or use raw prompts for base models."""
    prompts = [example["prompt"] for example in examples]
    if not tokenizer.chat_template:
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for prompt in prompts
    ]


def generate(llm, tokenizer, examples, max_tokens=MAX_TOKENS):
    """One greedy response per example; thinking models are scored on their final answer."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=max_tokens, seed=42)
    outputs = llm.generate(format_prompts(tokenizer, examples), params, use_tqdm=True)
    responses = []
    for output in outputs:
        response = output.outputs[0].text
        if "</think>" in response:
            response = response.rsplit("</think>", 1)[1].lstrip()
        responses.append(response)
    return responses


def response_variants(response, loose):
    """The variants used by the official loose evaluation."""
    if not loose:
        return [response]
    lines = response.split("\n")
    variants = [
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    return variants + [text.replace("*", "") for text in variants]


def check_response(example, response, loose=False):
    """Check every constraint attached to one benchmark prompt."""
    from ifbench.instructions_registry import INSTRUCTION_DICT

    results = []
    variants = response_variants(response, loose)
    for instruction_id, kwargs in zip(example["instruction_id_list"], example["kwargs"]):
        checker = INSTRUCTION_DICT[instruction_id](instruction_id)
        checker.build_description(**{key: value for key, value in kwargs.items() if value is not None})
        if "prompt" in (checker.get_instruction_args() or []):
            checker.build_description(prompt=example["prompt"])
        results.append(any(text.strip() and checker.check_following(text) for text in variants))
    return results


def evaluate(benchmark, examples, responses):
    """Score rows of one benchmark (eval.py's scores CSV, if_<score> metrics in %), the same
    on every call for the same responses.

    Scoring has two random sources, both seeded here per benchmark: checkers draw any
    argument their kwargs leave unset from `random`, and langdetect (the response-language
    and English-case checks) samples unless DetectorFactory.seed is set.
    """
    import random

    from langdetect import DetectorFactory

    random.seed(SCORING_SEED)
    DetectorFactory.seed = SCORING_SEED
    strict, loose = [], []
    # Strict then loose per example, the official order (some checkers draw from `random`).
    for example, response in zip(examples, responses):
        strict.append(check_response(example, response))
        loose.append(check_response(example, response, loose=True))
    rows = []
    for mode, results in (("strict", strict), ("loose", loose)):
        rows += [{"metric": f"if_prompt_{mode}", "test_set": benchmark, "n": len(results),
                  "count": sum(map(all, results))},
                 {"metric": f"if_instruction_{mode}", "test_set": benchmark,
                  "n": sum(map(len, results)), "count": sum(map(sum, results))}]
    for row in rows:
        row["score"] = percentage(row["count"], row["n"])
    return rows


def run(llm, tokenizer, output_dir, limit=None):
    """Generate and score IFBench and IFEval; save benchmark, prompt and generated_response
    (in the benchmarks' order) to <output_dir>/if_responses.csv and return the score rows:
    each benchmark's and their unweighted mean (Avg)."""
    rows, tables = [], []
    for benchmark, examples in load_benchmarks(limit).items():
        print(f"\nEvaluating {benchmark} ({len(examples)} prompts)")
        responses = generate(llm, tokenizer, examples)
        rows += evaluate(benchmark, examples, responses)
        tables.append(pd.DataFrame({"benchmark": benchmark,
                                    "prompt": [example["prompt"] for example in examples],
                                    "generated_response": responses}))
    rows = [row for name in SCORE_NAMES
            for row in with_average([row for row in rows if row["metric"] == f"if_{name}"])]
    print("\nPrompt-level loose accuracy (the primary IFBench metric):")
    for row in rows:
        if row["metric"] == "if_prompt_loose":
            print(f"  {row['test_set']:<8} {row['score']:6.2f}%")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "if_responses.csv")
    pd.concat(tables, ignore_index=True).to_csv(path, index=False)
    print(f"Saved: {path}")
    return rows

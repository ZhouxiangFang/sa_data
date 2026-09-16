#!/usr/bin/env python3
"""Evaluate a model on IFBench and IFEval with vLLM.

Install the evaluator directly from GitHub:
    python -m pip install "ifbench @ git+https://github.com/allenai/IFBench.git"

Example:
    python eval_if.py --model qwen2.5-ins
"""

import argparse
import csv
import json
import os
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from helper import model_mapping


SCORE_NAMES = [
    "prompt_strict",
    "instruction_strict",
    "prompt_loose",
    "instruction_loose",
]


def resolve_model(name):
    """Check helper.py's model_mapping first, then use the name as given."""
    key = name.lower()
    if key in model_mapping:
        return model_mapping[key], key

    if os.path.exists(name):
        path = os.path.abspath(name)
        return path, os.path.basename(os.path.normpath(path))

    return name, name.replace("/", "_")


def load_benchmarks(limit=None):
    """Load the official IFBench and IFEval examples."""
    try:
        from ifbench import data_path
    except ImportError as error:
        raise SystemExit(
            "Install IFBench with:\n"
            "python -m pip install \"ifbench @ "
            "git+https://github.com/allenai/IFBench.git\""
        ) from error

    with open(data_path(), encoding="utf-8") as file:
        ifbench = [json.loads(line) for line in file if line.strip()]

    ifeval = list(load_dataset("google/IFEval", split="train"))

    if limit:
        ifbench = ifbench[:limit]
        ifeval = ifeval[:limit]

    return {"IFBench": ifbench, "IFEval": ifeval}


def format_prompts(tokenizer, examples):
    """Apply the chat template, or use raw prompts for base models."""
    prompts = [example["prompt"] for example in examples]
    if not tokenizer.chat_template:
        return prompts

    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def generate(llm, tokenizer, examples, max_tokens):
    """Generate one deterministic response for every example."""
    params = SamplingParams(temperature=0, max_tokens=max_tokens, seed=42)
    outputs = llm.generate(
        format_prompts(tokenizer, examples), params, use_tqdm=True
    )

    responses = []
    for output in outputs:
        response = output.outputs[0].text
        # Thinking models should be scored on their final answer only.
        if "</think>" in response:
            response = response.rsplit("</think>", 1)[1].lstrip()
        responses.append(response)
    return responses


def response_variants(response, loose):
    """Create the variants used by the official loose evaluation."""
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

    for instruction_id, kwargs in zip(
        example["instruction_id_list"], example["kwargs"]
    ):
        checker = INSTRUCTION_DICT[instruction_id](instruction_id)
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        checker.build_description(**kwargs)

        if "prompt" in (checker.get_instruction_args() or []):
            checker.build_description(prompt=example["prompt"])

        followed = any(
            text.strip() and checker.check_following(text) for text in variants
        )
        results.append(bool(followed))

    return results


def evaluate(benchmark, examples, responses, model_name, model_id):
    """Calculate strict and loose prompt/instruction accuracy."""
    strict_results = []
    loose_results = []

    for example, response in zip(examples, responses):
        strict_results.append(check_response(example, response, loose=False))
        loose_results.append(check_response(example, response, loose=True))

    num_prompts = len(examples)
    num_instructions = sum(len(result) for result in strict_results)

    return {
        "model": model_name,
        "model_id": model_id,
        "benchmark": benchmark,
        "num_prompts": num_prompts,
        "num_instructions": num_instructions,
        "prompt_strict": sum(all(result) for result in strict_results)
        / num_prompts,
        "instruction_strict": sum(map(sum, strict_results)) / num_instructions,
        "prompt_loose": sum(all(result) for result in loose_results) / num_prompts,
        "instruction_loose": sum(map(sum, loose_results)) / num_instructions,
    }


def add_average(rows):
    """Add an unweighted average of IFBench and IFEval."""
    average = {
        "model": rows[0]["model"],
        "model_id": rows[0]["model_id"],
        "benchmark": "Average",
        "num_prompts": sum(row["num_prompts"] for row in rows),
        "num_instructions": sum(row["num_instructions"] for row in rows),
    }
    for name in SCORE_NAMES:
        average[name] = sum(row[name] for row in rows) / len(rows)
    rows.append(average)


def save_csv(rows, output_dir, model_name):
    """Save one summary CSV containing both benchmarks and their average."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
    output_file = os.path.join(output_dir, f"{safe_name}_if_summary.csv")

    columns = [
        "model",
        "model_id",
        "benchmark",
        "num_prompts",
        "num_instructions",
        *SCORE_NAMES,
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return output_file


def print_results(rows):
    """Print the main score for each benchmark."""
    print("\nPrompt-level loose accuracy (the primary IFBench metric):")
    for row in rows:
        print(f"  {row['benchmark']:<8} {row['prompt_loose'] * 100:6.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="../results/if_eval")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, help="Use only the first N examples for a quick test"
    )
    args = parser.parse_args()

    model_id, model_name = resolve_model(args.model)
    print(f"Model: {args.model} -> {model_id}")

    benchmarks = load_benchmarks(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, tensor_parallel_size=args.tensor_parallel_size)

    rows = []
    for benchmark, examples in benchmarks.items():
        print(f"\nEvaluating {benchmark} ({len(examples)} prompts)")
        responses = generate(llm, tokenizer, examples, args.max_tokens)
        rows.append(evaluate(benchmark, examples, responses, args.model, model_id))

    add_average(rows)
    output_file = save_csv(rows, args.output_dir, model_name)
    print_results(rows)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()

"""IFBench and IFEval: generation and scoring (run by eval.py).

Install the evaluator directly from GitHub:
    python -m pip install "ifbench @ git+https://github.com/allenai/IFBench.git"
"""

import csv
import json
import os

SCORE_NAMES = ["prompt_strict", "instruction_strict", "prompt_loose", "instruction_loose"]
MAX_TOKENS = 4096


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


def evaluate(benchmark, examples, responses, model_name, model_id):
    """Strict and loose prompt/instruction accuracy."""
    strict, loose = [], []
    # Strict then loose per example, the official order (some checkers draw from `random`).
    for example, response in zip(examples, responses):
        strict.append(check_response(example, response))
        loose.append(check_response(example, response, loose=True))
    num_prompts = len(examples)
    num_instructions = sum(map(len, strict))
    return {
        "model": model_name,
        "model_id": model_id,
        "benchmark": benchmark,
        "num_prompts": num_prompts,
        "num_instructions": num_instructions,
        "prompt_strict": sum(map(all, strict)) / num_prompts,
        "instruction_strict": sum(map(sum, strict)) / num_instructions,
        "prompt_loose": sum(map(all, loose)) / num_prompts,
        "instruction_loose": sum(map(sum, loose)) / num_instructions,
    }


def add_average(rows):
    """Append the unweighted average of the benchmarks."""
    rows.append({
        "model": rows[0]["model"],
        "model_id": rows[0]["model_id"],
        "benchmark": "Average",
        "num_prompts": sum(row["num_prompts"] for row in rows),
        "num_instructions": sum(row["num_instructions"] for row in rows),
        **{name: sum(row[name] for row in rows) / len(rows) for name in SCORE_NAMES},
    })


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def run(llm, tokenizer, model_name, model_id, output_dir, prefix, limit=None):
    """Generate, score and save <prefix>_if_{summary,responses}.csv in output_dir."""
    rows, response_rows = [], []
    for benchmark, examples in load_benchmarks(limit).items():
        print(f"\nEvaluating {benchmark} ({len(examples)} prompts)")
        responses = generate(llm, tokenizer, examples)
        response_rows.extend(
            {"prompt": example["prompt"], "responses": response, "prompt_source": benchmark}
            for example, response in zip(examples, responses)
        )
        rows.append(evaluate(benchmark, examples, responses, model_name, model_id))
    add_average(rows)

    print("\nPrompt-level loose accuracy (the primary IFBench metric):")
    for row in rows:
        print(f"  {row['benchmark']:<8} {row['prompt_loose'] * 100:6.2f}%")
    os.makedirs(output_dir, exist_ok=True)
    write_csv(
        os.path.join(output_dir, f"{prefix}_if_summary.csv"),
        ["model", "model_id", "benchmark", "num_prompts", "num_instructions", *SCORE_NAMES],
        ({**row, **{name: f"{row[name]:.4f}" for name in SCORE_NAMES}} for row in rows),
    )
    write_csv(os.path.join(output_dir, f"{prefix}_if_responses.csv"),
              ["prompt", "responses", "prompt_source"], response_rows)
    return rows

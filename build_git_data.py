import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


LIMA = "GAIR/lima"
NO_ROBOTS = "HuggingFaceH4/no_robots"
TULU = "allenai/tulu-3-sft-personas-instruction-following"
DOLLY = "databricks/databricks-dolly-15k"


def sample_rows(df, n, seed, source):
    """Sample n rows without replacement."""
    if n < 0:
        raise ValueError(f"Sample size for {source} must be non-negative")
    if n > len(df):
        raise ValueError(f"Requested {n} rows from {source}, but only {len(df)} are available")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def clean_rows(rows, source):
    """Create a clean prompt/response/source DataFrame."""
    df = pd.DataFrame(rows, columns=["prompt", "response"])
    df["prompt"] = df["prompt"].astype("string").str.strip()
    df["response"] = df["response"].astype("string").str.strip()
    df = df.dropna(subset=["prompt", "response"])
    df = df[(df["prompt"] != "") & (df["response"] != "")].copy()
    df["source"] = source
    return df


def load_lima():
    # GAIR/lima uses a legacy loading script, so read its JSONL file directly.
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/GAIR/lima/train.jsonl",
        split="train",
    )
    rows = [
        (example["conversations"][0], example["conversations"][1])
        for example in dataset
        if len(example["conversations"]) == 2
    ]
    return clean_rows(rows, LIMA)


def load_message_dataset(dataset_name, excluded_categories=()):
    """Load examples containing exactly one user/assistant exchange."""
    dataset = load_dataset(dataset_name, split="train")
    excluded_categories = {category.lower() for category in excluded_categories}
    rows = []
    for example in dataset:
        if example.get("category", "").lower() in excluded_categories:
            continue
        messages = example["messages"]
        if len(messages) != 2:
            continue
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        rows.append((messages[0]["content"], messages[1]["content"]))
    return clean_rows(rows, dataset_name)


def load_dolly():
    dataset = load_dataset(DOLLY, split="train")
    rows = []
    for example in dataset:
        prompt = example["instruction"].strip()
        context = example["context"].strip()
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"
        rows.append((prompt, example["response"]))
    return clean_rows(rows, DOLLY)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a single-turn general instruction-tuning dataset."
    )
    parser.add_argument("--lima", type=int, default=1000)
    parser.add_argument("--no_robots", type=int, default=5000)
    parser.add_argument("--tulu", type=int, default=2000)
    parser.add_argument("--dolly", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        help="Output CSV path. By default, saves to data/git_data_{n}k.csv.",
    )
    return parser.parse_args()


def default_output_path(num_rows):
    num_thousands = f"{num_rows / 1000:g}"
    data_dir = Path(__file__).resolve().parent.parent / "data"
    return data_dir / f"git_data_{num_thousands}k.csv"


def main():
    args = parse_args()

    datasets = [
        sample_rows(load_lima(), args.lima, args.seed, LIMA),
        sample_rows(
            load_message_dataset(NO_ROBOTS, excluded_categories={"coding"}),
            args.no_robots,
            args.seed,
            NO_ROBOTS,
        ),
        sample_rows(load_message_dataset(TULU), args.tulu, args.seed, TULU),
        sample_rows(load_dolly(), args.dolly, args.seed, DOLLY),
    ]
    result = pd.concat(datasets, ignore_index=True)
    result = result.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    output_path = Path(args.output) if args.output else default_output_path(len(result))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved {len(result):,} examples to {output_path}")
    print(result["source"].value_counts().to_string())


if __name__ == "__main__":
    main()

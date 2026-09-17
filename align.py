"""Safety-tune a model on one safety subcategory plus benign examples.

Defaults are sized for full fine-tuning a 7B/8B model on four 48 GB A6000s::

    conda activate nlp
    deepspeed --num_gpus 4 align.py \
        --model qwen2.5-ins \
        --alignment_dataset wildguardmix \
        --abbr cyberattack \
        --harmful_rate 0.5

Only safe/unharmful assistant responses are used as training targets. With four
GPUs, the default micro-batch of 1 and eight accumulation steps give an
effective batch size of 32 examples.
"""

import argparse
import logging
import math
import os
import re
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from helper import load_safety_dataset, model_mapping, print_config


logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("Gloo").setLevel(logging.WARNING)

MODULE_DIR = Path(__file__).resolve().parent
SAFE_LABELS = {"safe", "unharmful"}
HARMFUL_LABELS = {"harmful", "unsafe"}
BENIGN_DATA_TYPES = ("vanilla_benign", "adversarial_benign", "mix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safety-tune on prompts from one dataset subcategory and "
            "benign prompt/response pairs from allenai/wildjailbreak."
        )
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-ins",
        help="Model shortcut from helper.model_mapping, or a Hugging Face model ID.",
    )
    parser.add_argument(
        "--alignment_dataset",
        choices=("wildguardmix", "aegis"),
        required=True,
        help="Dataset containing the selected safety subcategory.",
    )
    parser.add_argument(
        "--abbr",
        required=True,
        help="Subcategory abbreviation from subcategory_stats.json.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=800,
        help="Total number of harmful plus benign training examples (default: 800).",
    )
    parser.add_argument(
        "--harmful_rate",
        type=float,
        required=True,
        help="Fraction of examples that use harmful prompts, between 0 and 1.",
    )
    parser.add_argument(
        "--benign_data_type",
        dest="benign_data_type",
        choices=BENIGN_DATA_TYPES,
        default="vanilla_benign",
        help=(
            "WildJailbreak benign data to use. 'mix' samples from both vanilla "
            "and adversarial benign examples (default: vanilla_benign)."
        ),
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--learning_rate", "--lr", dest="learning_rate", type=float, default=5e-6
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Micro-batch per GPU (default: 1, conservative for 4096-token inputs).",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--dataset_num_proc",
        type=int,
        default=4,
        help="Processes used by TRL to format/tokenize the dataset (default: 4).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--deepspeed",
        type=Path,
        default=MODULE_DIR / "ds_config.json",
        help="DeepSpeed config (default: ds_config.json next to this script).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/scratch/zf28/ckpts"),
        help="Root directory for the trained checkpoint (default: /scratch/zf28/ckpts).",
    )
    parser.add_argument(
        "--run_name",
        help=(
            "Optional output subdirectory name. The default is "
            "<model>_<dataset>_<subcategory>_<total>."
        ),
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    if args.num_train <= 0:
        parser.error("--num_train must be positive")
    if not 0.0 <= args.harmful_rate <= 1.0:
        parser.error("--harmful_rate must be between 0 and 1")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup_ratio must be between 0 and 1")
    if args.weight_decay < 0:
        parser.error("--weight_decay must be non-negative")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive")
    if args.logging_steps <= 0:
        parser.error("--logging_steps must be positive")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    if args.dataset_num_proc <= 0:
        parser.error("--dataset_num_proc must be positive")
    if not args.deepspeed.is_file():
        parser.error(f"DeepSpeed config not found: {args.deepspeed}")
    return args


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().str.strip()


def clean_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing or empty prompt/response pairs."""
    df = df.dropna(subset=["prompt", "response"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    return df[(df["prompt"] != "") & (df["response"] != "")]


def load_harmful_subcategory(dataset_name: str, abbr: str) -> pd.DataFrame:
    """Load prompts in one subcategory with safe response targets."""
    df = load_safety_dataset(dataset_name, "train")
    prompt_label_column = (
        "prompt_harm_label" if dataset_name == "wildguardmix" else "prompt_label"
    )
    required = {
        "prompt",
        "response",
        "response_harm_label",
        prompt_label_column,
        "sub_abbrs",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")

    is_subcategory = df["sub_abbrs"].apply(
        lambda abbreviations: isinstance(abbreviations, (list, tuple, set))
        and abbr in abbreviations
    )
    prompt_labels = SAFE_LABELS if abbr.lower() == "benign" else HARMFUL_LABELS
    has_target_prompt_label = normalize_label(df[prompt_label_column]).isin(
        prompt_labels
    )
    is_safe_response = normalize_label(df["response_harm_label"]).isin(SAFE_LABELS)
    return clean_pairs(df[is_subcategory & has_target_prompt_label & is_safe_response])


def load_wildjailbreak_benign(data_type: str) -> pd.DataFrame:
    """Load benign prompt/completion pairs from WildJailbreak."""
    df = load_dataset(
        "allenai/wildjailbreak",
        "train",
        split="train",
        delimiter="\t",
        keep_default_na=False,
    ).to_pandas()
    selected_types = (
        ["vanilla_benign", "adversarial_benign"]
        if data_type == "mix"
        else [data_type]
    )
    df = df[df["data_type"].isin(selected_types)].copy()
    df["prompt"] = df.apply(
        lambda row: (
            row["adversarial"]
            if row["data_type"] == "adversarial_benign"
            else row["vanilla"]
        ),
        axis=1,
    )
    df.rename(columns={"completion": "response"}, inplace=True)
    return clean_pairs(df)


def sample_pairs(
    df: pd.DataFrame,
    count: int,
    source: str,
    seed: int,
    tokenizer,
    max_length: int,
) -> tuple[pd.DataFrame, int]:
    """Sample pairs whose formatted prompt leaves room for completion tokens."""
    if count == 0:
        return pd.DataFrame(columns=["prompt", "response"]), 0
    if count > len(df):
        raise ValueError(
            f"Requested {count:,} examples from {source}, but only {len(df):,} are available"
        )

    shuffled = df.sample(frac=1, random_state=seed)
    selected_indices: list[int] = []
    overlong_count = 0
    # Work in bounded batches so a large WildJailbreak pool does not create a
    # second full in-memory copy of every formatted prompt.
    batch_size = max(256, min(2048, count * 2))
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled.iloc[start : start + batch_size]
        conversations = [
            [{"role": "user", "content": prompt}] for prompt in batch["prompt"]
        ]
        formatted_prompts = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        lengths = tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            return_length=True,
            truncation=False,
        )["length"]
        for index, length in zip(batch.index, lengths):
            if length < max_length:
                selected_indices.append(index)
                if len(selected_indices) == count:
                    selected = df.loc[selected_indices, ["prompt", "response"]]
                    return selected.reset_index(drop=True), overlong_count
            else:
                overlong_count += 1

    raise ValueError(
        f"Requested {count:,} examples from {source}, but only "
        f"{len(selected_indices):,} have a formatted prompt shorter than "
        f"--max_length {max_length:,}"
    )


def build_training_data(
    args: argparse.Namespace, tokenizer
) -> tuple[pd.DataFrame, int, int, int]:
    # Round to the nearest example; the benign count absorbs any remainder.
    harmful_count = int(args.num_train * args.harmful_rate + 0.5)
    benign_count = args.num_train - harmful_count

    harmful_pool = (
        load_harmful_subcategory(args.alignment_dataset, args.abbr)
        if harmful_count
        else pd.DataFrame(columns=["prompt", "response"])
    )
    benign_pool = (
        load_wildjailbreak_benign(args.benign_data_type)
        if benign_count
        else pd.DataFrame(columns=["prompt", "response"])
    )

    harmful, harmful_overlong = sample_pairs(
        harmful_pool,
        harmful_count,
        f"subcategory {args.abbr!r} in {args.alignment_dataset}",
        args.seed,
        tokenizer,
        args.max_length,
    )
    benign, benign_overlong = sample_pairs(
        benign_pool,
        benign_count,
        f"WildJailbreak {args.benign_data_type!r}",
        args.seed + 1,
        tokenizer,
        args.max_length,
    )
    train_df = pd.concat([harmful, benign], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    return (
        train_df,
        len(harmful_pool),
        len(benign_pool),
        harmful_overlong + benign_overlong,
    )


def filename_component(value: str) -> str:
    """Return a readable, filesystem-safe checkpoint-name component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return component or "run"


def main() -> None:
    args = parse_args()
    is_main_process = args.local_rank in (-1, 0)
    if is_main_process:
        print("Starting alignment training:")
        print_config(args)

    model_id = model_mapping.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.eos_token is None:
        raise ValueError(f"{model_id} has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(
            f"{model_id} has no chat template. Use an instruct model or set a chat template."
        )

    harmful_count = int(args.num_train * args.harmful_rate + 0.5)
    benign_count = args.num_train - harmful_count
    train_df, harmful_available, benign_available, overlong_count = (
        build_training_data(args, tokenizer)
    )
    if is_main_process:
        prompt_kind = "benign" if args.abbr.lower() == "benign" else "harmful"
        print(
            f"Training mixture: {harmful_count:,} {prompt_kind} prompts from "
            f"{args.alignment_dataset}/{args.abbr} ({harmful_available:,} available) + "
            f"{benign_count:,} benign prompts from WildJailbreak/"
            f"{args.benign_data_type} ({benign_available:,} available)"
        )
        print(f"Overlong candidates skipped while sampling: {overlong_count:,}")

    train_dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": prompt}] for prompt in train_df["prompt"]
            ],
            "completion": [
                [{"role": "assistant", "content": response}]
                for response in train_df["response"]
            ],
        }
    )

    checkpoint_name = args.run_name or (
        f"{filename_component(args.model)}_{args.alignment_dataset}_"
        f"{filename_component(args.abbr)}_{args.num_train}"
    )
    checkpoint_name = filename_component(checkpoint_name)
    output_dir = args.output_dir / checkpoint_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose --run_name or pass "
            "--overwrite_output_dir if reusing it is intentional."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = (
        world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    updates_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    if is_main_process:
        print(f"Model: {model_id}")
        print(f"Output directory: {output_dir}")
        print(
            f"Effective batch size: {effective_batch_size} "
            f"({world_size} GPUs x {args.per_device_train_batch_size} micro-batch "
            f"x {args.gradient_accumulation_steps} accumulation)"
        )
        print(
            f"Planned optimizer updates: about {updates_per_epoch * args.epochs} "
            f"({updates_per_epoch} per epoch x {args.epochs} epochs)"
        )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=str(args.deepspeed),
        bf16=True,
        tf32=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        eos_token=tokenizer.eos_token,
        max_length=args.max_length,
        packing=False,
        dataset_num_proc=args.dataset_num_proc,
        seed=args.seed,
        data_seed=args.seed,
        local_rank=args.local_rank,
        overwrite_output_dir=args.overwrite_output_dir,
    )

    # Construct SFTConfig first so Transformers initializes ZeRO-3 before the model.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    if is_main_process:
        print("Start fine-tuning...")
    trainer.train()

    trainer.model.config.use_cache = True
    if trainer.accelerator.is_main_process:
        print(f"Saving model to {output_dir}")
    # Trainer handles ZeRO-3 weight gathering and only writes on the main process.
    trainer.save_model(str(output_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        print("Model saved successfully.")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

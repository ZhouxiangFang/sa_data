"""Safety-tune on harmful safety, benign safety, and general instructions.

Defaults are sized for full fine-tuning a 7B/8B model on four 48 GB A6000s::

    conda activate nlp
    deepspeed --num_gpus 4 align.py \
        --model qwen2.5-ins \
        --alignment_dataset wildguardmix \
        --abbr cyberattack

The default 1,800 examples mix the three sources 1:1:1 (600 each). General
instructions default to data/git_data_20k.csv. Safety targets use only
safe/unharmful responses (in WildGuardMix these include compliances judged harmless);
--response_filter refusal (WildGuardMix only) keeps only refusals, and --dedup keeps
one row per prompt (the first; default: every row). Both change the pools of every
stage alike (helper.eligible_rows) and add their tag to the sample folder, so give
such runs their own --output_dir and results folder (chili/run.sh does).
--seed (default 43) decides every sampled example
(through helper.sample_examples, the same code calculate_diversity.py samples
with), the shuffle and the training order. The examples are read from the run's
sample files under data/train_samples/seed<seed>/ (found from the seed and the
run settings; drawn and saved there by the first run or diversity score that
needs them), so every model trained with a seed, and its diversity score, use
the same examples; the log prints each file with its SHA-256. The checkpoint
name ends in _seed<seed>, and eval.py writes its results to <results>/seed<seed>/
(IF results to <results>/seed<seed>/if_eval/). Each optimizer update uses
--effective_batch_size examples (default 32) at learning rate 1e-5; with four
GPUs and the default micro-batch of 1 that is eight accumulation steps.
train_jobs.py lists the jobs of a whole run.

Use --alignment_dataset gretel --abbr information_hazards for Gretel's
risk categories. Gretel safe_response supplies the response training target.

Use --abbr random or --abbr evenly for baselines over the non-benign
subcategories with enough examples to train on their own: random samples their
pooled prompts uniformly, so larger subcategories contribute more; evenly gives
each subcategory the same share.

Use --group_by tactic (Gretel only) to train on one jailbreak tactic instead,
e.g. --alignment_dataset gretel --group_by tactic --abbr phishing. Tactic rows
pass the same filters as the subcategories (helper.tactic_pools); random and
evenly then pool the tactics with enough rows for harmful_count.

Use --group_by selection to train on a subset chosen by select_data.py: --abbr is
then the subset's name, e.g. --abbr semantic_vendi_bge-m3 or --abbr lexical_hdd,
read from data/seed<seed>/<dataset>/<abbr>_<harmful count>.csv. The harmful part is
that whole subset (it must hold exactly the harmful count), and the benign and
general parts are the seed's usual samples. Train these runs into their own
--output_dir and evaluate them into ablations/selected (chili/run.sh --selected).
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from helper import (
    BENIGN_DATA_TYPES, DEFAULT_SEED, GROUP_BYS, TRAIN_SAMPLES_DIR, add_pool_arguments,
    build_training_data, checkpoint_name, filename_component, mixture_counts, model_mapping,
    pool_options, print_config, sample_digest,
)


logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("Gloo").setLevel(logging.WARNING)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_NUM_TRAIN = 1800
DEFAULT_RATE = 1 / 3
DEFAULT_GENERAL_DATASET = MODULE_DIR.parent / "data" / "git_data_20k.csv"
# Checkpoint root: CKPT_ROOT (set by range/common.sh), else /scratch/zf28/ckpts.
DEFAULT_OUTPUT_DIR = Path(os.environ.get("CKPT_ROOT", "/scratch/zf28/ckpts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safety-tune on prompts from one dataset subcategory and "
            "benign pairs from allenai/wildjailbreak, plus general instructions."
        )
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-ins",
        help="Model shortcut from helper.model_mapping, or a Hugging Face model ID.",
    )
    parser.add_argument(
        "--alignment_dataset",
        choices=("wildguardmix", "aegis", "gretel"),
        required=True,
        help="Dataset containing the selected safety subcategory.",
    )
    parser.add_argument(
        "--abbr",
        required=True,
        help=(
            "Subcategory abbreviation from data/subcategory_stats.json. "
            "For Gretel, use its configured abbreviation (e.g. 'information_hazards'). "
            "Baselines over the non-benign subcategories with enough examples to "
            "train on their own: 'random' samples them in proportion to their size; "
            "'evenly' samples each equally."
        ),
    )
    parser.add_argument(
        "--group_by",
        choices=GROUP_BYS,
        default="subcategory",
        help=("Select harmful prompts by subcategory, by jailbreak tactic (gretel only; "
              "--abbr is then a tactic abbreviation), or take a subset chosen by "
              "select_data.py (selection; --abbr is then the subset's name)."),
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=DEFAULT_NUM_TRAIN,
        help="Total examples across all three sources (default: 1800).",
    )
    parser.add_argument(
        "--harmful_rate",
        type=float,
        default=DEFAULT_RATE,
        help="Fraction from the selected safety subcategory (default: 1/3).",
    )
    parser.add_argument(
        "--general_rate", type=float, default=DEFAULT_RATE,
        help="General instruction fraction (default: 1/3); benign uses the remainder.",
    )
    parser.add_argument(
        "--general_dataset", type=Path, default=DEFAULT_GENERAL_DATASET,
        help="General instruction prompt/response CSV (default: data/git_data_20k.csv).",
    )
    parser.add_argument(
        "--benign_data_type",
        dest="benign_data_type",
        choices=BENIGN_DATA_TYPES,
        default="vanilla_benign",
        help=(
            "WildJailbreak benign data to use. 'mix' samples equal halves of vanilla "
            "and adversarial benign examples (default: vanilla_benign)."
        ),
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--learning_rate", "--lr", dest="learning_rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Micro-batch per GPU (default: 1, conservative for 4096-token inputs).",
    )
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        default=32,
        help=(
            "Examples per optimizer update over all GPUs (default: 32). Gradient "
            "accumulation is this / (GPUs x micro-batch), e.g. 8 steps on four GPUs."
        ),
    )
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
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=(f"Run seed (default: {DEFAULT_SEED}); decides the sampled examples and the "
              "training order. The checkpoint name ends in _seed<seed>."),
    )
    parser.add_argument(
        "--samples_dir", type=Path, default=TRAIN_SAMPLES_DIR,
        help="Root of the per-seed sample files (default: data/train_samples).",
    )
    add_pool_arguments(parser)
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
        help=f"Root directory for the trained checkpoint (default: {DEFAULT_OUTPUT_DIR}, "
             "plus /<tag> with --response_filter refusal or --dedup).",
    )
    parser.add_argument(
        "--run_name",
        help=(
            "Optional output subdirectory name. The default is "
            "<model>_<num_train>_<dataset>_<subcategory>_seed<seed>."
        ),
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    try:
        _, _, general_count = mixture_counts(
            args.num_train, args.harmful_rate, args.general_rate
        )
    except ValueError as error:
        parser.error(str(error))
    if args.group_by == "tactic" and args.alignment_dataset != "gretel":
        parser.error("--group_by tactic requires --alignment_dataset gretel")
    try:
        pool_options(args).check(args.alignment_dataset)
    except ValueError as error:
        parser.error(str(error))
    # Runs with other pool options reuse the checkpoint names, so keep them apart.
    args.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / pool_options(args).tag
    args.general_dataset = args.general_dataset.expanduser()
    if general_count and not args.general_dataset.is_file():
        parser.error(f"General instruction CSV not found: {args.general_dataset}")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size must be positive")
    # The DeepSpeed launcher sets WORLD_SIZE before this script starts.
    per_step = int(os.environ.get("WORLD_SIZE", "1")) * args.per_device_train_batch_size
    if args.effective_batch_size <= 0 or args.effective_batch_size % per_step:
        parser.error(
            f"--effective_batch_size must be a positive multiple of GPUs x micro-batch "
            f"({per_step}), got {args.effective_batch_size}"
        )
    args.gradient_accumulation_steps = args.effective_batch_size // per_step
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

    harmful_count, benign_count, general_count = mixture_counts(
        args.num_train, args.harmful_rate, args.general_rate
    )
    train_df, harmful_available, benign_available, overlong_count = (
        build_training_data(args, tokenizer, samples_dir=args.samples_dir)
    )
    if is_main_process:
        prompt_kind = "benign" if args.abbr.lower() == "benign" else "harmful"
        print(
            f"Training mixture: {harmful_count:,} {prompt_kind} prompts from "
            f"{args.alignment_dataset} {args.group_by} {args.abbr} "
            f"({harmful_available:,} available; {args.response_filter} responses"
            f"{', one row per prompt' if args.dedup else ''}) + "
            f"{benign_count:,} benign prompts from WildJailbreak/"
            f"{args.benign_data_type} ({benign_available:,} available) + "
            f"{general_count:,} general instructions from {args.general_dataset} "
            f"({train_df.attrs['general_available']:,} available)"
        )
        pooled = train_df.attrs.get("baseline_groups")
        if pooled:
            print(f"Baseline {args.group_by} groups ({len(pooled)}): {', '.join(pooled)}")
        subcategory_counts = train_df.attrs.get("harmful_subcategory_counts")
        if subcategory_counts:
            print(f"Examples per {args.group_by}:")
            for abbr, count in sorted(subcategory_counts.items()):
                print(f"  {abbr}: {count:,}")
        print(f"Overlong candidates skipped while sampling: {overlong_count:,}")
        for part, path in train_df.attrs["sample_files"].items():
            print(f"Sample file ({part}): {path} sha256 {sample_digest(path)}")

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

    output_dir = args.output_dir / (
        filename_component(args.run_name) if args.run_name else
        checkpoint_name(args.model, args.num_train, args.alignment_dataset, args.abbr, args.seed)
    )
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

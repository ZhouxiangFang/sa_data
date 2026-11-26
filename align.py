"""Safety-tune one git20k model on one harmful training set for one seed.

    deepspeed --num_gpus 2 align.py --model llama3.1_git20k --abbr cyberattack --seed 43 \
        --output_dir /scratch/zf28/ckpts/seed43/default --per_device_train_batch_size 4
    deepspeed --num_gpus 2 align.py --model llama3.1_git20k --selected wildjailbreak \
        --abbr semantic_vendi --seed 43 --output_dir /scratch/zf28/ckpts/seed43/selected_wildjailbreak

The --num_train examples (default 1,800) are a third each: harmful prompts of the subcategory
with their unharmful WildGuardMix responses (--response_label_filter refusal: refusals only),
WildJailbreak vanilla_benign pairs and data/git_data_20k.csv instructions, read from the seed's
sample files (helper.training_examples; the log prints each file's SHA-256). --abbr random or
evenly trains a baseline over every subcategory large enough to train on. Stage 2: with
--selected DATASET the harmful part is the subset select_data.py selected from DATASET with
method --abbr (samples/stage2_selected/seed<seed>/<dataset>[_refusal]/<abbr>_<harmful count>.csv).
The checkpoint is <output_dir>/<model>_<num_train>_<dataset>_<abbr>_seed<seed>, with run.json: the
run's fingerprint (helper.run_fingerprint: base model, seed, training settings, data files'
contents), which train_jobs.py checks. Each optimizer update uses --effective_batch_size
examples; accumulation = that / (GPUs x micro-batch).
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from helper import (DATASET, DEFAULT_NUM_TRAIN, DEFAULT_SEED, RUN_FILE, SELECTION_DATASETS,
                    TRAINING_SETTINGS, add_pool_argument, add_training_arguments,
                    checkpoint_name, mixture_counts, model_mapping, run_fingerprint,
                    sample_digest, training_examples)

MODULE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Model shortcut (helper.model_mapping, e.g. llama3.1_git20k) or "
                             "Hugging Face model ID")
    parser.add_argument("--abbr", required=True,
                        help="WildGuardMix subcategory abbreviation (data/subcategory_stats.json), "
                             "random / evenly, or with --selected the selection method")
    parser.add_argument("--selected", choices=SELECTION_DATASETS,
                        help="Stage 2: train on the subset selected from this dataset")
    parser.add_argument("--mixed_diversity", action="store_true",
                        help="With --selected: the subset of select_data.py's mixed-diversity "
                             "ablation")
    parser.add_argument("--with_unnatural", action="store_true",
                        help="With --selected: the subset of select_data.py's --with_unnatural "
                             "ablation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Run seed: the sampled examples and the training order")
    parser.add_argument("--num_train", type=int, default=DEFAULT_NUM_TRAIN)
    add_pool_argument(parser)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Folder of the checkpoint folder")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    add_training_arguments(parser)  # epochs, learning_rate, effective_batch_size, max_length
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--deepspeed", type=Path, default=MODULE_DIR / "range" / "ds_zero2.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    try:
        mixture_counts(args.num_train)
    except ValueError as error:
        parser.error(str(error))
    # The DeepSpeed launcher sets WORLD_SIZE before this script starts.
    per_step = int(os.environ.get("WORLD_SIZE", "1")) * args.per_device_train_batch_size
    if args.effective_batch_size <= 0 or args.effective_batch_size % per_step:
        parser.error(f"--effective_batch_size must be a positive multiple of GPUs x micro-batch "
                     f"({per_step})")
    args.gradient_accumulation_steps = args.effective_batch_size // per_step
    if not args.deepspeed.is_file():
        parser.error(f"DeepSpeed config not found: {args.deepspeed}")
    return args


def main():
    args = parse_args()
    main_process = args.local_rank in (-1, 0)
    model_id = model_mapping.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.eos_token is None or tokenizer.chat_template is None:
        raise ValueError(f"{model_id} needs an EOS token and a chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df, info = training_examples(args.abbr, args.seed, tokenizer, args.num_train,
                                       args.response_label_filter, args.max_length,
                                       selected=args.selected, mixed=args.mixed_diversity,
                                       with_unnatural=args.with_unnatural)
    dataset = args.selected or DATASET
    output_dir = args.output_dir / checkpoint_name(args.model, args.num_train, args.abbr,
                                                   args.seed, dataset)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    settings = {name: getattr(args, name) for name in TRAINING_SETTINGS}
    run = {"fingerprint": run_fingerprint(args.model, args.seed, info["files"], settings),
           "model": model_id, "seed": args.seed, **settings,
           "files": {part: {"path": str(path), "sha256": sample_digest(path)}
                     for part, path in info["files"].items()}}
    if main_process:
        print("Configuration:", *(f"  {key}: {value}" for key, value in vars(args).items()),
              sep="\n")
        counts = dict(zip(("harmful", "benign", "general"), mixture_counts(args.num_train)))
        for part, path in info["files"].items():
            print(f"{part}: {counts[part]:,} of {info['available'][part]:,} examples, "
                  f"{path} sha256 {sample_digest(path)}")
        print(f"Harmful part: {dataset} {args.abbr}, {args.response_label_filter} responses; "
              f"overlong prompts skipped while drawing: {info['overlong']:,}")
        print(f"Model: {model_id}\nOutput directory: {output_dir}\n"
              f"Fingerprint: {run['fingerprint']}")
        updates = math.ceil(len(train_df) / args.effective_batch_size) * args.epochs
        print(f"Effective batch {args.effective_batch_size} ({world_size} GPUs x "
              f"{args.per_device_train_batch_size} x {args.gradient_accumulation_steps} "
              f"accumulation), about {updates} updates")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": prompt}] for prompt in train_df["prompt"]],
        "completion": [[{"role": "assistant", "content": response}]
                       for response in train_df["response"]],
    })
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=str(args.deepspeed),
        bf16=True,
        tf32=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        eos_token=tokenizer.eos_token,
        max_length=args.max_length,
        packing=False,
        dataset_num_proc=4,
        seed=args.seed,
        data_seed=args.seed,
        local_rank=args.local_rank,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    # SFTConfig first, so Transformers initializes DeepSpeed before the model is loaded.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset,
                         processing_class=tokenizer)
    trainer.train()

    trainer.model.config.use_cache = True
    trainer.save_model(str(output_dir))  # gathers sharded weights; writes on the main process
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        # train_jobs.py compares the fingerprint with the planned run's.
        (output_dir / RUN_FILE).write_text(json.dumps(run, indent=1) + "\n")
        # Written last: train_jobs.py takes the tokenizer as the mark of a complete save.
        tokenizer.save_pretrained(output_dir)
        print(f"Saved {output_dir}")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

"""Does align.py's training fit this node's GPUs, and how long does a run take?

    deepspeed nersc/memtest.py --model llama3.1_git20k --general GIT_DATA_CSV \
        [--deepspeed nersc/ds_zero3.json] [--per_device_train_batch_size 1] [--save_dir DIR]

Trains the model with align.py's settings (its SFTConfig: gradient checkpointing, bf16, the
4,096-token cap, completion-only loss; the DeepSpeed config; effective batch 32) for a few
updates and prints each update's time and peak GPU memory (the maximum over the GPUs). Nothing
is read from or written to data/:
  --long_updates   first, updates of examples truncated to --max_length tokens, the worst case
                   for memory (with a micro-batch of 1 the longest sequence sets the peak);
  --updates        then a mixture like a run's: a third each of WildGuardMix harmful prompts
                   with unharmful responses, WildJailbreak vanilla_benign and --general (e.g.
                   build_git_data.py --output elsewhere), drawn in memory, which times a real
                   run's updates (2 epochs of 1,800 examples are 114 updates).
--save_dir also times saving the trained model as align.py does (ZeRO-3 gathers the weights).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sa_data, for helper
from helper import (DEFAULT_MAX_LENGTH, DEFAULT_NUM_TRAIN, clean_pairs, eligible_rows,
                    load_wildjailbreak_benign, mixture_counts, model_mapping, sample_pairs)

NERSC_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--general", type=Path, required=True,
                        help="CSV of general prompt/response pairs (git_data_20k.csv's format)")
    parser.add_argument("--long_updates", type=int, default=2)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--effective_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--deepspeed", type=Path, default=NERSC_DIR / "ds_zero3.json")
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path(os.environ.get(
        "SCRATCH", "/tmp")) / "memtest" / "trainer")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    per_step = int(os.environ.get("WORLD_SIZE", "1")) * args.per_device_train_batch_size
    if args.effective_batch_size % per_step:
        parser.error(f"--effective_batch_size must be a multiple of GPUs x micro-batch ({per_step})")
    args.gradient_accumulation_steps = args.effective_batch_size // per_step
    return args


def harmful_pool():
    """WildGuardMix train's harmful prompts with unharmful responses (helper.load_train without
    the subcategory abbreviations, which need data/subcategory_stats.json)."""
    df = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train").to_pandas()
    return eligible_rows(df[df["prompt"].notna() & df["response"].notna()])


def long_examples(general, count, tokenizer, max_length):
    """count prompt/completion pairs of at least max_length formatted tokens: a general prompt
    with general responses joined until the pair is long enough (truncated in training)."""
    responses = general["response"].tolist()
    examples, cursor = [], 0
    for prompt in general["prompt"].iloc[:count]:
        parts = []
        while True:
            parts.append(responses[cursor % len(responses)])
            cursor += 1
            completion = "\n\n".join(parts)
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
                tokenize=True)
            if len(ids) > max_length + 64:
                break
        examples.append((prompt, completion))
    return pd.DataFrame(examples, columns=["prompt", "response"])


class Probe(TrainerCallback):
    """Prints each update's wall time and peak memory, the maximum over the ranks."""

    def __init__(self, long_updates):
        self.long_updates = long_updates
        self.rows = []

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        free, total = torch.cuda.mem_get_info()
        stats = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(),
                              total - free, now - self.start], device="cuda", dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        torch.cuda.reset_peak_memory_stats()
        self.start = now
        allocated, reserved, used, seconds = stats.tolist()
        phase = "long" if state.global_step <= self.long_updates else "mixture"
        self.rows.append((phase, seconds, allocated, reserved, used))
        if state.is_world_process_zero:
            print(f"update {state.global_step:3d} {phase:7}: {seconds:6.1f} s, peak allocated "
                  f"{allocated / 2**30:5.1f} GiB, reserved {reserved / 2**30:5.1f} GiB, "
                  f"device used now {used / 2**30:5.1f} of {total / 2**30:.1f} GiB", flush=True)


class SequentialSFTTrainer(SFTTrainer):
    """Keeps the dataset order, so the long examples come first."""

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None
                                 else self.train_dataset)


def main():
    args = parse_args()
    main_process = args.local_rank in (-1, 0)
    model_id = model_mapping.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    general = clean_pairs(pd.read_csv(args.general))
    count = args.updates * args.effective_batch_size
    harmful_count, benign_count, general_count = mixture_counts(max(count, DEFAULT_NUM_TRAIN))
    parts = [sample_pairs(pool, n, name, args.seed, tokenizer, args.max_length)[0]
             for pool, n, name in ((harmful_pool(), harmful_count, "harmful"),
                                   (load_wildjailbreak_benign(), benign_count, "benign"),
                                   (general, general_count, "general"))]
    mixture = (pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed)
               .iloc[:count])
    long = long_examples(general.sample(frac=1, random_state=args.seed),
                         args.long_updates * args.effective_batch_size, tokenizer,
                         args.max_length)
    train_df = pd.concat([long, mixture], ignore_index=True)
    lengths = [len(tokenizer.apply_chat_template(
        [{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=True))
        for p, r in zip(mixture["prompt"], mixture["response"])]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if main_process:
        print(f"Model {model_id}; {world_size} GPUs x {args.per_device_train_batch_size} x "
              f"{args.gradient_accumulation_steps} accumulation; DeepSpeed {args.deepspeed}")
        print(f"{len(long)} long examples (truncated to {args.max_length}), then {len(mixture)} "
              f"mixture examples: mean {sum(lengths) / len(lengths):.0f} tokens, max "
              f"{max(lengths)}, {sum(n > args.max_length for n in lengths)} over the cap",
              flush=True)

    train_dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": prompt}] for prompt in train_df["prompt"]],
        "completion": [[{"role": "assistant", "content": response}]
                       for response in train_df["response"]],
    })
    training_args = SFTConfig(  # align.py's, but for max_steps and no logs
        output_dir=str(args.output_dir),
        max_steps=args.long_updates + args.updates,
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
        logging_steps=1,
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
    )
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    probe = Probe(args.long_updates)
    trainer = SequentialSFTTrainer(model=model, args=training_args, train_dataset=train_dataset,
                                   processing_class=tokenizer, callbacks=[probe])
    if main_process:
        print(f"Model loaded and data tokenized in {time.perf_counter() - load_start:.0f} s",
              flush=True)
    trainer.train()

    if args.save_dir:
        save_start = time.perf_counter()
        trainer.save_model(str(args.save_dir))
        trainer.accelerator.wait_for_everyone()
        if main_process:
            print(f"Saved to {args.save_dir} in {time.perf_counter() - save_start:.0f} s")
    if main_process:
        def summary(phase):
            rows = [row for row in probe.rows if row[0] == phase]
            # The first update includes DeepSpeed's warm-up, so the mean time skips it.
            times = [row[1] for row in rows[1:]] or [rows[0][1]]
            return (f"{phase}: peak allocated {max(r[2] for r in rows) / 2**30:.1f} GiB, "
                    f"reserved {max(r[3] for r in rows) / 2**30:.1f} GiB, "
                    f"{sum(times) / len(times):.1f} s per update")
        per_update = [row[1] for row in probe.rows if row[0] == "mixture"][1:]
        updates = math.ceil(DEFAULT_NUM_TRAIN / args.effective_batch_size) * 2
        print("SUMMARY", args.model, "|", summary("long"), "|", summary("mixture"), "|",
              f"~{updates * sum(per_update) / max(len(per_update), 1) / 60:.0f} min for "
              f"{updates} updates", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

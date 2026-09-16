"""Instruction-tune a pretrained model on a prompt/response CSV."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import SFTConfig, SFTTrainer

from helper import model_mapping


CSV_REQUIRED_COLUMNS = ("prompt", "response")

# Train with the base tokenizer, using the corresponding official instruct
# chat template and EOS token without changing vocabulary IDs.
instruct_tokenizer_mapping = {
    "llama3": "llama3-ins",
    "llama3.1": "llama3.1-ins",
    "qwen2.5": "qwen2.5-ins",
    "qwen3": "qwen3-ins",
    "olmo2": "olmo2-ins",
    "mistral": "mistral-ins",
}


def configure_chat_tokenizer(tokenizer, template_tokenizer, generation_config):
    """Reuse an instruct format without changing the base model's vocabulary."""
    if not template_tokenizer.chat_template:
        raise ValueError(f"{template_tokenizer.name_or_path} has no chat template")

    vocab = tokenizer.get_vocab()
    if vocab != template_tokenizer.get_vocab():
        raise ValueError("Base and instruct token-to-ID mappings differ")

    # Matching vocabularies must also use the same tokenization rules.
    if not tokenizer.is_fast or not template_tokenizer.is_fast:
        raise ValueError("Fast tokenizers are required to verify tokenizer compatibility")
    base_backend = json.loads(tokenizer.backend_tokenizer.to_str())
    instruct_backend = json.loads(template_tokenizer.backend_tokenizer.to_str())
    for component in ("model", "normalizer", "pre_tokenizer", "added_tokens", "decoder"):
        if base_backend.get(component) != instruct_backend.get(component):
            raise ValueError(f"Base and instruct tokenizer {component} differ")
    if tokenizer.bos_token != template_tokenizer.bos_token:
        raise ValueError("Base and instruct BOS tokens differ")

    eos_token = template_tokenizer.eos_token
    if eos_token not in vocab or tokenizer.encode(
        eos_token, add_special_tokens=False
    ) != [vocab[eos_token]]:
        raise ValueError(f"Instruct EOS {eos_token!r} is not a single base-tokenizer token")

    # Keep the old EOS available as padding when no pad token is defined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = template_tokenizer.pad_token or tokenizer.eos_token
    tokenizer.chat_template = template_tokenizer.chat_template
    tokenizer.eos_token = eos_token

    # Preserve official generation stops and include the training EOS.
    stop_ids = generation_config.eos_token_id
    if stop_ids is None:
        stop_ids = []
    elif isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    else:
        stop_ids = list(stop_ids)
    if tokenizer.eos_token_id not in stop_ids:
        stop_ids.append(tokenizer.eos_token_id)
    if not set(stop_ids).issubset(vocab.values()):
        raise ValueError("Generation stop IDs are not in the base tokenizer vocabulary")
    generation_config.eos_token_id = stop_ids
    generation_config.pad_token_id = tokenizer.pad_token_id


def load_csv_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in CSV_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}; found {list(df.columns)}"
        )

    df = df.dropna(subset=list(CSV_REQUIRED_COLUMNS)).copy()
    for column in CSV_REQUIRED_COLUMNS:
        df[column] = df[column].astype(str).str.strip()
    return df[(df["prompt"] != "") & (df["response"] != "")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instruction-tune a pretrained model on a prompt/response CSV."
    )
    parser.add_argument(
        "--model",
        choices=sorted(instruct_tokenizer_mapping),
        default="qwen3",
        help="Pretrained model shortcut from helper.model_mapping.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        help="Number of examples to sample. By default, use the full CSV.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.deepspeed.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed}")

    model_id = model_mapping[args.model]
    template_tokenizer_id = model_mapping[instruct_tokenizer_mapping[args.model]]
    train_df = load_csv_dataset(args.dataset)
    # Preserve the CSV's size label even when training filters or samples rows.
    size_match = re.search(r"_(\d+(?:\.\d+)?k)$", args.dataset.stem)
    data_size = size_match.group(1) if size_match else f"{len(train_df) / 1000:g}k"
    output_dir = args.output_dir / f"{args.model}_git{data_size}"

    # Use base weights and tokenization with the official instruct format.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    template_tokenizer = AutoTokenizer.from_pretrained(
        template_tokenizer_id,
        trust_remote_code=True,
    )
    generation_config = GenerationConfig.from_pretrained(template_tokenizer_id)
    configure_chat_tokenizer(tokenizer, template_tokenizer, generation_config)
    del template_tokenizer

    # TRL truncates sequences from the right. Drop prompts that already fill the
    # context window; otherwise their assistant response would receive no loss.
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in train_df["prompt"]
    ]
    prompt_lengths = tokenizer(
        formatted_prompts,
        add_special_tokens=False,
        return_length=True,
    )["length"]
    keep = [length < args.max_length for length in prompt_lengths]
    num_dropped = len(keep) - sum(keep)
    train_df = train_df.loc[keep]

    if args.num_train is not None:
        if not 0 < args.num_train <= len(train_df):
            raise ValueError(
                f"--num_train must be between 1 and {len(train_df):,}; got {args.num_train}"
            )
        train_df = train_df.sample(n=args.num_train, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)

    # Prompt-completion format lets TRL mask prompt tokens while still applying
    # the tokenizer's official conversational template.
    train_dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": prompt}]
                for prompt in train_df["prompt"]
            ],
            "completion": [
                [{"role": "assistant", "content": response}]
                for response in train_df["response"]
            ],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_rank in (-1, 0):
        print(f"Model weights: {model_id}")
        print(f"Base tokenizer: {model_id}")
        print(f"Chat template: {template_tokenizer_id}")
        print(f"Training EOS: {tokenizer.eos_token!r} (ID {tokenizer.eos_token_id})")
        print(f"Generation stop IDs: {generation_config.eos_token_id}")
        print(f"Training examples: {len(train_dataset):,}")
        print(f"Overlong prompts dropped: {num_dropped:,}")
        print(f"Output directory: {output_dir}")

    # Build SFTConfig before loading the model so Transformers can initialize
    # the ZeRO-3 integration before from_pretrained allocates model weights.
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=str(args.deepspeed),
        bf16=True,
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
        local_rank=args.local_rank,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config = generation_config

    model_vocab_size = model.get_input_embeddings().num_embeddings
    max_token_id = max(tokenizer.get_vocab().values())
    if max_token_id >= model_vocab_size:
        raise ValueError(
            f"Tokenizer ID {max_token_id:,} exceeds the model vocabulary size "
            f"of {model_vocab_size:,}"
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.config.use_cache = True
    trainer.save_model(str(output_dir))

    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()

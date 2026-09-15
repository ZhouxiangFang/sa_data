"""Instruction-tune a pretrained model on a prompt/response CSV."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from helper import model_mapping


CSV_REQUIRED_COLUMNS = ("prompt", "response")

# Train the base model weights, but format data with the corresponding official
# instruction-tuned model's tokenizer and chat template.
instruct_tokenizer_mapping = {
    "llama3": "llama3-ins",
    "llama3.1": "llama3.1-ins",
    "qwen2.5": "qwen2.5-ins",
    "qwen3": "qwen3-ins",
    "olmo2": "olmo2-ins",
    "olmo3": "olmo3-ins",
}


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


def main():
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
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    if not args.deepspeed.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed}")

    model_id = model_mapping[args.model]
    tokenizer_id = model_mapping[instruct_tokenizer_mapping[args.model]]
    train_df = load_csv_dataset(args.dataset)

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

    original_name = model_id.rsplit("/", 1)[-1]
    ckpt_name = f"{original_name}-git"
    output_dir = args.output_dir / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_rank in (-1, 0):
        print(f"Model weights: {model_id}")
        print(f"Chat tokenizer: {tokenizer_id}")
        print(f"Training examples: {len(train_dataset):,}")
        print(f"Output directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if not tokenizer.chat_template:
        raise ValueError(f"The tokenizer {tokenizer_id} has no chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model_vocab_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != model_vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: {len(tokenizer):,} != "
            f"{model_vocab_size:,}"
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()

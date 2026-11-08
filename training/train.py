"""CPU fine-tuning of Pythia-70M on cleaned teacher-generated command/JSON pairs.

Examples follow the assignment format:
    Command: <command>
    JSON: <teacher JSON>

Usage:
    make train
    make train SUBSET=50
    python training/train.py --config training/config.yaml --out models/v2 --extra-epoch
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

REQUIRED_FIELDS = {"id", "text", "intent", "date", "time", "place"}


class CommandJsonDataset(Dataset):
    """Tokenize complete command/teacher-JSON sequences for causal-LM training.

    Padding positions are masked using attention_mask. This deliberately avoids
    masking the real end-of-sequence token when pad_token_id == eos_token_id,
    which is common for GPT-style tokenizers such as Pythia.
    """

    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples: list[dict[str, torch.Tensor]] = []
        for record in records:
            teacher_json = json.dumps(
                {
                    "intent": record["intent"],
                    "date": record["date"],
                    "time": record["time"],
                    "place": record["place"],
                },
                ensure_ascii=False,
                separators=(",", ": "),
            )
            text = f"Command: {record['text']}\nJSON: {teacher_json}{tokenizer.eos_token}"
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = [
                token_id if mask else -100
                for token_id, mask in zip(input_ids, attention_mask)
            ]
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: str, limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("Training record limit must be positive")
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields {sorted(missing)}"
                )
            records.append(record)
            if len(records) == limit:
                break
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def write_loss_csv(log_history: list[dict[str, Any]], path: str) -> int:
    rows = [
        (entry.get("epoch"), entry.get("loss"))
        for entry in log_history
        if "loss" in entry and "epoch" in entry
    ]
    # A logging event per epoch is expected with logging_strategy='epoch'.
    if not rows:
        raise RuntimeError("Trainer did not record any per-epoch training losses")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--out", default="models/v1")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--extra-epoch", action="store_true")
    parser.add_argument("--loss-out", default="training/loss.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    set_seed(seed)

    configured_limit = int(config["max_train_records"])
    limit = args.subset if args.subset is not None else configured_limit
    if limit > configured_limit:
        raise ValueError(
            f"Requested {limit} records, but config allows at most {configured_limit}"
        )

    records = load_records(config["train_file"], limit)
    print(f"Training on {len(records)} records from {config['train_file']}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.eos_token is None:
        raise ValueError("Base tokenizer does not define an EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["base_model"])
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = CommandJsonDataset(
        records, tokenizer, int(config["max_seq_length"])
    )
    epochs = int(config["num_epochs"]) + (1 if args.extra_epoch else 0)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=int(config["batch_size"]),
        num_train_epochs=epochs,
        learning_rate=float(config["learning_rate"]),
        seed=seed,
        data_seed=seed,
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    loss_rows = write_loss_csv(trainer.state.log_history, args.loss_out)

    metadata = {
        "base_model": config["base_model"],
        "seed": seed,
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "epochs": epochs,
        "max_seq_length": int(config["max_seq_length"]),
        "training_records": len(records),
        "train_file": config["train_file"],
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Saved model and tokenizer to {output_dir}", flush=True)
    print(f"Saved {loss_rows} epoch-loss rows to {args.loss_out}", flush=True)


if __name__ == "__main__":
    main()

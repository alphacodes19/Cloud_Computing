"""Task 2: fine-tune EleutherAI/pythia-70m on teacher-generated replies.

Each training example is built as:
    Command: <the command>
    JSON: <the teacher's JSON reply>
and the model is trained, causal-LM style, to produce the whole sequence
(sequence-level distillation: the teacher's raw output is the target, no
teacher logits are used).

Usage:
    python training/train.py --config training/config.yaml --out models/v1
    python training/train.py --config training/config.yaml --out models/v1 --subset 50

Writes training/loss.csv with one row per epoch (epoch, train_loss).
"""
import argparse
import csv
import json
import os
import random

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class CommandJsonDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        for r in records:
            teacher_json = json.dumps(
                {"intent": r["intent"], "date": r["date"], "time": r["time"], "place": r["place"]}
            )
            text = f"Command: {r['text']}\nJSON: {teacher_json}{tokenizer.eos_token}"
            enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            enc["labels"] = list(enc["input_ids"])
            self.examples.append(enc)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_records(path, limit):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--out", default="models/v1")
    ap.add_argument("--subset", type=int, default=None,
                     help="train on only the first N records (for a quick smoke test)")
    ap.add_argument("--extra-epoch", action="store_true",
                     help="train one extra epoch, for producing models/v2 (Task 2 step 5)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    limit = args.subset if args.subset is not None else cfg["max_train_records"]
    records = load_records(cfg["train_file"], limit)
    print(f"training on {len(records)} records")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])

    dataset = CommandJsonDataset(records, tokenizer, cfg["max_seq_length"])

    num_epochs = cfg["num_epochs"] + (1 if args.extra_epoch else 0)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=num_epochs,
        learning_rate=cfg["learning_rate"],
        seed=cfg["seed"],
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    result = trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # Pull per-epoch training loss out of the trainer's log history.
    loss_rows = [(entry["epoch"], entry["loss"])
                 for entry in trainer.state.log_history if "loss" in entry]
    loss_path = "training/loss.csv"
    with open(loss_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss"])
        for epoch, loss in loss_rows:
            w.writerow([epoch, loss])

    print(f"saved model + tokenizer to {args.out}")
    print(f"wrote {loss_path}")


if __name__ == "__main__":
    main()

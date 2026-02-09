from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


LABELS = ["nest", "grid", "ring", "stack", "shell"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


@dataclass
class Sample:
    text: str
    label: int


class JsonlTextDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        enc = self.tokenizer(
            s.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(s.label)
        return item


def _load_samples(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            structure = obj.get("structure", "")
            if structure not in LABEL_TO_ID:
                continue
            samples.append(Sample(text=text, label=LABEL_TO_ID[structure]))
    return samples


def _train_eval_split(samples: List[Sample], eval_split: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if eval_split <= 0:
        return samples, []
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=rng).tolist()
    split = max(1, int(len(samples) * (1.0 - eval_split)))
    train_idx = indices[:split]
    eval_idx = indices[split:]
    train_samples = [samples[i] for i in train_idx]
    eval_samples = [samples[i] for i in eval_idx]
    return train_samples, eval_samples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BERT structure classifier")
    parser.add_argument("--data", required=True, help="JSONL dataset path")
    parser.add_argument("--outdir", required=True, help="Output model directory")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke run (few steps)")
    parser.add_argument("--eval_split", type=float, default=0.2, help="Fraction of data for eval")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = _load_samples(args.data)
    if len(samples) == 0:
        raise ValueError("Dataset is empty or contains no valid labels")
    train_samples, eval_samples = _train_eval_split(samples, args.eval_split, args.seed)
    train_dataset = JsonlTextDataset(train_samples, tokenizer, max_length=args.max_length)
    eval_dataset = JsonlTextDataset(eval_samples, tokenizer, max_length=args.max_length) if eval_samples else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or install CUDA-enabled torch.")
    use_cpu = args.device == "cpu"

    max_steps = 10 if args.smoke else -1
    eval_strategy = "epoch" if eval_dataset is not None else "no"
    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        seed=args.seed,
        logging_steps=20,
        report_to=[],
        use_cpu=use_cpu,
        max_steps=max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)


if __name__ == "__main__":
    main()

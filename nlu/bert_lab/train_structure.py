from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


LABELS = ["nest", "grid", "ring", "stack", "shell", "unknown"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


@dataclass
class Sample:
    text: str
    label: int


def _resolve_data_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    candidates = [
        "nlu/bert_lab/data/controlled_structure.jsonl",
        "nlu/bert_lab/data/structure_mix_v2.jsonl",
        "nlu/bert_lab/data/bert_lab_samples_norm.jsonl",
        "nlu/bert_lab/data/bert_lab_samples.jsonl",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No default structure dataset found. Pass --data explicitly.")


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
    with open(path, "r", encoding="utf-8") as f:
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
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        by_label.setdefault(s.label, []).append(i)
    train_idx: List[int] = []
    eval_idx: List[int] = []
    for _, idxs in by_label.items():
        rng.shuffle(idxs)
        n_total = len(idxs)
        n_eval = max(1, int(round(n_total * eval_split))) if n_total > 1 else 0
        n_train = n_total - n_eval
        train_idx.extend(idxs[:n_train])
        eval_idx.extend(idxs[n_train:])
    rng.shuffle(train_idx)
    rng.shuffle(eval_idx)
    train_samples = [samples[i] for i in train_idx]
    eval_samples = [samples[i] for i in eval_idx]
    return train_samples, eval_samples


def _macro_f1(preds, labels, n_labels: int) -> float:
    f1_list: List[float] = []
    for label in range(n_labels):
        tp = int(((preds == label) & (labels == label)).sum())
        fp = int(((preds == label) & (labels != label)).sum())
        fn = int(((preds != label) & (labels == label)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_list.append(f1)
    return float(sum(f1_list) / len(f1_list)) if f1_list else 0.0


def _class_weights(train_samples: List[Sample], n_labels: int) -> torch.Tensor:
    counts = torch.zeros(n_labels, dtype=torch.float)
    for s in train_samples:
        counts[s.label] += 1.0
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    # Normalize around 1.0 to keep loss scale stable.
    weights = inv * (n_labels / inv.sum())
    return weights


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = float((preds == labels).mean().item())
    macro_f1 = _macro_f1(preds, labels, len(LABELS))
    return {"accuracy": acc, "macro_f1": macro_f1}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._class_weights is not None:
            weight = self._class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BERT structure classifier")
    parser.add_argument("--data", default=None, help="JSONL dataset path")
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
    parser.add_argument("--no_class_weighting", action="store_true", help="Disable class-weighted loss")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data)
    print(f"[train_structure] using dataset: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = _load_samples(data_path)
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
    load_best = eval_dataset is not None and not args.smoke
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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    class_weights = None if args.no_class_weighting else _class_weights(train_samples, len(LABELS))
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        with open(f"{args.outdir}/eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()



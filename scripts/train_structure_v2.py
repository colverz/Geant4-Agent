from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LABELS = ["nest", "grid", "ring", "stack", "shell", "unknown"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


@dataclass
class Sample:
    text: str
    label: int
    source: str


class JsonlTextDataset(Dataset):
    def __init__(self, samples: list[Sample], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(sample.label)
        return item


def _load_config(path: str | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if path:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg


def _cfg(cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def _resolve_data_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    candidates = [
        "nlu/bert_lab/data/controlled_structure_v2.jsonl",
        "nlu/bert_lab/data/controlled_structure.jsonl",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("No structure dataset found. Pass --data explicitly.")


def _load_samples(path: str) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            structure = str(obj.get("structure", "")).strip()
            source = str(obj.get("source", "unknown"))
            if not text or structure not in LABEL_TO_ID:
                continue
            samples.append(Sample(text=text, label=LABEL_TO_ID[structure], source=source))
    return samples


def _split_samples(samples: list[Sample], eval_split: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if eval_split <= 0:
        return samples, []
    rng = random.Random(seed)
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    train: list[Sample] = []
    evals: list[Sample] = []
    for label, bucket in by_label.items():
        rng.shuffle(bucket)
        n_total = len(bucket)
        n_eval = max(1, int(round(n_total * eval_split))) if n_total > 1 else 0
        split = n_total - n_eval
        train.extend(bucket[:split])
        evals.extend(bucket[split:])
    rng.shuffle(train)
    rng.shuffle(evals)
    return train, evals


def _macro_f1(preds: Any, labels: Any, n_labels: int) -> float:
    f1_values: list[float] = []
    for label in range(n_labels):
        tp = int(((preds == label) & (labels == label)).sum())
        fp = int(((preds == label) & (labels != label)).sum())
        fn = int(((preds != label) & (labels == label)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_values.append(f1)
    return float(sum(f1_values) / len(f1_values)) if f1_values else 0.0


def _compute_class_weights(train_samples: list[Sample], unknown_boost: float) -> torch.Tensor:
    counts = torch.zeros(len(LABELS), dtype=torch.float)
    for sample in train_samples:
        counts[sample.label] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    unknown_idx = LABEL_TO_ID["unknown"]
    weights[unknown_idx] = weights[unknown_idx] * float(max(1.0, unknown_boost))
    weights = weights * (len(LABELS) / weights.sum())
    return weights


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = float((preds == labels).mean().item())
    macro_f1 = _macro_f1(preds, labels, len(LABELS))
    return {"accuracy": acc, "macro_f1": macro_f1}


class WeightedSmoothingTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None, label_smoothing: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._label_smoothing = float(max(0.0, label_smoothing))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._class_weights is not None:
            weights = self._class_weights.to(logits.device)
        else:
            weights = None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=self._label_smoothing)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train structure classifier v2 (normalized-text focused).")
    parser.add_argument("--data", default=None)
    parser.add_argument("--outdir", default="nlu/bert_lab/models/structure_controlled_v5_e2")
    parser.add_argument("--config", default="nlu/bert_lab/configs/structure_train_v2.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke training.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_path = _resolve_data_path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_name = str(_cfg(cfg, "model", "distilbert-base-uncased"))
    epochs = int(_cfg(cfg, "epochs", 2))
    batch_size = int(_cfg(cfg, "batch_size", 16))
    lr = float(_cfg(cfg, "lr", 3e-5))
    max_length = int(_cfg(cfg, "max_length", 256))
    seed = int(_cfg(cfg, "seed", 42))
    eval_split = float(_cfg(cfg, "eval_split", 0.2))
    weight_decay = float(_cfg(cfg, "weight_decay", 0.01))
    warmup_ratio = float(_cfg(cfg, "warmup_ratio", 0.1))
    save_total_limit = int(_cfg(cfg, "save_total_limit", 2))
    label_smoothing = float(_cfg(cfg, "label_smoothing", 0.05))
    unknown_boost = float(_cfg(cfg, "unknown_class_boost", 1.0))
    grad_accum = int(_cfg(cfg, "gradient_accumulation_steps", 1))
    early_stop_patience = int(_cfg(cfg, "early_stopping_patience", 2))
    metric_for_best = str(_cfg(cfg, "metric_for_best_model", "macro_f1"))
    greater_is_better = bool(_cfg(cfg, "greater_is_better", True))

    samples = _load_samples(data_path)
    if not samples:
        raise ValueError("Dataset is empty or contains no supported labels.")
    train_samples, eval_samples = _split_samples(samples, eval_split, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    use_cpu = args.device == "cpu"

    train_dataset = JsonlTextDataset(train_samples, tokenizer, max_length=max_length)
    eval_dataset = JsonlTextDataset(eval_samples, tokenizer, max_length=max_length) if eval_samples else None
    class_weights = _compute_class_weights(train_samples, unknown_boost=unknown_boost)

    max_steps = 20 if args.smoke else -1
    eval_strategy = "epoch" if eval_dataset is not None else "no"
    load_best = eval_dataset is not None and not args.smoke
    training_args = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, grad_accum),
        learning_rate=lr,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        seed=seed,
        logging_steps=20,
        report_to=[],
        use_cpu=use_cpu,
        max_steps=max_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
    )

    callbacks = []
    if load_best and early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stop_patience))

    trainer = WeightedSmoothingTrainer(
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))

    metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        metrics = trainer.evaluate()

    (outdir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "training_profile.json").write_text(
        json.dumps(
            {
                "data": data_path,
                "config_path": args.config,
                "resolved_config": {
                    "model": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_length": max_length,
                    "seed": seed,
                    "eval_split": eval_split,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup_ratio,
                    "save_total_limit": save_total_limit,
                    "label_smoothing": label_smoothing,
                    "unknown_class_boost": unknown_boost,
                    "gradient_accumulation_steps": grad_accum,
                    "early_stopping_patience": early_stop_patience,
                },
                "n_train": len(train_samples),
                "n_eval": len(eval_samples),
                "label_distribution_train": {
                    name: sum(1 for x in train_samples if x.label == idx)
                    for name, idx in LABEL_TO_ID.items()
                },
                "label_distribution_eval": {
                    name: sum(1 for x in eval_samples if x.label == idx)
                    for name, idx in LABEL_TO_ID.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] model saved to {outdir}")
    print(f"[ok] eval metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

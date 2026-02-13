from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["nest", "grid", "ring", "stack", "shell", "unknown"]
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
    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            structure = obj.get("structure", "")
            if structure not in LABEL_TO_ID:
                continue
            out.append(Sample(text=obj.get("text", ""), label=LABEL_TO_ID[structure]))
    return out


def _macro_f1(conf: List[List[int]]) -> float:
    f1_vals: List[float] = []
    n = len(conf)
    for i in range(n):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(n) if r != i)
        fn = sum(conf[i][c] for c in range(n) if c != i)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        f1_vals.append(f1)
    return sum(f1_vals) / len(f1_vals) if f1_vals else 0.0


def evaluate(model_dir: str, data_path: str, max_length: int, device: str) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    samples = _load_samples(data_path)
    if not samples:
        raise ValueError("No valid samples found")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()

    dataset = JsonlTextDataset(samples, tokenizer, max_length=max_length)
    conf = [[0 for _ in LABELS] for _ in LABELS]
    correct = 0

    for item in dataset:
        labels = int(item["labels"].item())
        x = {k: v.unsqueeze(0).to(dev) for k, v in item.items() if k != "labels"}
        with torch.no_grad():
            logits = model(**x).logits.squeeze(0)
        pred = int(torch.argmax(logits).item())
        conf[labels][pred] += 1
        correct += int(pred == labels)

    total = len(samples)
    metrics = {
        "n_samples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": _macro_f1(conf),
        "labels": LABELS,
        "confusion_matrix": conf,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structure classifier on JSONL dataset")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    metrics = evaluate(args.model_dir, args.data, args.max_length, args.device)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

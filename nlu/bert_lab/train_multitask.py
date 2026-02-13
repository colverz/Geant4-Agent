from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments

from nlu.bert_lab.labels import STRUCTURE_LABELS, TOKEN_LABELS
from nlu.bert_lab.multitask import MultiTaskBert
from nlu.bert_lab.ollama_client import chat, extract_json


STRUCTURE_TO_ID = {name: i for i, name in enumerate(STRUCTURE_LABELS)}
TOKEN_TO_ID = {name: i for i, name in enumerate(TOKEN_LABELS)}
ID_TO_TOKEN = {i: name for i, name in enumerate(TOKEN_LABELS)}


@dataclass
class Sample:
    text: str
    structure: str
    spans: Optional[List[Dict[str, object]]] = None


def _resolve_data_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    candidates = [
        "nlu/bert_lab/data/controlled_multitask.jsonl",
        "nlu/bert_lab/data/bert_lab_multitask_samples.jsonl",
        "nlu/bert_lab/data/bert_lab_samples_norm.jsonl",
        "nlu/bert_lab/data/bert_lab_samples.jsonl",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No default multitask dataset found. Pass --data explicitly.")


def _labels_from_spans(offsets: List[List[int]], spans: List[Dict[str, object]]) -> List[int]:
    labels = [TOKEN_TO_ID["O"] for _ in offsets]
    for sp in spans:
        key = sp["key"]
        if f"B-{key}" not in TOKEN_TO_ID:
            continue
        start = int(sp["start"])
        end = int(sp["end"])
        b_label = TOKEN_TO_ID[f"B-{key}"]
        i_label = TOKEN_TO_ID[f"I-{key}"]
        first = True
        for i, (s, e) in enumerate(offsets):
            if e <= start or s >= end:
                continue
            if first:
                labels[i] = b_label
                first = False
            else:
                labels[i] = i_label
    return labels


class MultiTaskDataset(Dataset):
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
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").squeeze(0).tolist()
        attention = enc["attention_mask"].squeeze(0).tolist()

        token_labels = None
        if s.spans:
            token_labels = _labels_from_spans(offsets, s.spans)
            for i, (span, mask) in enumerate(zip(offsets, attention)):
                if mask == 0 or (span[0] == 0 and span[1] == 0):
                    token_labels[i] = -100

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["structure_labels"] = torch.tensor(STRUCTURE_TO_ID.get(s.structure, STRUCTURE_TO_ID["unknown"]))
        if token_labels is not None:
            item["token_labels"] = torch.tensor(token_labels)
        return item


def _load_samples(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            structure = obj.get("structure", "unknown")
            spans = obj.get("spans")
            samples.append(Sample(text=text, structure=structure, spans=spans))
    return samples


def _train_eval_split(samples: List[Sample], eval_split: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if eval_split <= 0:
        return samples, []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = max(1, int(round(len(samples) * eval_split)))
    eval_idx = set(idx[:n_eval])
    train = [samples[i] for i in idx if i not in eval_idx]
    evals = [samples[i] for i in idx if i in eval_idx]
    return train, evals


def _compute_metrics(eval_pred):
    if hasattr(eval_pred, "predictions"):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        preds, labels = eval_pred

    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        structure_logits, token_logits = preds
    else:
        structure_logits, token_logits = preds, None

    if isinstance(labels, (list, tuple)) and len(labels) == 2:
        structure_labels, token_labels = labels
    else:
        structure_labels, token_labels = labels, None

    pred_structure = np.argmax(structure_logits, axis=-1)
    struct_acc = float((pred_structure == structure_labels).mean())

    if token_labels is None or token_logits is None:
        return {"structure_acc": struct_acc}

    mask = token_labels != -100
    pred_tokens = np.argmax(token_logits, axis=-1)
    correct = int((pred_tokens[mask] == token_labels[mask]).sum())
    total = int(mask.sum())
    token_acc = (correct / total) if total else 0.0
    return {"structure_acc": struct_acc, "token_acc": token_acc}


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["structure_labels", "token_labels"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        structure_labels = inputs.get("structure_labels")
        token_labels = inputs.get("token_labels")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                structure_labels=structure_labels,
                token_labels=token_labels,
            )
            loss = None
            if structure_labels is not None:
                loss = outputs.structure_loss
                if outputs.token_loss is not None:
                    loss = loss + outputs.token_loss
            logits = (outputs.structure_logits, outputs.token_logits)

        if prediction_loss_only:
            return (loss, None, None)

        labels = None
        if structure_labels is not None:
            labels = (structure_labels, token_labels)
        return (loss, logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        structure_labels = inputs.pop("structure_labels")
        token_labels = inputs.pop("token_labels", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            structure_labels=structure_labels,
            token_labels=token_labels,
        )
        loss = outputs.structure_loss
        if outputs.token_loss is not None:
            loss = loss + outputs.token_loss
        return (loss, (outputs.structure_logits, outputs.token_logits)) if return_outputs else loss


def _generate_llm_samples(n: int, ollama_config: str) -> List[Sample]:
    prompt = (
        "Generate a single JSON object for a detector request.\n"
        "Return JSON with keys: text, structure.\n"
        "Structure must be one of: nest, grid, ring, stack, shell, single_box, single_tubs, unknown.\n"
        "The text should be a short natural-language request containing geometry hints.\n"
        "JSON:"
    )
    samples: List[Sample] = []
    for _ in range(n):
        resp = chat(prompt, config_path=ollama_config, temperature=0.4)
        obj = extract_json(resp.get("response", "")) or {}
        text = obj.get("text") or ""
        structure = obj.get("structure") or "unknown"
        if not text:
            continue
        samples.append(Sample(text=text, structure=structure, spans=None))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-task BERT (structure + token) model")
    parser.add_argument("--data", default=None, help="JSONL dataset path")
    parser.add_argument("--outdir", required=True, help="Output model directory")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--llm_aug_n", type=int, default=0, help="Extra LLM samples (structure-only)")
    parser.add_argument("--ollama_config", default="nlu/bert_lab/configs/ollama_config.json")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data)
    print(f"[train_multitask] using dataset: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = _load_samples(data_path)
    if args.llm_aug_n > 0:
        samples.extend(_generate_llm_samples(args.llm_aug_n, args.ollama_config))
    if len(samples) == 0:
        raise ValueError("Dataset is empty")

    train_samples, eval_samples = _train_eval_split(samples, args.eval_split, args.seed)
    train_dataset = MultiTaskDataset(train_samples, tokenizer, max_length=args.max_length)
    eval_dataset = MultiTaskDataset(eval_samples, tokenizer, max_length=args.max_length) if eval_samples else None

    model = MultiTaskBert(
        args.model,
        num_structure_labels=len(STRUCTURE_LABELS),
        num_token_labels=len(TOKEN_LABELS),
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or install CUDA-enabled torch.")
    use_cpu = args.device == "cpu"

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
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    with open(f"{args.outdir}/multitask_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.model,
                "structure_labels": STRUCTURE_LABELS,
                "token_labels": TOKEN_LABELS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()

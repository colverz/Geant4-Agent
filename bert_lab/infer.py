from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from .postprocess import merge_params


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _parse_value(text: str) -> float | None:
    text = text.strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", text)
    if not m:
        return None
    value = float(m.group(1))
    if "cm" in text:
        value *= 10.0
    return value


def _key_to_type(key: str) -> str:
    if key in {"nx", "ny", "n"}:
        return "int"
    return "float"


def predict_structure(
    text: str,
    model_dir: str,
    device: str = "auto",
    min_confidence: float = 0.6,
) -> Tuple[str, Dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        dev = torch.device("cuda")
    elif device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(0)
    probs = _softmax(logits).cpu().tolist()
    id2label = model.config.id2label
    best_id = int(torch.argmax(logits).item())
    best_label = id2label[best_id]
    best_prob = float(probs[best_id])
    if best_prob < min_confidence:
        best_label = "unknown"
    scores = {id2label[i]: float(p) for i, p in enumerate(probs)}
    scores["best_prob"] = best_prob
    return best_label, scores


def extract_params(text: str, model_dir: str, device: str = "auto") -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        dev = torch.device("cuda")
    elif device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    offsets = enc.pop("offset_mapping").squeeze(0).tolist()
    enc = {k: v.to(dev) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits.squeeze(0)
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    id2label = model.config.id2label

    params: Dict[str, float] = {}
    current_key = None
    current_start = None
    current_end = None

    for (start, end), pred_id in zip(offsets, pred_ids):
        if start == end:
            continue
        label = id2label[pred_id]
        if label == "O":
            if current_key is not None:
                span_text = text[current_start:current_end]
                value = _parse_value(span_text)
                if value is not None:
                    if _key_to_type(current_key) == "int":
                        params[current_key] = int(round(value))
                    else:
                        params[current_key] = float(value)
                current_key = None
                current_start = None
                current_end = None
            continue
        if label.startswith("B-"):
            if current_key is not None:
                span_text = text[current_start:current_end]
                value = _parse_value(span_text)
                if value is not None:
                    if _key_to_type(current_key) == "int":
                        params[current_key] = int(round(value))
                    else:
                        params[current_key] = float(value)
            current_key = label[2:]
            current_start = start
            current_end = end
        elif label.startswith("I-"):
            key = label[2:]
            if current_key == key:
                current_end = end
            else:
                current_key = key
                current_start = start
                current_end = end

    if current_key is not None:
        span_text = text[current_start:current_end]
        value = _parse_value(span_text)
        if value is not None:
            if _key_to_type(current_key) == "int":
                params[current_key] = int(round(value))
            else:
                params[current_key] = float(value)

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer structure and params from text")
    parser.add_argument("--text", required=True)
    parser.add_argument("--structure_model", default="bert_lab/bert_model")
    parser.add_argument("--ner_model", default="bert_lab/bert_ner_model")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    structure, scores = predict_structure(args.text, args.structure_model, args.device, args.min_confidence)
    params = extract_params(args.text, args.ner_model, args.device)
    params, notes = merge_params(args.text, params)

    print(json.dumps({"structure": structure, "scores": scores, "params": params, "notes": notes}, indent=2))


if __name__ == "__main__":
    main()

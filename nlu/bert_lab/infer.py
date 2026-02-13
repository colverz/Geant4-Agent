from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from .postprocess import merge_params


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"
_STRUCTURE_CACHE: Dict[tuple[str, str], tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]] = {}
_NER_CACHE: Dict[tuple[str, str], tuple[AutoTokenizer, AutoModelForTokenClassification, torch.device]] = {}


def _default_structure_model() -> str:
    for name in ["structure_controlled_v4c_e1", "structure_controlled_v3_e1", "structure_controlled_smoke", "structure_opt_v3", "structure_opt_v2", "structure"]:
        p = MODELS_DIR / name
        if (p / "config.json").exists():
            return str(p)
    return "nlu/bert_lab/models/structure_controlled_v4c_e1"


def _default_ner_model() -> str:
    p = MODELS_DIR / "ner"
    if (p / "config.json").exists():
        return str(p)
    return "nlu/bert_lab/models/ner"


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


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _device_tag(dev: torch.device) -> str:
    return f"{dev.type}:{dev.index if dev.index is not None else 0}"


def _load_structure_model(model_dir: str, device: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    dev = _resolve_device(device)
    key = (str(Path(model_dir).resolve()), _device_tag(dev))
    cached = _STRUCTURE_CACHE.get(key)
    if cached is not None:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    model.eval()
    out = (tokenizer, model, dev)
    _STRUCTURE_CACHE[key] = out
    return out


def _load_ner_model(model_dir: str, device: str) -> tuple[AutoTokenizer, AutoModelForTokenClassification, torch.device]:
    dev = _resolve_device(device)
    key = (str(Path(model_dir).resolve()), _device_tag(dev))
    cached = _NER_CACHE.get(key)
    if cached is not None:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(dev)
    model.eval()
    out = (tokenizer, model, dev)
    _NER_CACHE[key] = out
    return out


def _rule_calibration(text: str, labels: List[str]) -> Dict[str, float]:
    text_l = text.lower()
    bonus = {label: 0.0 for label in labels}

    if any(k in text_l for k in ["ring", "circular", "circle", "radius", "annulus"]):
        bonus["ring"] = bonus.get("ring", 0.0) + 0.25
    if any(k in text_l for k in ["grid", "array", "matrix", "nx", "ny", "pitch"]):
        bonus["grid"] = bonus.get("grid", 0.0) + 0.25
    if any(k in text_l for k in ["stack", "layer", "layers", "along z"]):
        bonus["stack"] = bonus.get("stack", 0.0) + 0.2
    if any(k in text_l for k in ["nest", "inside", "contain", "contains", "inner", "outer"]):
        bonus["nest"] = bonus.get("nest", 0.0) + 0.2
    if any(k in text_l for k in ["shell", "concentric", "thickness", "coaxial"]):
        bonus["shell"] = bonus.get("shell", 0.0) + 0.2
    return bonus


def _unknown_intent_signal(text: str) -> float:
    t = text.lower()
    strong = [
        "geometry_intent: unresolved",
        "candidate_pattern",
        "undecided",
        "ambiguous",
        "not fixed yet",
        "exact arrangement is not fixed",
        "final arrangement remains ambiguous",
    ]
    weak = [
        "may be",
        "either",
        "depending on constraints",
        "not sure",
        "tentative",
        "provide more constraints",
    ]
    score = 0.0
    for s in strong:
        if s in t:
            score += 1.0
    for s in weak:
        if s in t:
            score += 0.4
    return score


def predict_structure(
    text: str,
    model_dir: str,
    device: str = "auto",
    min_confidence: float = 0.6,
    use_rule_calibration: bool = True,
    low_confidence_accept: float = 0.45,
    ambiguity_margin: float = 0.03,
) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
    tokenizer, model, dev = _load_structure_model(model_dir, device)

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(0)

    id2label = model.config.id2label
    if use_rule_calibration:
        labels = [id2label[i] for i in range(len(id2label))]
        bonus = _rule_calibration(text, labels)
        for i, label in enumerate(labels):
            logits[i] = logits[i] + bonus.get(label, 0.0)

    probs = _softmax(logits).cpu().tolist()
    best_id = int(torch.argmax(logits).item())
    best_label = id2label[best_id]
    best_prob = float(probs[best_id])

    sorted_probs = sorted(probs, reverse=True)
    second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    gap = best_prob - second_prob
    unknown_signal = _unknown_intent_signal(text)
    dynamic_min = min_confidence
    dynamic_margin = ambiguity_margin
    dynamic_low_accept = low_confidence_accept
    if unknown_signal >= 1.0:
        dynamic_min = max(dynamic_min, 0.72)
        dynamic_margin = max(dynamic_margin, 0.08)
        dynamic_low_accept = max(dynamic_low_accept, 0.55)

    accept_low_conf = (best_prob >= dynamic_low_accept) and (gap >= dynamic_margin)
    # Keep unknown-class probability for rejection checks below.
    unknown_id = None
    for i, lb in id2label.items():
        if lb == "unknown":
            unknown_id = int(i)
            break
    unknown_prob = float(probs[unknown_id]) if unknown_id is not None else 0.0

    reject_by_threshold = (best_prob < dynamic_min and not accept_low_conf)
    reject_by_ambiguity = (unknown_signal >= 1.0 and gap < dynamic_margin)
    reject_by_unknown_prob = (unknown_prob >= 0.22 and gap < 0.18)

    if reject_by_threshold or reject_by_ambiguity or reject_by_unknown_prob:
        best_label = "unknown"
    scores = {id2label[i]: float(p) for i, p in enumerate(probs)}
    scores["best_prob"] = best_prob
    scores["second_prob"] = second_prob
    scores["margin"] = gap
    scores["unknown_signal"] = float(unknown_signal)
    scores["dynamic_min_conf"] = float(dynamic_min)
    scores["dynamic_margin"] = float(dynamic_margin)
    scores["unknown_prob"] = float(unknown_prob)
    ranked = sorted(((id2label[i], float(p)) for i, p in enumerate(probs)), key=lambda x: x[1], reverse=True)
    return best_label, scores, ranked


def extract_params(text: str, model_dir: str, device: str = "auto") -> Dict[str, float]:
    tokenizer, model, dev = _load_ner_model(model_dir, device)

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
    parser.add_argument("--structure_model", default=_default_structure_model())
    parser.add_argument("--ner_model", default=_default_ner_model())
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()

    structure, scores, ranked = predict_structure(
        args.text, args.structure_model, args.device, args.min_confidence
    )
    params = extract_params(args.text, args.ner_model, args.device)
    params, notes = merge_params(args.text, params)

    top_k = max(1, args.top_k)
    candidates = [{"structure": name, "prob": prob} for name, prob in ranked[:top_k]]

    print(
        json.dumps(
            {"structure": structure, "scores": scores, "candidates": candidates, "params": params, "notes": notes},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()



from __future__ import annotations

import json
import os
import re
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from nlu.training.bert_lab.labels import ENTITY_KEYS, STRUCTURE_LABELS, TOKEN_LABELS
from nlu.training.bert_lab.multitask import MultiTaskBert


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"

STRUCTURE_TO_ID = {name: i for i, name in enumerate(STRUCTURE_LABELS)}
ID_TO_STRUCTURE = {i: name for i, name in enumerate(STRUCTURE_LABELS)}
ID_TO_TOKEN = {i: name for i, name in enumerate(TOKEN_LABELS)}
hf_logging.set_verbosity_error()
_CACHE: Dict[str, Any] = {}


def _parse_value(text: str) -> float | None:
    text = text.strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", text)
    if not m:
        return None
    value = float(m.group(1))
    if "cm" in text:
        value *= 10.0
    if "m" in text and "mm" not in text and "cm" not in text:
        value *= 1000.0
    return value


def _key_to_type(key: str) -> str:
    if key in {"nx", "ny", "n"}:
        return "int"
    return "float"


def _resolve_base_model(model_dir: Path) -> str:
    cfg = model_dir / "multitask_config.json"
    if cfg.exists():
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
            model_name = str(obj.get("base_model", "")).strip()
            if model_name:
                return model_name
        except Exception:
            pass
    local_backbones = [
        MODELS_DIR / "structure_opt_v3",
        MODELS_DIR / "structure",
    ]
    for p in local_backbones:
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            return str(p)
    return "distilbert-base-uncased"


def _pick_multitask_model() -> str | None:
    p = MODELS_DIR / "multitask"
    if (p / "model.safetensors").exists() and (p / "tokenizer.json").exists():
        return str(p)
    return None


def _decode_token_spans(text: str, offsets: List[List[int]], pred_ids: List[int]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    current_key = None
    current_start = None
    current_end = None

    def flush() -> None:
        nonlocal current_key, current_start, current_end
        if current_key is not None and current_start is not None and current_end is not None:
            span_text = text[current_start:current_end].strip()
            if span_text:
                values[current_key] = span_text
        current_key = None
        current_start = None
        current_end = None

    for (start, end), pred_id in zip(offsets, pred_ids):
        if start == end:
            continue
        label = ID_TO_TOKEN.get(pred_id, "O")
        if label == "O":
            flush()
            continue
        if label.startswith("B-"):
            flush()
            current_key = label[2:]
            current_start = start
            current_end = end
            continue
        if label.startswith("I-"):
            key = label[2:]
            if current_key == key and current_end is not None:
                current_end = end
            else:
                flush()
                current_key = key
                current_start = start
                current_end = end

    flush()
    return values


def predict_multitask(
    text: str,
    model_dir: str | None = None,
    device: str = "auto",
    min_confidence: float = 0.6,
) -> Dict[str, Any]:
    model_dir = model_dir or _pick_multitask_model()
    if not model_dir:
        raise FileNotFoundError("multitask model not found")

    model_path = Path(model_dir)
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model_path}|{dev.type}"
    cache = _CACHE.get(cache_key)
    if cache is None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            base_model = _resolve_base_model(model_path)
            model = MultiTaskBert(
                model_name=base_model,
                num_structure_labels=len(STRUCTURE_LABELS),
                num_token_labels=len(TOKEN_LABELS),
            )
            state_dict = load_file(str(model_path / "model.safetensors"))
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model = model.to(dev)
        cache = {"tokenizer": tokenizer, "model": model, "base_model": base_model}
        _CACHE[cache_key] = cache
    tokenizer = cache["tokenizer"]
    model = cache["model"]
    base_model = cache["base_model"]

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
        outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    structure_logits = outputs.structure_logits.squeeze(0)
    token_logits = outputs.token_logits.squeeze(0)

    probs = torch.softmax(structure_logits, dim=-1).cpu().tolist()
    ranked = sorted(((ID_TO_STRUCTURE[i], float(p)) for i, p in enumerate(probs)), key=lambda x: x[1], reverse=True)
    best_label, best_prob = ranked[0]
    if best_prob < min_confidence:
        best_label = "unknown"
    scores = {ID_TO_STRUCTURE[i]: float(p) for i, p in enumerate(probs)}
    scores["best_prob"] = float(best_prob)

    pred_ids = torch.argmax(token_logits, dim=-1).cpu().tolist()
    span_map = _decode_token_spans(text, offsets, pred_ids)
    params: Dict[str, float] = {}
    entities: Dict[str, str] = {}
    for key, span_text in span_map.items():
        if key in ENTITY_KEYS:
            entities[key] = span_text
            continue
        val = _parse_value(span_text)
        if val is None:
            continue
        if _key_to_type(key) == "int":
            params[key] = int(round(val))
        else:
            params[key] = float(val)

    return {
        "structure": best_label,
        "scores": scores,
        "ranked": ranked,
        "params": params,
        "entities": entities,
        "backend": "multitask",
        "model_dir": str(model_path),
        "base_model": base_model,
    }

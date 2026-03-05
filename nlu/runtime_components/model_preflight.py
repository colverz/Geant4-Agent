from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"
STRUCTURE_MODEL_CANDIDATES = [
    "structure_controlled_v4c_e1",
    "structure_controlled_v3_e1",
    "structure_controlled_smoke",
    "structure_opt_v3",
    "structure_opt_v2",
    "structure",
]

_REQUIRED_FILES = ["config.json"]
_TOKENIZER_ALT_GROUPS = [
    ["tokenizer.json"],
    ["spiece.model"],
    ["sentencepiece.bpe.model"],
    ["vocab.txt"],
    ["vocab.json", "merges.txt"],
]


def _all_exist(base: Path, names: list[str]) -> bool:
    return all((base / name).exists() for name in names)


def validate_model_dir(path: Path, *, label: str) -> dict[str, Any]:
    missing: list[str] = []
    warnings: list[str] = []
    if not path.exists() or not path.is_dir():
        missing.append("directory")
        return {
            "ok": False,
            "label": label,
            "path": str(path),
            "missing_files": missing,
            "warnings": warnings,
        }

    for name in _REQUIRED_FILES:
        if not (path / name).exists():
            missing.append(name)

    has_tokenizer = any(_all_exist(path, group) for group in _TOKENIZER_ALT_GROUPS)
    if not has_tokenizer:
        missing.append("tokenizer_assets")
    if not (path / "tokenizer_config.json").exists():
        warnings.append("tokenizer_config.json missing (recommended)")

    return {
        "ok": len(missing) == 0,
        "label": label,
        "path": str(path),
        "missing_files": missing,
        "warnings": warnings,
    }


def select_default_structure_dir(models_dir: Path = MODELS_DIR) -> Path:
    for name in STRUCTURE_MODEL_CANDIDATES:
        candidate = models_dir / name
        report = validate_model_dir(candidate, label="structure")
        if report["ok"]:
            return candidate
    return models_dir / STRUCTURE_MODEL_CANDIDATES[0]


def select_default_ner_dir(models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / "ner"


def runtime_model_readiness(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    structure_path = select_default_structure_dir(models_dir)
    ner_path = select_default_ner_dir(models_dir)
    structure_report = validate_model_dir(structure_path, label="structure")
    ner_report = validate_model_dir(ner_path, label="ner")
    return {
        "models_dir": str(models_dir),
        "structure": structure_report,
        "ner": ner_report,
        "ready": bool(structure_report["ok"] and ner_report["ok"]),
    }


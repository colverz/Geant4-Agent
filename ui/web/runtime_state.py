from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nlu.llm_support.ollama_client import load_config
from nlu.runtime_components.model_preflight import runtime_model_readiness


ROOT = Path(__file__).parent
OLLAMA_CONFIG_DIR = ROOT.parent.parent / "nlu" / "llm_support" / "configs"
CURRENT_OLLAMA_CONFIG = os.getenv("OLLAMA_CONFIG_PATH", "nlu/llm_support/configs/ollama_config.json")
_CURRENT_PATH_OBJ = Path(CURRENT_OLLAMA_CONFIG)
if _CURRENT_PATH_OBJ.exists():
    CURRENT_OLLAMA_CONFIG = str(_CURRENT_PATH_OBJ.resolve()).replace("\\", "/")


def get_ollama_config_path() -> str:
    return CURRENT_OLLAMA_CONFIG


def set_ollama_config_path(path: str) -> tuple[bool, str]:
    global CURRENT_OLLAMA_CONFIG
    if not path:
        return False, "empty config path"
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = (ROOT.parent.parent / p).resolve()
    if not p.exists() or not p.is_file():
        return False, f"config not found: {path}"
    p = p.resolve()
    try:
        load_config(p)
    except Exception as ex:
        return False, f"invalid config: {ex}"
    CURRENT_OLLAMA_CONFIG = str(p).replace("\\", "/")
    return True, CURRENT_OLLAMA_CONFIG


def runtime_config_payload() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if OLLAMA_CONFIG_DIR.exists():
        for p in sorted(OLLAMA_CONFIG_DIR.glob("*.json")):
            if p.name.endswith(".example.json"):
                continue
            try:
                cfg = load_config(p)
                items.append(
                    {
                        "path": str(p).replace("\\", "/"),
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "base_url": cfg.base_url,
                        "timeout_s": cfg.timeout_s,
                    }
                )
            except Exception:
                continue
    current_raw = get_ollama_config_path()
    current_path = current_raw
    p = Path(current_raw)
    if p.exists():
        current_path = str(p.resolve()).replace("\\", "/")
    current_provider = ""
    current_model = ""
    current_base = ""
    try:
        cur = load_config(p if p.exists() else current_raw)
        current_provider = cur.provider
        current_model = cur.model
        current_base = cur.base_url
    except Exception:
        current_provider = ""
        pass
    model_preflight = runtime_model_readiness()
    return {
        "current_path": current_path,
        "current_provider": current_provider,
        "current_model": current_model,
        "current_base_url": current_base,
        "available": items,
        "model_preflight": model_preflight,
    }


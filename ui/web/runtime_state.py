from __future__ import annotations

import os
import threading
from copy import deepcopy
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
_RECOMMENDED_CONFIG_LOCK = threading.RLock()
_RECOMMENDED_CONFIG_BY_SESSION: dict[str, dict[str, Any]] = {}
_SIMULATION_DESIGN_BY_SESSION: dict[str, dict[str, Any]] = {}
_CANDIDATE_STATUS_BY_SESSION: dict[str, dict[str, Any]] = {}
_AGENT_PLAN_BY_SESSION: dict[str, dict[str, Any]] = {}
_AGENT_STATE_BY_SESSION: dict[str, dict[str, Any]] = {}
_AGENT_STATE_SUMMARY_BY_SESSION: dict[str, dict[str, Any]] = {}


def get_ollama_config_path() -> str:
    return CURRENT_OLLAMA_CONFIG


def set_latest_recommended_config(session_id: str | None, config: dict[str, Any] | None) -> None:
    key = str(session_id or "").strip()
    if not key or not isinstance(config, dict) or not config:
        return
    with _RECOMMENDED_CONFIG_LOCK:
        _RECOMMENDED_CONFIG_BY_SESSION[key] = deepcopy(config)


def set_latest_simulation_design(
    session_id: str | None,
    *,
    user_text: str,
    candidate: dict[str, Any] | None,
    recommended_config: dict[str, Any] | None,
    design_advice: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    key = str(session_id or "").strip()
    if not key:
        return
    record = {
        "user_text": str(user_text or ""),
        "candidate": deepcopy(candidate or {}),
        "recommended_config": deepcopy(recommended_config or {}),
        "design_advice": deepcopy(design_advice or {}),
        "source": str(source or ""),
        "status": "proposed",
    }
    with _RECOMMENDED_CONFIG_LOCK:
        _SIMULATION_DESIGN_BY_SESSION[key] = record
        _CANDIDATE_STATUS_BY_SESSION[key] = {
            "status": "proposed",
            "source": str(source or ""),
            "has_recommended_config": bool(isinstance(recommended_config, dict) and recommended_config),
        }
        if isinstance(recommended_config, dict) and recommended_config:
            _RECOMMENDED_CONFIG_BY_SESSION[key] = deepcopy(recommended_config)


def get_latest_simulation_design(session_id: str | None) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        return deepcopy(_SIMULATION_DESIGN_BY_SESSION.get(key) or {})


def get_latest_recommended_config(session_id: str | None) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        return deepcopy(_RECOMMENDED_CONFIG_BY_SESSION.get(key) or {})


def get_candidate_status(session_id: str | None) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        record = deepcopy(_CANDIDATE_STATUS_BY_SESSION.get(key) or {})
        design = _SIMULATION_DESIGN_BY_SESSION.get(key) or {}
        if design and "candidate" not in record:
            record["candidate"] = deepcopy(design.get("candidate") or {})
        return record


def set_latest_agent_plan(session_id: str | None, plan: dict[str, Any] | None) -> None:
    key = str(session_id or "").strip()
    if not key or not isinstance(plan, dict) or not plan:
        return
    with _RECOMMENDED_CONFIG_LOCK:
        _AGENT_PLAN_BY_SESSION[key] = deepcopy(plan)


def get_latest_agent_plan(session_id: str | None) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        return deepcopy(_AGENT_PLAN_BY_SESSION.get(key) or {})


def set_latest_agent_state(session_id: str | None, state: dict[str, Any] | None) -> None:
    key = str(session_id or "").strip()
    if not key or not isinstance(state, dict) or not state:
        return
    with _RECOMMENDED_CONFIG_LOCK:
        _AGENT_STATE_BY_SESSION[key] = deepcopy(state)


def get_latest_agent_state(session_id: str | None) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        return deepcopy(_AGENT_STATE_BY_SESSION.get(key) or {})


def set_latest_agent_state_summary(session_id: str | None, summary: dict[str, Any] | None) -> None:
    key = str(session_id or "").strip()
    if not key or not isinstance(summary, dict) or not summary:
        return
    with _RECOMMENDED_CONFIG_LOCK:
        _AGENT_STATE_SUMMARY_BY_SESSION[key] = deepcopy(summary)


def get_latest_agent_state_summary(session_id: str | None) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        return deepcopy(_AGENT_STATE_SUMMARY_BY_SESSION.get(key) or {})


def mark_latest_candidate_accepted(session_id: str | None, *, committed: bool = False) -> dict[str, Any]:
    key = str(session_id or "").strip()
    if not key:
        return {}
    with _RECOMMENDED_CONFIG_LOCK:
        design = _SIMULATION_DESIGN_BY_SESSION.get(key)
        if not isinstance(design, dict) or not design:
            return {}
        status = "committed" if committed else "accepted"
        design["status"] = status
        _SIMULATION_DESIGN_BY_SESSION[key] = deepcopy(design)
        record = {
            "status": status,
            "source": str(design.get("source") or ""),
            "has_recommended_config": bool(design.get("recommended_config")),
            "candidate": deepcopy(design.get("candidate") or {}),
        }
        _CANDIDATE_STATUS_BY_SESSION[key] = deepcopy(record)
        return deepcopy(record)


def clear_latest_recommended_config(session_id: str | None) -> None:
    key = str(session_id or "").strip()
    if not key:
        return
    with _RECOMMENDED_CONFIG_LOCK:
        _RECOMMENDED_CONFIG_BY_SESSION.pop(key, None)
        _SIMULATION_DESIGN_BY_SESSION.pop(key, None)
        _CANDIDATE_STATUS_BY_SESSION.pop(key, None)
        _AGENT_PLAN_BY_SESSION.pop(key, None)
        _AGENT_STATE_BY_SESSION.pop(key, None)
        _AGENT_STATE_SUMMARY_BY_SESSION.pop(key, None)


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
    llm_ready = False
    try:
        if current_path and Path(current_path).exists():
            load_config(current_path)
            llm_ready = True
    except Exception:
        pass
    return {
        "current_path": current_path,
        "current_provider": current_provider,
        "current_model": current_model,
        "current_base_url": current_base,
        "available": items,
        "model_preflight": model_preflight,
        "llm_ready": llm_ready,
    }

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.config.defaults import build_legacy_default_config


@dataclass
class SessionState:
    history: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    last_question: str = ""


SESSIONS: Dict[str, SessionState] = {}


def ensure_session(session_id: Optional[str]) -> Tuple[str, SessionState]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    SESSIONS[sid] = SessionState(config=build_legacy_default_config())
    return sid, SESSIONS[sid]


def build_context_summary(config: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    geo = config.get("geometry", {})
    mats = config.get("materials", {})
    src = config.get("source", {})
    phy = config.get("physics", {})
    out = config.get("output", {})

    last_user = ""
    for item in reversed(history):
        if item.get("role") == "user":
            last_user = str(item.get("content", "")).strip()
            if last_user:
                break

    parts = [
        f"structure: {geo.get('structure') or ''}",
        f"geometry_params: {json.dumps(geo.get('params', {}), ensure_ascii=False)}",
        f"materials: {', '.join(mats.get('selected_materials', []))}",
        f"particle: {src.get('particle') or ''}",
        f"source_type: {src.get('type') or ''}",
        f"source_energy_MeV: {src.get('energy') if src.get('energy') is not None else ''}",
        f"source_position: {json.dumps(src.get('position'), ensure_ascii=False) if src.get('position') else ''}",
        f"source_direction: {json.dumps(src.get('direction'), ensure_ascii=False) if src.get('direction') else ''}",
        f"physics_list: {phy.get('physics_list') or ''}",
        f"output_format: {out.get('format') or ''}",
        f"output_path: {out.get('path') or ''}",
        f"last_user_turn: {last_user}",
    ]
    return "; ".join(parts)


def extract_semantic_frame_legacy(*args, **kwargs):
    from legacy.runtime.bert_lab.semantic import extract_semantic_frame

    return extract_semantic_frame(*args, **kwargs)

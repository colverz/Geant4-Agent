from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import V3AgentState


V3_SESSION_SCHEMA_VERSION = "geant4_agent_v3_session.v1"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class V3SessionEnvelope:
    session_id: str
    state: V3AgentState
    updated_at: str = ""
    last_turn_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "updated_at": self.updated_at,
            "last_turn_id": self.last_turn_id,
            "state": self.state.to_dict(),
        }


@dataclass(slots=True)
class V3SessionStore:
    sessions_dir: Path

    def __post_init__(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def state_path(self, session_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        return self.sessions_dir / f"{safe}.json"

    def save(self, state: V3AgentState, *, last_turn_id: str = "") -> None:
        envelope = V3SessionEnvelope(
            session_id=state.session_id,
            state=state,
            updated_at=datetime.now(timezone.utc).isoformat(),
            last_turn_id=last_turn_id,
        )
        try:
            self.state_path(state.session_id).write_text(
                json.dumps(envelope.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save v3 agent session %s: %s", state.session_id, exc)

    def load(self, session_id: str) -> V3AgentState | None:
        path = self.state_path(session_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("session_json_not_object")
            state_data = raw.get("state") if raw.get("schema_version") == V3_SESSION_SCHEMA_VERSION else raw
            if not isinstance(state_data, dict):
                raise ValueError("session_state_not_object")
            return V3AgentState.from_dict(state_data)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Failed to load v3 agent session %s from %s: %s", session_id, path, exc)
            self.quarantine(path)
            return None

    def reset(self, session_id: str | None = None) -> None:
        if session_id:
            try:
                self.state_path(session_id).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to delete v3 agent session %s: %s", session_id, exc)
            return
        for path in self.sessions_dir.glob("*.json"):
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to delete v3 agent session file %s: %s", path, exc)

    def quarantine(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        for idx in range(1000):
            suffix = ".corrupt" if idx == 0 else f".corrupt{idx}"
            target = path.with_name(path.name + suffix)
            if target.exists():
                continue
            try:
                path.rename(target)
                return target
            except OSError as exc:
                logger.warning("Failed to quarantine v3 agent session file %s: %s", path, exc)
                return None
        return None


__all__ = ["V3_SESSION_SCHEMA_VERSION", "V3SessionEnvelope", "V3SessionStore"]

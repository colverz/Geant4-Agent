from __future__ import annotations

from typing import Any

from core.agent_v3.service import V3AgentTurnService
from ui.web.runtime_state import get_ollama_config_path


_V3_AGENT_TURN_SERVICE = V3AgentTurnService()


def handle_v3_agent_post(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if path == "/api/v3/agent/turn":
        turn_payload = dict(payload)
        if turn_payload.get("llm_design_enabled") and not turn_payload.get("llm_config_path"):
            turn_payload["llm_config_path"] = get_ollama_config_path()
        body = _V3_AGENT_TURN_SERVICE.run_turn(turn_payload)
        return (200 if body.get("ok") else 400), body
    if path == "/api/v3/agent/reset":
        session_id = str(payload.get("session_id") or "").strip()
        _V3_AGENT_TURN_SERVICE.reset(session_id or None)
        return 200, {"ok": True, "session_id": session_id}
    if path == "/api/v3/agent/state":
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return 400, {"ok": False, "error": "missing_session_id", "session_id": session_id}
        return _V3_AGENT_TURN_SERVICE.get_state_payload(
            session_id,
            lang=str(payload.get("lang") or "zh").strip() or "zh",
        )
    return 404, {"ok": False, "error": "unknown_v3_agent_path"}

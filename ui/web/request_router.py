from __future__ import annotations

from typing import Any, Callable

from ui.web.geant4_api import handle_geant4_post
from ui.web.runtime_state import runtime_config_payload, set_ollama_config_path
from ui.web.strict_api import handle_strict_audit, handle_strict_reset
from ui.web.async_jobs import create_step_job, get_job


POST_PATHS = {
    "/api/solve",
    "/api/step",
    "/api/step_async",
    "/api/step_status",
    "/api/reset",
    "/api/runtime",
    "/api/audit",
    "/api/geant4/apply",
    "/api/geant4/initialize",
    "/api/geant4/run",
    "/api/geant4/log",
    "/api/geant4/viewer/open",
}


def is_supported_post_path(path: str) -> bool:
    return path in POST_PATHS


def handle_post_request(
    path: str,
    payload: dict[str, Any],
    *,
    legacy_sessions: dict[str, Any] | None,
    solve_fn: Callable[[dict[str, Any]], dict[str, Any]],
    step_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    if path.startswith("/api/geant4/"):
        return handle_geant4_post(path, payload)

    if path == "/api/solve":
        return 200, solve_fn(payload)

    if path == "/api/reset":
        session_id = payload.get("session_id")
        if legacy_sessions is not None and session_id in legacy_sessions:
            del legacy_sessions[session_id]
        handle_strict_reset(str(session_id) if session_id else None)
        return 200, {"ok": True}

    if path == "/api/audit":
        session_id = str(payload.get("session_id", "")).strip()
        return 200, {"session_id": session_id, "audit": handle_strict_audit(session_id)}

    if path == "/api/runtime":
        cfg_path = str(payload.get("ollama_config_path", "")).strip()
        ok, message = set_ollama_config_path(cfg_path)
        body = {"ok": ok, "message": message, **runtime_config_payload()}
        return (200 if ok else 400), body

    if path == "/api/step_async":
        return 200, create_step_job(payload, step_fn)

    if path == "/api/step_status":
        job_id = str(payload.get("job_id", "")).strip()
        job = get_job(job_id)
        if job is None:
            return 404, {"error": "job_not_found", "job_id": job_id}
        return 200, job

    return 200, step_fn(payload)

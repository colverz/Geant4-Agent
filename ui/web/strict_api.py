from __future__ import annotations


def _load_session_manager():
    from core.orchestrator.session_manager import (
        get_session_config_summary as get_session_config_summary_v2,
        get_session_audit as get_session_audit_v2,
        process_turn as process_turn_v2,
        reset_session as reset_session_v2,
    )

    return get_session_audit_v2, process_turn_v2, reset_session_v2, get_session_config_summary_v2


def handle_strict_step(payload: dict, progress_cb=None) -> dict:
    from ui.web.runtime_state import get_ollama_config_path

    if progress_cb:
        progress_cb("loading_runtime", "Loading runtime", "Importing strict orchestration modules and model dependencies.")
    try:
        _, process_turn_v2, _, _ = _load_session_manager()
    except ModuleNotFoundError as ex:
        missing = ex.name or "unknown_dependency"
        result = {
            "session_id": payload.get("session_id"),
            "assistant_message": f"Strict runtime unavailable: missing dependency `{missing}`.",
            "phase": "runtime_unavailable",
            "phase_title": "Runtime Unavailable",
            "asked_fields": [],
            "asked_fields_friendly": [],
            "is_complete": False,
            "delta_paths": [],
            "display": {},
            "config": {},
            "config_min": {},
            "history": [],
            "llm_used": False,
            "fallback_reason": f"missing_dependency:{missing}",
            "temperatures": {},
            "rejected_updates": [],
            "violations": [],
            "applied_rules": [],
            "internal_trace": {"missing_dependency": missing},
        }
        if progress_cb:
            progress_cb("runtime_unavailable", "Runtime unavailable", f"missing dependency: {missing}")
        return result

    if progress_cb:
        progress_cb("runtime_ready", "Runtime ready", "Strict orchestration runtime is initialized.")
    return process_turn_v2(
        payload=payload,
        ollama_config_path=get_ollama_config_path(),
        min_confidence=float(payload.get("min_confidence", 0.6)),
        lang=str(payload.get("lang", "zh")).lower(),
        progress_cb=progress_cb,
    )


def handle_strict_reset(session_id: str | None) -> None:
    if not session_id:
        return
    try:
        _, _, reset_session_v2, _ = _load_session_manager()
    except ModuleNotFoundError:
        return
    reset_session_v2(str(session_id))


def handle_strict_audit(session_id: str) -> list[dict]:
    try:
        get_session_audit_v2, _, _, _ = _load_session_manager()
    except ModuleNotFoundError:
        return []
    return get_session_audit_v2(session_id)


def handle_strict_config_summary(session_id: str, *, lang: str = "zh") -> dict:
    try:
        _, _, _, get_session_config_summary_v2 = _load_session_manager()
    except ModuleNotFoundError as ex:
        return {
            "ok": False,
            "error": f"missing_dependency:{ex.name or 'unknown_dependency'}",
            "session_id": session_id,
            "action_safety_class": "read_only",
        }
    return get_session_config_summary_v2(session_id, lang=lang)

from __future__ import annotations


def _load_session_manager():
    from core.orchestrator.session_manager import (
        commit_recommended_config as commit_recommended_config_v2,
        get_session_config_summary as get_session_config_summary_v2,
        get_session_audit as get_session_audit_v2,
        process_turn as process_turn_v2,
        reset_session as reset_session_v2,
    )

    return (
        get_session_audit_v2,
        process_turn_v2,
        reset_session_v2,
        get_session_config_summary_v2,
        commit_recommended_config_v2,
    )


def _session_manager_parts():
    parts = tuple(_load_session_manager())
    if len(parts) == 4:
        get_audit, process_turn, reset_session, get_config_summary = parts
        return get_audit, process_turn, reset_session, get_config_summary, None
    if len(parts) >= 5:
        return parts[:5]
    raise ValueError("invalid_session_manager_loader")


def handle_strict_step(payload: dict, progress_cb=None) -> dict:
    from ui.web.runtime_state import get_ollama_config_path

    if progress_cb:
        progress_cb("loading_runtime", "Loading runtime", "Importing strict orchestration modules and model dependencies.")
    try:
        _, process_turn_v2, _, _, _ = _session_manager_parts()
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
    mainline_payload = dict(payload)
    llm_enabled = bool(mainline_payload.get("llm_router", True)) and bool(mainline_payload.get("normalize_input", True))
    if llm_enabled:
        mainline_payload.setdefault("geometry_pipeline", "v2")
        mainline_payload.setdefault("source_pipeline", "v2")
    return process_turn_v2(
        payload=mainline_payload,
        ollama_config_path=get_ollama_config_path(),
        min_confidence=float(payload.get("min_confidence", 0.6)),
        lang=str(payload.get("lang", "zh")).lower(),
        progress_cb=progress_cb,
    )


def handle_strict_reset(session_id: str | None) -> None:
    if not session_id:
        return
    from ui.web.runtime_state import clear_latest_recommended_config

    clear_latest_recommended_config(session_id)
    try:
        _, _, reset_session_v2, _, _ = _session_manager_parts()
    except ModuleNotFoundError:
        return
    reset_session_v2(str(session_id))


def handle_strict_audit(session_id: str) -> list[dict]:
    try:
        get_session_audit_v2, _, _, _, _ = _session_manager_parts()
    except ModuleNotFoundError:
        return []
    return get_session_audit_v2(session_id)


def handle_strict_config_summary(session_id: str, *, lang: str = "zh") -> dict:
    try:
        _, _, _, get_session_config_summary_v2, _ = _session_manager_parts()
    except ModuleNotFoundError as ex:
        return {
            "ok": False,
            "error": f"missing_dependency:{ex.name or 'unknown_dependency'}",
            "session_id": session_id,
            "action_safety_class": "read_only",
        }
    return get_session_config_summary_v2(session_id, lang=lang)


def handle_strict_accept_candidate(payload: dict) -> dict:
    from core.agent import build_agent_state
    from ui.web.runtime_state import (
        get_latest_agent_plan,
        get_latest_recommended_config,
        mark_latest_candidate_accepted,
        set_latest_agent_state,
        set_latest_recommended_config,
    )

    session_id = str(payload.get("session_id", "")).strip()
    config = payload.get("recommended_config")
    if not isinstance(config, dict) or not config:
        config = get_latest_recommended_config(session_id)
    if not session_id:
        return {
            "ok": False,
            "error": "missing_session_id",
            "action_safety_class": "config_mutation",
        }
    if not isinstance(config, dict) or not config:
        return {
            "ok": False,
            "error": "no_recommended_config_available",
            "session_id": session_id,
            "action_safety_class": "config_mutation",
        }
    try:
        _, _, _, _, commit_recommended_config_v2 = _session_manager_parts()
    except ModuleNotFoundError as ex:
        return {
            "ok": False,
            "error": f"missing_dependency:{ex.name or 'unknown_dependency'}",
            "session_id": session_id,
            "action_safety_class": "config_mutation",
        }
    if commit_recommended_config_v2 is None:
        return {
            "ok": False,
            "error": "accept_candidate_unavailable",
            "session_id": session_id,
            "action_safety_class": "config_mutation",
        }
    accepted = mark_latest_candidate_accepted(session_id, committed=False)
    result = commit_recommended_config_v2(
        session_id,
        config,
        source=str(payload.get("source") or "accepted_simulation_design"),
    )
    if result.get("ok"):
        set_latest_recommended_config(session_id, result.get("config") or config)
        accepted = mark_latest_candidate_accepted(session_id, committed=True) or accepted
        plan = get_latest_agent_plan(session_id)
        set_latest_agent_state(
            session_id,
            build_agent_state(plan=plan, candidate_status=accepted, runtime_ready=True),
        )
    result["candidate_status"] = accepted
    return result


def handle_strict_simulation_design(payload: dict, progress_cb=None) -> dict:
    from core.agent import build_agent_plan, build_agent_state
    from ui.web.runtime_state import (
        get_candidate_status,
        get_latest_simulation_design,
        set_latest_agent_plan,
        set_latest_agent_state,
        set_latest_simulation_design,
    )

    design_payload = dict(payload)
    session_id = payload.get("session_id")
    previous = get_latest_simulation_design(session_id)
    if previous and not payload.get("ignore_design_memory"):
        previous_text = str(previous.get("user_text") or "").strip()
        current_text = str(design_payload.get("text", "")).strip()
        if previous_text and current_text:
            design_payload["text"] = (
                "Existing simulation design context:\n"
                f"{previous_text}\n\n"
                "User follow-up request:\n"
                f"{current_text}"
            )
    design_payload["enable_simulation_design"] = True
    design_payload.setdefault("llm_router", True)
    design_payload.setdefault("llm_question", False)
    design_payload.setdefault("normalize_input", False)
    body = handle_strict_step(design_payload, progress_cb=progress_cb)
    set_latest_simulation_design(
        body.get("session_id") or session_id,
        user_text=str(body.get("simulation_design", {}).get("goal") or design_payload.get("text", "")),
        candidate=body.get("simulation_design"),
        recommended_config=body.get("recommended_config"),
        source=str(body.get("simulation_design_source") or ""),
    )
    agent_plan = build_agent_plan(body.get("simulation_design"), recommended_config=body.get("recommended_config"))
    set_latest_agent_plan(body.get("session_id") or session_id, agent_plan)
    candidate_status = get_candidate_status(body.get("session_id") or session_id)
    agent_state = build_agent_state(plan=agent_plan, candidate_status=candidate_status)
    set_latest_agent_state(body.get("session_id") or session_id, agent_state)
    body["agent_plan"] = agent_plan
    body["agent_state"] = agent_state
    body["candidate_status"] = candidate_status
    return body

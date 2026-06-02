from __future__ import annotations

"""Compatibility-only legacy web API glue.

The default UI and agent workflow are v3-first. This module remains importable
for older endpoints and regression tests only.
"""

import re
from typing import Any, Dict, List

from core.config.phase_registry import phase_title
from core.config.prompt_registry import completion_message
from ui.web.legacy_dialogue import (
    ask_for_missing as _ask_llm,
    diff_paths as _diff_paths,
    friendly_fields as _friendly_fields,
    is_complete as _is_complete,
    select_phase_and_fields as _select_phase_fields,
)
from ui.web.legacy_knowledge import (
    is_physics_recommend_request as _is_physics_recommend_request,
    load_knowledge,
    recommend_physics_with_llm as _recommend_physics_with_llm,
)
from ui.web.legacy_router import (
    decide_focus as _decide_focus,
    has_explicit_geometry_assignment as _has_explicit_geometry_assignment,
    infer_geometry_hint as _infer_geometry_hint,
    should_freeze_geometry_update as _should_freeze_geometry_update,
)
from ui.web.legacy_runtime_mapper import (
    apply_frame as _apply_frame,
    apply_text_overrides as _apply_text_overrides,
    build_user_friendly as _build_user_friendly,
    compute_missing as _compute_missing,
    ensure_material_volume_map as _ensure_material_volume_map,
    export_min_config as _export_min_config,
)
from ui.web.legacy_session import (
    SESSIONS,
    build_context_summary as _build_context_summary,
    ensure_session as _ensure_session,
    extract_semantic_frame_legacy as _extract_semantic_frame_legacy,
)
from ui.web.legacy_solver import solve_payload
from ui.web.runtime_state import get_ollama_config_path

KNOWLEDGE = load_knowledge()


def legacy_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    session_id = payload.get("session_id")
    llm_router = bool(payload.get("llm_router", True))
    llm_question = bool(payload.get("llm_question", True))
    normalize_input = bool(payload.get("normalize_input", True))
    min_conf = float(payload.get("min_confidence", 0.6))
    autofix = bool(payload.get("autofix", False))
    lang = str(payload.get("lang", "zh")).lower()
    sid, state = _ensure_session(session_id)
    if not text:
        return {"error": "missing text", "session_id": sid}

    state.history.append({"role": "user", "content": text})
    routed = _decide_focus(text, state.missing_fields, llm_router)
    prev_missing_fields = list(state.missing_fields)
    prev_exported = _export_min_config(state.config)
    context_summary = _build_context_summary(state.config, state.history[:-1])

    frame, debug = _extract_semantic_frame_legacy(
        text,
        min_confidence=min_conf,
        device="auto",
        normalize_with_llm=normalize_input,
        normalize_config_path=get_ollama_config_path(),
        context_summary=context_summary,
    )
    needs_norm = bool(debug.get("requires_llm_normalization", False))
    if needs_norm:
        msg = (
            "请先启用或修复 LLM 归一化（Ollama），再继续提取参数。"
            if lang == "zh"
            else "Please enable/fix LLM normalization (Ollama) before parameter extraction."
        )
        state.history.append({"role": "assistant", "content": msg})
        state.last_question = msg
        return {
            "session_id": sid,
            "router": routed,
            "inference_backend": debug.get("inference_backend", "deferred_non_english"),
            "normalized_text": debug.get("normalized_text", text),
            "normalization": debug.get("normalization", {}),
            "normalization_degraded": bool(debug.get("normalization_degraded", False)),
            "requires_llm_normalization": True,
            "missing_fields": state.missing_fields,
            "assistant_message": msg,
            "phase": "normalization",
            "phase_title": "输入归一化" if lang == "zh" else "Input Normalization",
            "asked_fields": [],
            "asked_fields_friendly": [],
            "is_complete": False,
            "delta_paths": [],
            "display": _build_user_friendly(state.config),
            "config": state.config,
            "config_min": _export_min_config(state.config),
            "history": state.history[-10:],
        }
    if not routed.get("use_geometry_bert", True):
        frame.geometry.structure = None
        frame.geometry.params = {}
    freeze_geo, freeze_reason = _should_freeze_geometry_update(
        text=text,
        routed=routed,
        existing_structure=state.config.get("geometry", {}).get("structure"),
        incoming_structure=frame.geometry.structure,
        incoming_params=frame.geometry.params,
        previous_missing_fields=prev_missing_fields,
    )
    if freeze_geo:
        frame.geometry.structure = None
        frame.geometry.params = {}
        frame.geometry.graph_program = None
        frame.geometry.chosen_skeleton = None
        frame.notes.append(f"geometry_update_skipped:{freeze_reason}")
    geometry_hint = routed.get("geometry_hint") or _infer_geometry_hint(text)
    if freeze_geo:
        geometry_hint = None
    # Do not allow implicit hint to overwrite an existing geometry unless user explicitly requests geometry change.
    if state.config.get("geometry", {}).get("structure") and not _has_explicit_geometry_assignment(text):
        geometry_hint = None
    state.config = _apply_frame(
        state.config,
        frame,
        debug,
        autofix,
        geometry_hint,
    )
    _apply_text_overrides(state.config, text)
    _ensure_material_volume_map(state.config)
    physics_recommendation: Dict[str, Any] = {}
    if _is_physics_recommend_request(text):
        physics_recommendation = _recommend_physics_with_llm(
            text,
            context_summary,
            lang,
            choices=KNOWLEDGE.get("physics_lists", []),
        )
        if physics_recommendation.get("physics_list"):
            state.config["physics"]["physics_list"] = physics_recommendation["physics_list"]
            state.config["physics"]["backup_physics_list"] = physics_recommendation.get("backup_physics_list")
            state.config["physics"]["selection_reasons"] = physics_recommendation.get("reasons", [])
            state.config["physics"]["covered_processes"] = physics_recommendation.get("covered_processes", [])
            state.config["physics"]["selection_source"] = "llm_recommender"
    state.missing_fields = _compute_missing(state.config)
    phase, asked_fields = _select_phase_fields(state.missing_fields)
    asked_fields_friendly = _friendly_fields(asked_fields, lang)
    complete = _is_complete(state.config, state.missing_fields)
    delta_paths = _diff_paths(prev_exported, _export_min_config(state.config))

    question = _ask_llm(asked_fields, asked_fields_friendly, state.history, lang, use_llm=llm_question)
    if complete:
        question = completion_message(lang)
    if question:
        state.history.append({"role": "assistant", "content": question})
        state.last_question = question
    else:
        state.last_question = ""

    return {
        "session_id": sid,
        "router": routed,
        "geometry_update_skipped": freeze_geo,
        "geometry_update_skipped_reason": freeze_reason,
        "context_summary_used": context_summary,
        "inference_backend": debug.get("inference_backend", "dual_model"),
        "normalized_text": debug.get("normalized_text", text),
        "normalization": debug.get("normalization", {}),
        "normalization_degraded": bool(debug.get("normalization_degraded", False)),
        "missing_fields": state.missing_fields,
        "assistant_message": question or completion_message(lang),
        "phase": phase,
        "phase_title": phase_title(phase, lang),
        "asked_fields": asked_fields,
        "asked_fields_friendly": asked_fields_friendly,
        "is_complete": complete,
        "delta_paths": delta_paths,
        "graph_candidates": debug.get("graph_candidates", []),
        "physics_recommendation": physics_recommendation,
        "display": _build_user_friendly(state.config),
        "config": state.config,
        "config_min": _export_min_config(state.config),
        "history": state.history[-10:],
    }


def legacy_solve(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"error": "missing text"}
    return solve_payload(
        text=text,
        top_k=int(payload.get("top_k", 1)),
        min_confidence=float(payload.get("min_confidence", 0.6)),
        normalize_input=bool(payload.get("normalize_input", True)),
        prompt_format=str(payload.get("prompt_format", "json_schema")),
        autofix=bool(payload.get("autofix", False)),
        llm_fill_missing=bool(payload.get("llm_fill_missing", False)),
        params_override=payload.get("params_override", {}) or {},
        extract_semantic_frame=_extract_semantic_frame_legacy,
    )




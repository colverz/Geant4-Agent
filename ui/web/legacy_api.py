from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.config.defaults import build_legacy_default_config
from core.config.field_registry import friendly_labels
from core.config.phase_registry import phase_title, select_phase_fields
from core.config.prompt_registry import clarification_fallback, completion_message
from nlu.llm_support.llm_bridge import build_missing_params_prompt, build_missing_params_schema
from planner.agent import ask_missing
from ui.web.legacy_knowledge import (
    is_physics_recommend_request as _is_physics_recommend_request,
    load_knowledge,
    recommend_physics_with_llm as _recommend_physics_with_llm,
)
from ui.web.legacy_runtime_mapper import (
    apply_frame as _apply_frame,
    apply_text_overrides as _apply_text_overrides,
    build_user_friendly as _build_user_friendly,
    compute_missing as _compute_missing,
    ensure_material_volume_map as _ensure_material_volume_map,
    export_min_config as _export_min_config,
)
from ui.web.runtime_state import get_ollama_config_path

@dataclass
class SessionState:
    history: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    last_question: str = ""


SESSIONS: Dict[str, SessionState] = {}


KNOWLEDGE = load_knowledge()


def _extract_semantic_frame_legacy(*args, **kwargs):
    from nlu.bert_lab.semantic import extract_semantic_frame

    return extract_semantic_frame(*args, **kwargs)


def _match_any(text: str, items: List[str]) -> Optional[str]:
    if not text:
        return None
    ordered = sorted((it for it in items if it), key=len, reverse=True)
    for item in ordered:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])", re.IGNORECASE)
        if pat.search(text):
            return item
    return None


def _default_config() -> Dict[str, Any]:
    return build_legacy_default_config()


def _ensure_session(session_id: Optional[str]) -> Tuple[str, SessionState]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    SESSIONS[sid] = SessionState(config=_default_config())
    return sid, SESSIONS[sid]


def _build_context_summary(config: Dict[str, Any], history: List[Dict[str, str]]) -> str:
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


def _heuristic_focus(text: str) -> List[str]:
    low = text.lower()
    focus = []
    geom_keys = ["box", "tubs", "ring", "grid", "stack", "nest", "shell", "cylinder", "sphere"]
    if any(k in low for k in geom_keys):
        focus.append("geometry")
    if any(k in low for k in ["material", "steel", "aluminum", "g4_"]):
        focus.append("materials")
    if any(k in low for k in ["source", "beam", "gamma", "electron", "proton", "neutron"]):
        focus.append("source")
    if any(k in low for k in ["physics list", "ftfp", "qgsp", "qb"]):
        focus.append("physics")
    return focus or ["geometry"]


def _infer_geometry_hint(text: str) -> Optional[str]:
    low = text.lower()
    if any(k in low for k in ["cube", "box", "立方体", "长方体"]):
        return "single_box"
    if any(k in low for k in ["cylinder", "tubs", "圆柱"]):
        return "single_tubs"
    return None


def _has_explicit_geometry_assignment(text: str) -> bool:
    low = text.lower()
    if re.search(r"\b(structure|geometry)\s*[:=]", low):
        return True
    if re.search(r"(几何|结构)\s*[:=]", text):
        return True

    shape_tokens = (
        "ring",
        "grid",
        "stack",
        "nest",
        "shell",
        "box",
        "cube",
        "cylinder",
        "tubs",
        "sphere",
        "cons",
        "trd",
        "polycone",
        "cuttubs",
        "boolean",
        "环形",
        "阵列",
        "堆叠",
        "嵌套",
        "壳层",
        "立方体",
        "圆柱",
        "球",
    )
    verb_tokens = (
        "use",
        "build",
        "create",
        "make",
        "set",
        "change",
        "switch",
        "update",
        "鏀规垚",
        "鏀逛负",
        "鎹㈡垚",
        "鏀圭敤",
        "璁剧疆",
    )
    has_shape = any(tok in low or tok in text for tok in shape_tokens)
    has_verb = any(tok in low or tok in text for tok in verb_tokens)
    return has_shape and has_verb


def _should_freeze_geometry_update(
    *,
    text: str,
    routed: Dict[str, Any],
    existing_structure: Optional[str],
    incoming_structure: Optional[str],
    incoming_params: Dict[str, float],
    previous_missing_fields: List[str],
) -> Tuple[bool, str]:
    if not existing_structure:
        return False, ""
    explicit_geo = _has_explicit_geometry_assignment(text)
    if explicit_geo:
        return False, ""

    focus = routed.get("focus", [])
    focus_set = {str(x).strip().lower() for x in focus if isinstance(x, str)}
    geom_missing_before = any(f.startswith("geometry.params.") for f in previous_missing_fields)

    if incoming_structure and incoming_structure != existing_structure:
        return True, "structure_change_without_explicit_request"
    if not geom_missing_before and "geometry" not in focus_set and (incoming_structure or incoming_params):
        return True, "non_geometry_turn_locked"
    return False, ""


def _route_with_llm(text: str, missing_fields: List[str]) -> Dict[str, Any]:
    prompt = (
        "You are a router that decides which modules to run next in a Geant4 config builder.\n"
        "Return JSON with keys: use_geometry_bert (bool), focus (list of strings: geometry, materials, source, physics), "
        "geometry_hint (optional: single_box or single_tubs), reason (string).\n"
        f"User text: {text}\n"
        f"Missing fields: {missing_fields}\n"
        "JSON:"
    )
    try:
        resp = chat(prompt, config_path=get_ollama_config_path(), temperature=0.0)
        parsed = extract_json(resp.get("response", "")) or {}
        if isinstance(parsed, dict) and "focus" in parsed:
            return parsed
    except Exception:
        return {}
    return {}


def _decide_focus(text: str, missing_fields: List[str], llm_router: bool) -> Dict[str, Any]:
    allowed_focus = {"geometry", "materials", "source", "physics"}

    def _sanitize_routed(routed: Dict[str, Any]) -> Dict[str, Any]:
        focus_raw = routed.get("focus", [])
        if not isinstance(focus_raw, list):
            focus_raw = []
        focus = []
        for x in focus_raw:
            if not isinstance(x, str):
                continue
            v = x.strip().lower()
            if v in allowed_focus and v not in focus:
                focus.append(v)
        if not focus:
            focus = _heuristic_focus(text)

        hint = str(routed.get("geometry_hint", "")).strip().lower()
        if hint not in {
            "",
            "ring",
            "grid",
            "nest",
            "stack",
            "shell",
            "single_box",
            "single_tubs",
            "single_sphere",
            "single_cons",
            "single_trd",
            "single_polycone",
            "single_cuttubs",
            "boolean",
            "unknown",
        }:
            hint = ""

        use_geo = routed.get("use_geometry_bert")
        if not isinstance(use_geo, bool):
            use_geo = "geometry" in focus
        return {
            "use_geometry_bert": use_geo,
            "focus": focus,
            "geometry_hint": hint,
            "reason": str(routed.get("reason", "llm")).strip(),
        }

    if llm_router:
        routed = _route_with_llm(text, missing_fields)
        if routed:
            return _sanitize_routed(routed)
    focus = _heuristic_focus(text)
    hint = _infer_geometry_hint(text)
    return {
        "use_geometry_bert": "geometry" in focus,
        "focus": focus,
        "geometry_hint": hint,
        "reason": "heuristic",
    }


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (dict, list)) and len(value) == 0:
        return True
    return False


def _collect_required_missing(schema: Dict[str, Any], obj: Any, prefix: str = "") -> List[str]:
    missing: List[str] = []
    if not isinstance(schema, dict):
        return missing
    if schema.get("type") != "object":
        return missing

    props = schema.get("properties", {})
    required = schema.get("required", [])
    value = obj if isinstance(obj, dict) else {}

    for key in required:
        path = f"{prefix}.{key}" if prefix else key
        if key not in value or _is_missing_value(value.get(key)):
            missing.append(path)

    for key, sub_schema in props.items():
        if key in value and value.get(key) is not None:
            path = f"{prefix}.{key}" if prefix else key
            missing.extend(_collect_required_missing(sub_schema, value.get(key), path))

    return missing


def _select_phase_fields(missing_fields: List[str]) -> Tuple[str, List[str]]:
    return select_phase_fields(missing_fields)


def _friendly_fields(fields: List[str], lang: str) -> List[str]:
    return friendly_labels(fields, lang)


def _is_complete(config: Dict[str, Any], missing_fields: List[str]) -> bool:
    if missing_fields:
        return False
    feasible = config.get("geometry", {}).get("feasible")
    if feasible is False:
        return False
    return True


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, p))
        return out
    out[prefix] = obj
    return out


def _diff_paths(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    b = _flatten(before)
    a = _flatten(after)
    keys = sorted(set(b.keys()) | set(a.keys()))
    changed: List[str] = []
    for k in keys:
        if b.get(k) != a.get(k):
            changed.append(k)
    return changed


def _ask_llm(
    asked_fields: List[str],
    asked_fields_friendly: List[str],
    history: List[Dict[str, str]],
    lang: str,
    use_llm: bool = True,
) -> str:
    if not asked_fields:
        return ""
    if not use_llm:
        return clarification_fallback(asked_fields_friendly, lang)
    return ask_missing(asked_fields, lang, get_ollama_config_path(), temperature=1.0)


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

    top_k = int(payload.get("top_k", 1))
    min_conf = float(payload.get("min_confidence", 0.6))
    normalize_input = bool(payload.get("normalize_input", True))
    prompt_format = payload.get("prompt_format", "json_schema")
    autofix = bool(payload.get("autofix", False))
    llm_fill = bool(payload.get("llm_fill_missing", False))
    params_override = payload.get("params_override", {}) or {}

    frame, debug = _extract_semantic_frame_legacy(
        text,
        min_confidence=min_conf,
        device="auto",
        normalize_with_llm=normalize_input,
        normalize_config_path=get_ollama_config_path(),
        context_summary="",
    )
    if bool(debug.get("requires_llm_normalization", False)):
        return {
            "structure": "unknown",
            "inference_backend": debug.get("inference_backend", "deferred_non_english"),
            "normalized_text": debug.get("normalized_text", text),
            "normalization": debug.get("normalization", {}),
            "normalization_degraded": bool(debug.get("normalization_degraded", False)),
            "requires_llm_normalization": True,
            "scores": {},
            "params": {},
            "notes": list(frame.notes),
            "synthesis": {"error": "requires_llm_normalization"},
            "missing_prompt": "",
            "missing_schema": None,
            "candidates": [],
            "best_candidate": None,
        }
    structure = frame.geometry.structure or "unknown"
    params = dict(frame.geometry.params)
    params.update(params_override)
    notes = list(frame.notes)
    scores = debug.get("scores", {})
    ranked = debug.get("ranked", [])

    candidates = []
    for name, prob in ranked[: max(1, top_k)]:
        if prob < min_conf:
            continue
        synth = synthesize_from_params(name, params, seed=7, apply_autofix=autofix)
        missing = synth.get("missing_params", [])
        prompt = build_missing_params_prompt(name, missing, fmt=prompt_format)
        schema = build_missing_params_schema(name, missing) if prompt_format == "json_schema" else None
        filled = None
        if llm_fill and missing:
            resp = chat(prompt, config_path=get_ollama_config_path(), temperature=0.2)
            parsed = extract_json(resp.get("response", ""))
            if isinstance(parsed, dict):
                merged = dict(params)
                merged.update(parsed)
                filled = synthesize_from_params(name, merged, seed=7, apply_autofix=autofix)
        candidates.append(
            {
                "structure": name,
                "prob": prob,
                "synthesis": synth,
                "synthesis_filled": filled,
                "missing_prompt": prompt,
                "missing_schema": schema,
            }
        )

    if structure == "unknown":
        synthesis = {"error": "structure confidence below threshold"}
        missing_prompt = ""
        missing_schema = None
    else:
        synthesis = synthesize_from_params(structure, params, seed=7, apply_autofix=autofix)
        missing = synthesis.get("missing_params", [])
        missing_prompt = build_missing_params_prompt(structure, missing, fmt=prompt_format)
        missing_schema = build_missing_params_schema(structure, missing) if prompt_format == "json_schema" else None

    def _cand_score(c):
        synth = c.get("synthesis", {})
        feasible = bool(synth.get("feasible"))
        missing_n = len(synth.get("missing_params", []))
        errors_n = len(synth.get("errors", []))
        return (1 if feasible else 0, -missing_n, -errors_n, c.get("prob", 0.0))

    best = None
    if candidates:
        best = sorted(candidates, key=_cand_score, reverse=True)[0]

    return {
        "structure": structure,
        "inference_backend": debug.get("inference_backend", "dual_model"),
        "normalized_text": debug.get("normalized_text", text),
        "normalization": debug.get("normalization", {}),
        "normalization_degraded": bool(debug.get("normalization_degraded", False)),
        "scores": scores,
        "params": params,
        "notes": notes,
        "synthesis": synthesis,
        "missing_prompt": missing_prompt,
        "missing_schema": missing_schema,
        "candidates": candidates,
        "graph_candidates": debug.get("graph_candidates", []),
        "best_candidate": best,
    }




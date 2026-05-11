from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from core.audit.audit_log import append_audit_entry
from core.agent.composite_intent import detect_composite_intent
from core.agent.context_pack import build_context_pack
from core.agent.intent_router import route_user_turn
from core.agent.staged_patch import build_staged_patch_reference
from core.agent.turn_trace import NluTurnTrace, stable_hash
from core.agent.workflow_graph import graph_path_for_intent, terminal_state_for_intent
from core.config.defaults import build_strict_default_config
from core.config.field_registry import friendly_label
from core.config.phase_registry import phase_title
from core.domain.lexicon import BASE_MATERIAL_ALIASES, BASE_SOURCE_TYPE_ALIASES
from core.dialogue.grounding import enforce_message_grounding
from core.dialogue.policy import decide_dialogue_action
from core.dialogue.renderer import render_dialogue_message
from core.dialogue.state import build_raw_dialogue, collect_available_explanations, sync_dialogue_state
from core.dialogue.types import build_dialogue_trace
from core.contracts.slots import SlotFrame
from core.geometry.adapters.legacy_compare import compare_slot_frame_geometry
from core.geometry.resolver import (
    GeometryResolutionDraft,
    build_geometry_bridge_seed,
    build_slot_frame_from_geometry_bridge_seed,
    geometry_resolution_to_payload,
    resolve_geometry_from_merged,
)
from core.interpreter import merge_candidates, run_interpreter
from core.pipelines import (
    build_v2_geometry_updates,
    build_v2_geometry_updates_from_candidate,
    build_v2_geometry_updates_from_config,
    build_v2_source_updates,
    build_v2_source_updates_from_candidate,
    build_v2_source_updates_from_config,
    build_v2_spatial_updates,
)
from core.pipelines.selectors import select_pipelines
from core.source.adapters.legacy_compare import compare_slot_frame_source
from core.orchestrator.arbiter import arbitrate_candidates
from core.orchestrator.candidate_preprocess import (
    drop_updates_shadowed_by_anchor,
    filter_candidate_by_explicit_targets,
    partition_candidate_by_pending_paths,
)
from core.orchestrator.candidate_pipeline import (
    augment_geometry_targets as _augment_geometry_targets,
    augment_user_targets as _augment_user_targets,
    candidate_from_updates as _candidate_from_updates,
    candidate_has_update_prefix as _candidate_has_update_prefix,
    candidate_structure as _candidate_structure,
    dedupe_paths as _dedupe_paths,
    retag_candidate as _retag_candidate,
    strip_geometry_updates as _strip_geometry_updates,
    strip_source_updates as _strip_source_updates,
)
from core.orchestrator.confirmation_policy import (
    build_candidate_from_pending_confirmation,
    build_confirmation_payload,
    build_pending_confirmation_item,
    evaluate_confirmation_requirements,
    has_pending_confirmation_path,
    is_unset_for_confirmation,
    merge_pending_confirmations,
)
from core.orchestrator.constraint_ledger import lock_from_candidate
from core.orchestrator.graph_override_policy import should_prefer_extracted_graph
from core.orchestrator.path_ops import deep_copy, get_path, remove_path, set_path
from core.orchestrator.phase_machine import decide_phase_transition
from core.orchestrator.pipeline_debug import merge_v2_meta, merge_v2_missing_paths, prioritize_spatial_questions, prioritize_v2_compile_questions
from core.orchestrator.semantic_sync import build_semantic_sync_candidate
from core.orchestrator.slot_memory import _merge_slot_frame_with_memory, _refresh_slot_memory
from core.orchestrator.turn_transaction import begin_turn, commit_turn
from core.orchestrator.types import CandidateUpdate, Intent, Phase, Producer, SessionState, UpdateOp
from core.runtime.types import ActionSafetyClass
from core.slots.slot_mapper import slot_frame_to_candidates
from core.validation.error_codes import (
    E_CANDIDATE_REJECTED_BY_GATE,
    E_LLM_ROUTER_DISABLED,
    E_PENDING_OVERWRITE_CONFLICT,
)
from core.validation.validator_gate import (
    validate_all,
    validate_layer_c_completeness,
)
from nlu.bert.extractor import extract_candidates_from_normalized_text
from nlu.llm.slot_frame import build_llm_slot_frame
from nlu.llm.semantic_frame import build_llm_semantic_frame
from nlu.llm.normalizer import infer_user_turn_controls, normalize_user_turn
from nlu.llm.recommender import recommend_physics_list
from planner.question_planner import (
    advance_question_state,
    plan_questions,
    to_friendly_labels,
    update_question_attempts,
)


ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge" / "data"


SESSIONS: dict[str, SessionState] = {}
_SESSIONS_LOCK = threading.RLock()


def _load_knowledge() -> dict[str, list[str]]:
    materials = json.loads((KNOWLEDGE_DIR / "materials_geant4_nist.json").read_text(encoding="utf-8")).get(
        "materials", []
    )
    source_types = json.loads((KNOWLEDGE_DIR / "source_constraints.json").read_text(encoding="utf-8")).get("types", [])
    output_formats = json.loads((KNOWLEDGE_DIR / "output_formats.json").read_text(encoding="utf-8")).get("items", [])
    physics_lists = json.loads((KNOWLEDGE_DIR / "physics_lists.json").read_text(encoding="utf-8")).get("items", [])
    return {
        "materials": materials,
        "source_types": source_types or ["point", "beam", "plane", "isotropic"],
        "output_formats": output_formats or ["root", "csv", "hdf5", "xml", "json"],
        "physics_lists": physics_lists,
    }


KNOWLEDGE = _load_knowledge()

_MATERIAL_ALIASES = dict(BASE_MATERIAL_ALIASES)

_SOURCE_TYPE_ALIASES = dict(BASE_SOURCE_TYPE_ALIASES)


def _alias_match(text: str, mapping: dict[str, str], allowed: list[str]) -> str | None:
    low = (text or "").lower()
    for alias, canonical in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        alias_low = alias.lower()
        if re.search(r"[A-Za-z0-9_]", alias_low):
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias_low)}(?![A-Za-z0-9_])", low):
                if canonical in allowed:
                    return canonical
        elif alias in text and canonical in allowed:
            return canonical
    return None


def _extract_forced_material_choice(text: str) -> str | None:
    merged = text or ""
    low = merged.lower()
    # Avoid opportunistic one-token matches unless material is explicitly being set.
    explicit_material_context = bool(
        re.search(r"(?:\bmaterial\b|\bmade of\b|\buse\b|\bwith\b)", low)
        or any(token in merged for token in ("材料", "材质", "用"))
    )
    if not explicit_material_context:
        return None
    return _match_choice(merged, KNOWLEDGE["materials"]) or _alias_match(
        merged,
        _MATERIAL_ALIASES,
        KNOWLEDGE["materials"],
    )


def _extract_forced_source_type_choice(text: str) -> str | None:
    merged = text or ""
    low = merged.lower()
    if "point source" in low or "点源" in merged or "点状源" in merged:
        return "point"
    if any(token in low for token in ("pencil beam", "collimated beam", "beam")) or any(
        token in merged for token in ("束流", "粒子束", "准直束")
    ):
        return "beam"
    if "plane source" in low or "面源" in merged:
        return "plane"
    if "isotropic" in low or "各向同性" in merged:
        return "isotropic"
    return None


def default_config() -> dict[str, Any]:
    return build_strict_default_config()


def get_or_create_session(session_id: str | None) -> SessionState:
    sid = session_id or str(uuid.uuid4())
    with _SESSIONS_LOCK:
        if sid in SESSIONS:
            return SESSIONS[sid]
        state = SessionState(
            session_id=sid,
            phase=Phase.GEOMETRY,
            turn_id=0,
            config=default_config(),
        )
        SESSIONS[sid] = state
        return state


def reset_session(session_id: str) -> None:
    with _SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)


def _build_context_summary(state: SessionState) -> str:
    c = state.config
    parts = [
        f"phase={state.phase.value}",
        f"structure={get_path(c, 'geometry.structure', '')}",
        f"geometry_params={json.dumps(get_path(c, 'geometry.params', {}), ensure_ascii=False)}",
        f"materials={','.join(get_path(c, 'materials.selected_materials', []) or [])}",
        f"source_type={get_path(c, 'source.type', '')}",
        f"source_particle={get_path(c, 'source.particle', '')}",
        f"source_energy={get_path(c, 'source.energy', '')}",
        f"physics_list={get_path(c, 'physics.physics_list', '')}",
        f"output_format={get_path(c, 'output.format', '')}",
    ]
    slot_memory = getattr(state, "slot_memory", {}) or {}
    geometry_memory = slot_memory.get("geometry") if isinstance(slot_memory.get("geometry"), dict) else {}
    source_memory = slot_memory.get("source") if isinstance(slot_memory.get("source"), dict) else {}
    material_memory = slot_memory.get("materials") if isinstance(slot_memory.get("materials"), dict) else {}
    if geometry_memory:
        parts.append(f"memory_geometry={json.dumps(geometry_memory, ensure_ascii=False)}")
    if material_memory:
        parts.append(f"memory_materials={json.dumps(material_memory, ensure_ascii=False)}")
    if source_memory:
        parts.append(f"memory_source={json.dumps(source_memory, ensure_ascii=False)}")
    open_questions = [str(path) for path in getattr(state, "open_questions", []) if str(path)]
    if open_questions:
        parts.append(f"open_questions={json.dumps(open_questions, ensure_ascii=False)}")
    last_asked_paths = [str(path) for path in getattr(state, "last_asked_paths", []) if str(path)]
    if last_asked_paths:
        parts.append(f"last_asked_paths={json.dumps(last_asked_paths, ensure_ascii=False)}")
    return "; ".join(parts)


def _prune_slot_frame_to_targets(frame: SlotFrame, target_paths: list[str]) -> SlotFrame:
    targets = {str(path) for path in target_paths if isinstance(path, str) and path}
    if not targets:
        return frame

    def _wanted(*paths: str) -> bool:
        return any(path in targets for path in paths)

    if not _wanted("geometry.structure", "geometry.kind"):
        frame.geometry.kind = None
    if not any(path in targets for path in ("geometry.params.module_x", "geometry.params.module_y", "geometry.params.module_z")):
        frame.geometry.size_triplet_mm = None
    if not _wanted("geometry.params.child_rmax"):
        frame.geometry.radius_mm = None
    if not _wanted("geometry.params.child_hz"):
        frame.geometry.half_length_mm = None
    if not _wanted("materials.selected_materials", "materials.primary"):
        frame.materials.primary = None

    if not _wanted("source.type", "source.kind"):
        frame.source.kind = None
    if not _wanted("source.particle"):
        frame.source.particle = None
    if not _wanted("source.energy", "source.energy_mev"):
        frame.source.energy_mev = None
    if not _wanted("source.position", "source.position_mm"):
        frame.source.position_mm = None
    if not _wanted("source.direction", "source.direction_vec"):
        frame.source.direction_vec = None
    return frame


def _build_source_pending_dependency_items(
    state_like: Any,
    staged_pending_overwrite: list[dict[str, Any]],
    *,
    lang: str,
) -> list[dict[str, Any]]:
    pending_paths = {str(item.get("path", "")).strip() for item in staged_pending_overwrite}
    trigger_paths = {"source.type", "source.position"}
    if not (pending_paths & trigger_paths):
        return []
    if "source.direction" in pending_paths:
        return []
    old_direction = get_path(state_like.config, "source.direction")
    if is_unset_for_confirmation(old_direction):
        return []
    return [
        {
            "path": "source.direction",
            "field": friendly_label("source.direction", lang),
            "old": old_direction,
            "new": None,
            "producer": Producer.RULE_DEFAULT.value,
        }
    ]


def _stage_dependent_source_updates_for_pending(
    state_like: Any,
    candidates: list[CandidateUpdate],
    staged_pending_overwrite: list[dict[str, Any]],
    *,
    lang: str,
) -> tuple[list[CandidateUpdate], list[dict[str, Any]]]:
    dependency_items = _build_source_pending_dependency_items(
        state_like,
        staged_pending_overwrite,
        lang=lang,
    )
    if not dependency_items:
        return candidates, staged_pending_overwrite
    staged_pending_overwrite = merge_pending_confirmations(staged_pending_overwrite, dependency_items)
    return candidates, staged_pending_overwrite

def _apply_updates(config: dict, updates: list) -> None:
    for upd in updates:
        if upd.op == "remove":
            remove_path(config, upd.path)
            continue
        set_path(config, upd.path, upd.value)


def _build_confirmation_payload_with_staged_reference(
    items: list[dict[str, Any]],
    *,
    lang: str,
    session_id: str,
    turn_id: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = build_confirmation_payload(items, lang=lang)
    if not items:
        return payload
    patch = build_staged_patch_reference(
        session_id=session_id,
        base_turn_id=turn_id,
        base_config=config,
        pending_items=items,
    )
    payload["confirmation_id"] = patch.confirmation_id
    payload["patch_hash"] = patch.patch_hash
    payload["staged_patch"] = {
        "schema_version": patch.schema_version,
        "confirmation_id": patch.confirmation_id,
        "patch_hash": patch.patch_hash,
        "base_turn_id": patch.base_turn_id,
        "base_config_hash": patch.base_config_hash,
        "status": patch.status,
    }
    return payload


def _build_v2_bridge_candidates(
    *,
    base_config: dict[str, Any],
    bridge_source: CandidateUpdate | None,
    pipeline_selection: Any,
    turn_id: int,
    intent: Intent,
    confidence: float,
) -> tuple[list[CandidateUpdate], dict[str, Any]]:
    if bridge_source is None or not bridge_source.updates:
        return [], {}

    bridge_candidates: list[CandidateUpdate] = []
    bridge_meta: dict[str, Any] = {}
    if pipeline_selection.geometry == "v2":
        geometry_updates, geometry_targets, geometry_meta = build_v2_geometry_updates_from_candidate(
            base_config,
            bridge_source.updates,
            turn_id=turn_id,
            confidence=confidence,
        )
        bridge_meta["geometry_v2"] = geometry_meta
        geometry_candidate = _candidate_from_updates(
            intent=intent,
            updates=geometry_updates,
            target_paths=geometry_targets,
            confidence=confidence,
            rationale="v2_geometry_bridge_from_semantic_candidates",
        )
        if geometry_candidate is not None:
            bridge_candidates.append(geometry_candidate)
    if pipeline_selection.source == "v2":
        source_updates, source_targets, source_meta = build_v2_source_updates_from_candidate(
            base_config,
            bridge_source.updates,
            turn_id=turn_id,
            confidence=confidence,
        )
        bridge_meta["source_v2"] = source_meta
        source_candidate = _candidate_from_updates(
            intent=intent,
            updates=source_updates,
            target_paths=source_targets,
            confidence=confidence,
            rationale="v2_source_bridge_from_semantic_candidates",
        )
        if source_candidate is not None:
            bridge_candidates.append(source_candidate)
    return bridge_candidates, bridge_meta


def _build_geometry_evidence_from_slot_frame(frame: Any) -> dict[str, Any]:
    geometry = getattr(frame, "geometry", None)
    materials = getattr(frame, "materials", None)
    if geometry is None:
        return {}
    dimensions: dict[str, Any] = {}
    for key in (
        "size_triplet_mm",
        "radius_mm",
        "half_length_mm",
        "radius1_mm",
        "radius2_mm",
        "x1_mm",
        "x2_mm",
        "y1_mm",
        "y2_mm",
        "z_mm",
    ):
        value = getattr(geometry, key, None)
        if value is not None:
            dimensions[key] = value
    return {
        "kind": getattr(geometry, "kind", None),
        "material": getattr(materials, "primary", None) if materials is not None else None,
        "dimensions": dimensions,
    }


def _build_source_evidence_from_slot_frame(frame: Any) -> dict[str, Any]:
    source = getattr(frame, "source", None)
    if source is None:
        return {}
    position_value = getattr(source, "position_mm", None)
    direction_value = getattr(source, "direction_vec", None)
    position = {"position_mm": position_value} if position_value is not None else None
    direction = (
        {"mode": "explicit_vector", "hint": {"direction_vec": direction_value}}
        if direction_value is not None
        else None
    )
    return {
        "source_type": getattr(source, "kind", None),
        "particle": getattr(source, "particle", None),
        "energy_mev": getattr(source, "energy_mev", None),
        "position": position,
        "direction": direction,
    }


def _build_interpreter_sidecar(
    *,
    text: str,
    context_summary: str,
    slot_frame: Any,
    ollama_config_path: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": 0.0}
    if ollama_config_path:
        kwargs["config_path"] = ollama_config_path
    result = run_interpreter(text, context_summary, **kwargs)
    payload: dict[str, Any] = {
        "ok": result.ok,
        "fallback_reason": result.fallback_reason,
        "turn_summary": result.parsed.turn_summary.to_payload(),
        "geometry_candidate": result.parsed.geometry_candidate.to_payload(),
        "source_candidate": result.parsed.source_candidate.to_payload(),
    }
    if slot_frame is None:
        return payload
    merged = merge_candidates(
        result.parsed.turn_summary,
        result.parsed.geometry_candidate,
        result.parsed.source_candidate,
        geometry_evidence=_build_geometry_evidence_from_slot_frame(slot_frame),
        source_evidence=_build_source_evidence_from_slot_frame(slot_frame),
    )
    payload["merged"] = merged.to_payload()
    try:
        geometry_resolution = resolve_geometry_from_merged(merged.merged_geometry)
        payload["geometry_resolution"] = geometry_resolution_to_payload(geometry_resolution)
    except Exception as exc:
        payload["geometry_resolution"] = {
            "error": f"geometry_resolution_error:{type(exc).__name__}",
        }
    return payload


def _accepted_interpreter_field(field: Any) -> Any | None:
    if not isinstance(field, dict):
        return None
    if field.get("conflict"):
        return None
    return field.get("value")


def _build_interpreter_bridge_candidates(
    *,
    interpreter_debug: dict[str, Any] | None,
    pipeline_selection: Any,
    base_config: dict[str, Any],
    turn_id: int,
    intent: Intent,
    seed_updates: list[UpdateOp],
) -> tuple[list[CandidateUpdate], dict[str, Any]]:
    if not isinstance(interpreter_debug, dict):
        return [], {}
    merged = interpreter_debug.get("merged")
    if not isinstance(merged, dict):
        return [], {}

    bridge_candidates: list[CandidateUpdate] = []
    bridge_meta: dict[str, Any] = {}
    merged_geometry = merged.get("merged_geometry") if isinstance(merged.get("merged_geometry"), dict) else {}
    merged_source = merged.get("merged_source") if isinstance(merged.get("merged_source"), dict) else {}
    existing_paths = {str(update.path) for update in seed_updates if str(update.path)}

    interpreter_confidence = max(
        float(((interpreter_debug.get("geometry_candidate") or {}).get("confidence") or 0.0)),
        float(((interpreter_debug.get("source_candidate") or {}).get("confidence") or 0.0)),
        0.65,
    )

    if pipeline_selection.geometry == "v2":
        geometry_resolution = interpreter_debug.get("geometry_resolution") if isinstance(interpreter_debug, dict) else None
        geometry_draft = geometry_resolution.get("draft") if isinstance(geometry_resolution, dict) else None
        draft_obj = None
        if isinstance(geometry_draft, dict):
            draft_obj = GeometryResolutionDraft(
                structure=geometry_draft.get("structure"),
                material=geometry_draft.get("material"),
                params=dict(geometry_draft.get("params") or {}),
                conflicts=tuple(geometry_draft.get("conflicts") or ()),
                ambiguities=tuple(geometry_draft.get("ambiguities") or ()),
                open_questions=tuple(geometry_draft.get("open_questions") or ()),
                trust_report=dict(geometry_draft.get("trust_report") or {}),
                bridge_allowed=bool(geometry_draft.get("bridge_allowed", False)),
            )
        geometry_seed = build_geometry_bridge_seed(
            draft=draft_obj,
            merged_geometry_payload=merged_geometry,
        )
        geometry_seed_config = deep_copy(base_config)
        for update in seed_updates:
            if update.op == "set":
                set_path(geometry_seed_config, update.path, update.value)
        interp_frame = build_slot_frame_from_geometry_bridge_seed(
            geometry_seed,
            intent=intent,
            confidence=interpreter_confidence,
        )

        interp_geometry_updates, _, _ = build_v2_geometry_updates(interp_frame, turn_id=turn_id)
        for update in interp_geometry_updates:
            if update.op == "set":
                set_path(geometry_seed_config, update.path, update.value)

        geometry_updates, geometry_targets, geometry_meta = build_v2_geometry_updates_from_config(
            geometry_seed_config,
            turn_id=turn_id,
            confidence=interpreter_confidence,
        )
        filtered_geometry_updates = [update for update in geometry_updates if str(update.path) not in existing_paths]
        geometry_target_paths = [path for path in geometry_targets if str(path) not in existing_paths]
        if isinstance(geometry_seed.material, str) and geometry_seed.material and "materials.selected_materials" not in existing_paths:
            filtered_geometry_updates.append(
                UpdateOp(
                    path="materials.selected_materials",
                    op="set",
                    value=[geometry_seed.material],
                    producer=Producer.LLM_SEMANTIC_FRAME,
                    confidence=interpreter_confidence,
                    turn_id=turn_id,
                )
            )
            geometry_target_paths = _dedupe_paths(
                list(geometry_target_paths) + ["materials.selected_materials", "materials.volume_material_map"]
            )
        if filtered_geometry_updates:
            geometry_candidate = CandidateUpdate(
                producer=Producer.LLM_SEMANTIC_FRAME,
                intent=intent,
                target_paths=list(geometry_target_paths),
                updates=[
                    UpdateOp(
                        path=update.path,
                        op=update.op,
                        value=update.value,
                        producer=Producer.LLM_SEMANTIC_FRAME,
                        confidence=interpreter_confidence,
                        turn_id=update.turn_id,
                    )
                    for update in filtered_geometry_updates
                ],
                confidence=interpreter_confidence,
                rationale="interpreter_geometry_bridge",
            )
            bridge_candidates.append(geometry_candidate)
        bridge_meta["interpreter_geometry"] = {
            "used": bool(filtered_geometry_updates),
            "compile_ok": bool(geometry_meta.get("compile_ok")),
            "runtime_ready": bool(geometry_meta.get("runtime_ready")),
            "used_resolution": isinstance(geometry_draft, dict) and not geometry_draft.get("error"),
            "bridge_allowed": bool(geometry_draft.get("bridge_allowed", False)) if isinstance(geometry_draft, dict) else False,
        }

    if pipeline_selection.source == "v2":
        source_type = _accepted_interpreter_field(merged_source.get("source_type"))
        particle = _accepted_interpreter_field(merged_source.get("particle"))
        energy = _accepted_interpreter_field(merged_source.get("energy_mev"))
        position = _accepted_interpreter_field(merged_source.get("position"))
        direction = _accepted_interpreter_field(merged_source.get("direction"))

        source_seed_config = deep_copy(base_config)
        for update in seed_updates:
            if update.op == "set":
                set_path(source_seed_config, update.path, update.value)
        if isinstance(source_type, str) and source_type:
            set_path(source_seed_config, "source.type", source_type)
        if isinstance(particle, str) and particle:
            set_path(source_seed_config, "source.particle", particle)
        if energy is not None:
            set_path(source_seed_config, "source.energy", float(energy))
        if isinstance(position, dict):
            pos_value = position.get("position_mm")
            if isinstance(pos_value, list) and len(pos_value) == 3 and all(v is not None for v in pos_value):
                set_path(
                    source_seed_config,
                    "source.position",
                    {"type": "vector", "value": [float(pos_value[0]), float(pos_value[1]), float(pos_value[2])]},
                )
        if isinstance(direction, dict):
            hint = direction.get("hint")
            if isinstance(hint, dict):
                dir_value = hint.get("direction_vec")
                if isinstance(dir_value, list) and len(dir_value) == 3 and all(v is not None for v in dir_value):
                    set_path(
                        source_seed_config,
                        "source.direction",
                        {"type": "vector", "value": [float(dir_value[0]), float(dir_value[1]), float(dir_value[2])]},
                    )

        source_updates, source_targets, source_meta = build_v2_source_updates_from_config(
            source_seed_config,
            turn_id=turn_id,
            confidence=interpreter_confidence,
        )
        filtered_source_updates = [update for update in source_updates if str(update.path) not in existing_paths]
        source_target_paths = [path for path in source_targets if str(path) not in existing_paths]
        if filtered_source_updates:
            bridge_candidates.append(
                CandidateUpdate(
                    producer=Producer.LLM_SEMANTIC_FRAME,
                    intent=intent,
                    target_paths=list(source_target_paths),
                    updates=[
                        UpdateOp(
                            path=update.path,
                            op=update.op,
                            value=update.value,
                            producer=Producer.LLM_SEMANTIC_FRAME,
                            confidence=interpreter_confidence,
                            turn_id=update.turn_id,
                        )
                        for update in filtered_source_updates
                    ],
                    confidence=interpreter_confidence,
                    rationale="interpreter_source_bridge",
                )
            )
        bridge_meta["interpreter_source"] = {
            "used": bool(filtered_source_updates),
            "compile_ok": bool(source_meta.get("compile_ok")),
            "runtime_ready": bool(source_meta.get("runtime_ready")),
            "missing_fields": list(source_meta.get("missing_fields", []) or []),
        }

    return bridge_candidates, bridge_meta


def _apply_explicit_user_controls(
    user_candidate: CandidateUpdate,
    explicit_controls: dict[str, Any],
) -> CandidateUpdate:
    explicit_intent = explicit_controls.get("intent", user_candidate.intent)
    strict_targets = bool(explicit_controls.get("strict_targets", False))
    explicit_targets = [
        str(path)
        for path in explicit_controls.get("target_paths", [])
        if isinstance(path, str) and path
    ]
    intent = user_candidate.intent
    if explicit_intent in {Intent.CONFIRM, Intent.REJECT, Intent.MODIFY, Intent.REMOVE, Intent.QUESTION}:
        intent = explicit_intent
    elif (
        explicit_intent == Intent.SET
        and explicit_targets
        and user_candidate.intent in {Intent.QUESTION, Intent.OTHER}
    ):
        # Deterministic control parser can downgrade spurious LLM question intent
        # when clear set-targets are present in the same turn.
        intent = Intent.SET
    elif user_candidate.intent == Intent.OTHER:
        intent = explicit_intent
    existing_targets = [str(path) for path in user_candidate.target_paths if isinstance(path, str) and path]
    if explicit_targets:
        if strict_targets:
            target_paths = sorted(set(explicit_targets))
        else:
            # Keep explicit detections while preserving high-confidence slot paths,
            # otherwise we may accidentally drop extracted geometry/source slots.
            target_paths = sorted(set(explicit_targets) | set(existing_targets))
    else:
        target_paths = list(existing_targets)
    if intent == user_candidate.intent and target_paths == list(user_candidate.target_paths):
        return user_candidate
    return CandidateUpdate(
        producer=user_candidate.producer,
        intent=intent,
        target_paths=target_paths,
        updates=list(user_candidate.updates),
        confidence=user_candidate.confidence,
        rationale=f"{user_candidate.rationale}_explicit_controls",
    )


def _looks_like_focused_answer(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return False
    if len(compact) <= 48:
        return True
    lowered = compact.lower()
    return bool(
        re.fullmatch(r"(direction|energy|position)\s*(?:is|=|:)?\s*.+", lowered)
        or re.fullmatch(r"(方向|能量|位置)\s*(?:是|为|：|:)?\s*.+", compact)
    )


def _refine_explicit_controls_for_question_answer(
    *,
    state: SessionState,
    text: str,
    explicit_controls: dict[str, Any],
) -> dict[str, Any]:
    asked_paths = [str(path) for path in getattr(state, "last_asked_paths", []) if str(path)]
    if not asked_paths:
        return explicit_controls
    if explicit_controls.get("intent") in {Intent.CONFIRM, Intent.REJECT, Intent.MODIFY, Intent.REMOVE, Intent.QUESTION}:
        return explicit_controls
    explicit_targets = [
        str(path)
        for path in explicit_controls.get("target_paths", [])
        if isinstance(path, str) and path
    ]
    asked_set = set(asked_paths)
    if explicit_targets and set(explicit_targets).issubset(asked_set):
        refined = dict(explicit_controls)
        refined["target_paths"] = sorted(set(explicit_targets))
        refined["strict_targets"] = True
        return refined
    if not explicit_targets and len(asked_paths) == 1 and _looks_like_focused_answer(text):
        refined = dict(explicit_controls)
        refined["target_paths"] = [asked_paths[0]]
        refined["strict_targets"] = True
        return refined
    return explicit_controls


def _stage_dependent_geometry_updates_for_pending(
    draft: Any,
    candidates: list[CandidateUpdate],
    staged_pending_overwrite: list[dict[str, Any]],
    *,
    lang: str,
) -> tuple[list[CandidateUpdate], list[dict[str, Any]]]:
    # When structure overwrite is pending confirmation, geometry params from
    # the same turn must be committed atomically with structure after confirm.
    if not has_pending_confirmation_path(staged_pending_overwrite, "geometry.structure"):
        return candidates, staged_pending_overwrite

    dependency_items: list[dict[str, Any]] = []
    filtered_candidates: list[CandidateUpdate] = []
    for candidate in candidates:
        kept: list[UpdateOp] = []
        for update in candidate.updates:
            if update.path.startswith("geometry.params."):
                dependency_items.append(
                    build_pending_confirmation_item(
                        update,
                        draft=draft,
                        lang=lang,
                        producer=candidate.producer.value,
                    )
                )
                continue
            kept.append(update)
        if not kept:
            continue
        if len(kept) == len(candidate.updates):
            filtered_candidates.append(candidate)
            continue
        filtered_candidates.append(
            CandidateUpdate(
                producer=candidate.producer,
                intent=candidate.intent,
                target_paths=sorted({u.path for u in kept}),
                updates=kept,
                confidence=candidate.confidence,
                rationale=f"{candidate.rationale}_geometry_deferred",
            )
        )

    if dependency_items:
        staged_pending_overwrite = merge_pending_confirmations(staged_pending_overwrite, dependency_items)
    return filtered_candidates, staged_pending_overwrite


def _match_choice(text: str, options: list[str]) -> str | None:
    low = (text or "").lower()
    ordered = sorted((str(x) for x in options if str(x)), key=len, reverse=True)
    for item in ordered:
        pat = rf"(?<![A-Za-z0-9_]){item.lower()}(?![A-Za-z0-9_])"
        if re.search(pat, low):
            return item
    return None


def _has_explicit_physics_choice(text: str, options: list[str]) -> bool:
    return _match_choice(text, options) is not None


def _build_forced_explicit_candidate(
    *,
    text: str,
    normalized_text: str,
    user_candidate: CandidateUpdate,
    turn_id: int,
) -> CandidateUpdate | None:
    merged = f"{text} ; {normalized_text}".strip(" ;")
    updates: list[UpdateOp] = []

    physics_choice = _match_choice(merged, KNOWLEDGE["physics_lists"])
    should_try_physics = (
        user_candidate.intent in {Intent.SET, Intent.MODIFY}
        or "physics.physics_list" in user_candidate.target_paths
        or (physics_choice is not None and user_candidate.intent not in {Intent.CONFIRM, Intent.REMOVE})
    )
    if should_try_physics:
        if physics_choice:
            updates.append(
                UpdateOp(
                    path="physics.physics_list",
                    op="set",
                    value=physics_choice,
                    producer=Producer.USER_EXPLICIT,
                    confidence=1.0,
                    turn_id=turn_id,
                )
            )

    output_fmt = _match_choice(merged, KNOWLEDGE["output_formats"])
    should_try_output = (
        user_candidate.intent in {Intent.SET, Intent.MODIFY}
        or "output.format" in user_candidate.target_paths
        or (output_fmt is not None and user_candidate.intent not in {Intent.CONFIRM, Intent.REMOVE})
    )
    if should_try_output:
        if output_fmt:
            updates.append(
                UpdateOp(
                    path="output.format",
                    op="set",
                    value=output_fmt,
                    producer=Producer.USER_EXPLICIT,
                    confidence=1.0,
                    turn_id=turn_id,
                )
            )

    if not updates:
        return None
    return CandidateUpdate(
        producer=Producer.USER_EXPLICIT,
        intent=Intent.SET,
        target_paths=sorted({u.path for u in updates}),
        updates=updates,
        confidence=1.0,
        rationale="forced_explicit_text_choice",
    )


def _has_geometry_signal(*, debug: dict[str, Any], committed_updates: list[UpdateOp], user_candidate: CandidateUpdate) -> bool:
    graph_choice = debug.get("graph_choice", {}) if isinstance(debug, dict) else {}
    chosen_skeleton = str(graph_choice.get("chosen_skeleton", "") or "").strip()
    structure = str(graph_choice.get("structure", "") or "").strip()
    if chosen_skeleton:
        return True
    if structure and structure != "unknown":
        return True
    if any(update.path.startswith("geometry.") for update in committed_updates):
        return True
    return any(str(path).startswith("geometry.") for path in user_candidate.target_paths)


def _semantic_missing_from_debug(debug: dict[str, Any]) -> list[str]:
    if not isinstance(debug, dict):
        return []
    graph_choice = debug.get("graph_choice", {})
    if not isinstance(graph_choice, dict):
        return []
    return _dedupe_paths(list(graph_choice.get("dialogue_missing_paths", []) or []))
def _progress(progress_cb, stage: str, label: str, detail: str | None = None) -> None:
    if progress_cb:
        progress_cb(stage, label, detail)


def process_turn(
    payload: dict,
    *,
    ollama_config_path: str,
    min_confidence: float = 0.6,
    lang: str = "zh",
    progress_cb=None,
) -> dict:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"error": "missing text"}
    _progress(progress_cb, "start", "Reading request", "Preparing session state and turn context.")
    state = get_or_create_session(payload.get("session_id"))
    turn_id_before = state.turn_id
    intent_decision = route_user_turn(text, lang)
    composite_intent = detect_composite_intent(text)
    previous_missing_paths = _dedupe_paths(
        validate_layer_c_completeness(state.config).missing_required_paths + list(state.semantic_missing_paths)
    )
    state.turn_id += 1
    state.history.append({"role": "user", "content": text})

    before_config = deep_copy(state.config)
    draft = begin_turn(state)
    context_summary = _build_context_summary(state)
    context_pack = build_context_pack(
        user_turn=text,
        intent_decision=intent_decision,
        config=before_config,
        staged_patch_summary={"pending_overwrite_count": len(state.pending_overwrite)}
        if state.pending_overwrite
        else None,
    )
    pipeline_selection = select_pipelines(
        geometry=str(payload.get("geometry_pipeline", "")).strip() or None,
        source=str(payload.get("source_pipeline", "")).strip() or None,
    )
    llm_router = bool(payload.get("llm_router", True))
    llm_question = bool(payload.get("llm_question", True))
    internal_temperature = 0.0
    user_temperature = float(payload.get("user_temperature", 1.0))
    normalize_input = bool(payload.get("normalize_input", True))
    apply_autofix = bool(payload.get("autofix", False))
    enable_compare = bool(payload.get("enable_compare", True))
    enable_interpreter = bool(payload.get("enable_interpreter", False))
    enable_llm_first = bool(llm_router and normalize_input)
    llm_used = False
    fallback_reason = E_LLM_ROUTER_DISABLED if (not llm_router and normalize_input) else "E_LLM_DISABLED"
    llm_raw = ""
    llm_schema_errors: list[str] = []
    llm_stage_failures: list[str] = []
    slot_debug: dict[str, Any] = {}
    geometry_compare: dict[str, Any] | None = None
    source_compare: dict[str, Any] | None = None
    interpreter_debug: dict[str, Any] | None = None
    debug: dict[str, Any] = {"graph_candidates": []}
    semantic_missing_paths = list(state.semantic_missing_paths)
    normalized_text = text
    content_candidates: list[CandidateUpdate] = []
    user_candidate: CandidateUpdate | None = None
    applying_pending_overwrite = False
    confirm_apply_failed = False
    staged_pending_overwrite: list[dict[str, Any]] = []
    rejected_overwrite_preview: list[dict[str, Any]] = []
    slot_frame_memory_snapshot: SlotFrame | None = None
    explicit_controls = infer_user_turn_controls(text)
    explicit_controls = _refine_explicit_controls_for_question_answer(
        state=state,
        text=text,
        explicit_controls=explicit_controls,
    )
    _progress(progress_cb, "intent", "Interpreting intent", "User controls and overwrite intent parsed.")

    if enable_llm_first:
        _progress(progress_cb, "slot_frame", "Building slot frame", "Running LLM-first slot extraction.")
        slot_result = build_llm_slot_frame(
            text,
            context_summary=context_summary,
            config_path=ollama_config_path,
        )
        if slot_result.ok and slot_result.frame:
            if explicit_controls.get("strict_targets"):
                slot_result.frame = _prune_slot_frame_to_targets(
                    slot_result.frame,
                    list(explicit_controls.get("target_paths", [])),
                )
            slot_result.frame = _merge_slot_frame_with_memory(slot_result.frame, getattr(state, "slot_memory", {}))
            slot_frame_memory_snapshot = deep_copy(slot_result.frame)
            spatial_v2_meta: dict[str, Any] | None = None
            source_v2_meta: dict[str, Any] | None = None
            slot_candidate, user_candidate = slot_frame_to_candidates(
                slot_result.frame,
                turn_id=state.turn_id,
                geometry_mode=pipeline_selection.geometry,
                source_mode=pipeline_selection.source,
            )
            user_candidate = _apply_explicit_user_controls(user_candidate, explicit_controls)
            normalized_text = slot_result.normalized_text or text
            slot_debug = dict(slot_result.stage_trace or {})
            slot_debug.setdefault("final_status", "ok")
            if pipeline_selection.geometry == "v2" and pipeline_selection.source == "v2":
                spatial_result = build_v2_spatial_updates(slot_result.frame, turn_id=state.turn_id)
                spatial_v2_meta = {
                    "warnings": list(spatial_result.warnings),
                    "spatial_meta": dict(spatial_result.spatial_meta),
                    "geometry_meta": dict(spatial_result.geometry_meta),
                    "source_meta": dict(spatial_result.source_meta),
                }
                slot_debug["spatial_v2"] = spatial_v2_meta
                slot_debug["geometry_v2"] = dict(spatial_result.geometry_meta)
                slot_debug["source_v2"] = dict(spatial_result.source_meta)
            elif pipeline_selection.geometry == "v2":
                _, _, geometry_v2_meta = build_v2_geometry_updates(slot_result.frame, turn_id=state.turn_id)
                slot_debug["geometry_v2"] = geometry_v2_meta
            elif pipeline_selection.source == "v2":
                _, _, source_v2_meta = build_v2_source_updates(slot_result.frame, turn_id=state.turn_id)
                slot_debug["source_v2"] = source_v2_meta
            if enable_compare:
                geometry_compare = compare_slot_frame_geometry(slot_result.frame, turn_id=state.turn_id)
                source_compare = compare_slot_frame_source(slot_result.frame, turn_id=state.turn_id)
            if enable_interpreter:
                try:
                    interpreter_debug = _build_interpreter_sidecar(
                        text=text,
                        context_summary=context_summary,
                        slot_frame=slot_result.frame,
                        ollama_config_path=ollama_config_path,
                    )
                except Exception as exc:
                    interpreter_debug = {
                        "ok": False,
                        "fallback_reason": f"interpreter_sidecar_error:{type(exc).__name__}",
                    }
            _progress(progress_cb, "semantic_extract", "Extracting semantic candidates", "Runtime semantic extraction from normalized text.")
            extracted_candidate, debug = extract_candidates_from_normalized_text(
                normalized_text,
                raw_text=text,
                turn_id=state.turn_id,
                min_confidence=min_confidence,
                context_summary=context_summary,
                config_path=ollama_config_path,
                apply_autofix=apply_autofix,
            )
            bridge_source_candidate = extracted_candidate
            slot_structure = _candidate_structure(slot_candidate)
            extracted_structure = _candidate_structure(extracted_candidate)
            if pipeline_selection.geometry == "v2":
                extracted_candidate = _strip_geometry_updates(extracted_candidate)
            if pipeline_selection.source == "v2":
                extracted_candidate = _strip_source_updates(extracted_candidate)
            slot_geometry_ready = bool((slot_debug.get("geometry_v2") or {}).get("runtime_ready"))
            prefer_extracted_graph = should_prefer_extracted_graph(
                text=text,
                extracted_structure=extracted_structure,
                slot_structure=slot_structure,
                slot_geometry_ready=slot_geometry_ready,
            )
            if prefer_extracted_graph:
                user_candidate = _augment_geometry_targets(user_candidate, extracted_candidate)
                slot_candidate = _strip_geometry_updates(slot_candidate)
            if slot_candidate is not None:
                slot_candidate = filter_candidate_by_explicit_targets(slot_candidate, list(user_candidate.target_paths))
            if user_candidate.target_paths:
                extracted_candidate = filter_candidate_by_explicit_targets(extracted_candidate, list(user_candidate.target_paths))
            extracted_candidate = drop_updates_shadowed_by_anchor(extracted_candidate, slot_candidate)
            bridge_candidates: list[CandidateUpdate] = []
            bridge_meta: dict[str, Any] = {}
            needs_geometry_bridge = pipeline_selection.geometry == "v2" and not _candidate_has_update_prefix(slot_candidate, "geometry.")
            needs_source_bridge = pipeline_selection.source == "v2" and not _candidate_has_update_prefix(slot_candidate, "source.")
            if needs_geometry_bridge or needs_source_bridge:
                bridge_candidates, bridge_meta = _build_v2_bridge_candidates(
                    base_config=draft.config,
                    bridge_source=bridge_source_candidate,
                    pipeline_selection=pipeline_selection,
                    turn_id=state.turn_id,
                    intent=user_candidate.intent,
                    confidence=max(float(slot_result.confidence or 0.0), float(user_candidate.confidence or 0.0), 0.8),
                )
                if not needs_geometry_bridge:
                    bridge_candidates = [candidate for candidate in bridge_candidates if not _candidate_has_update_prefix(candidate, "geometry.")]
                    bridge_meta.pop("geometry_v2", None)
                if not needs_source_bridge:
                    bridge_candidates = [candidate for candidate in bridge_candidates if not _candidate_has_update_prefix(candidate, "source.")]
                    bridge_meta.pop("source_v2", None)
                if bridge_candidates:
                    bridge_paths = [update.path for candidate in bridge_candidates for update in candidate.updates]
                    user_candidate = _augment_user_targets(user_candidate, bridge_paths)
                for key, meta in bridge_meta.items():
                    slot_debug[key] = merge_v2_meta(slot_debug.get(key), meta)
            interpreter_bridge_candidates: list[CandidateUpdate] = []
            interpreter_bridge_meta: dict[str, Any] = {}
            if enable_interpreter and interpreter_debug and interpreter_debug.get("ok"):
                interpreter_bridge_candidates, interpreter_bridge_meta = _build_interpreter_bridge_candidates(
                    interpreter_debug=interpreter_debug,
                    pipeline_selection=pipeline_selection,
                    base_config=draft.config,
                    turn_id=state.turn_id,
                    intent=user_candidate.intent,
                    seed_updates=[
                        *(slot_candidate.updates if slot_candidate is not None else []),
                        *[update for candidate in bridge_candidates for update in candidate.updates],
                    ],
                )
                if interpreter_bridge_candidates:
                    bridge_paths = [update.path for candidate in interpreter_bridge_candidates for update in candidate.updates]
                    user_candidate = _augment_user_targets(user_candidate, bridge_paths)
                for key, meta in interpreter_bridge_meta.items():
                    slot_debug[key] = meta
            if slot_candidate is not None and slot_candidate.updates:
                content_candidates.append(slot_candidate)
            content_candidates.extend(bridge_candidates)
            content_candidates.extend(interpreter_bridge_candidates)
            if extracted_candidate.updates:
                content_candidates.append(extracted_candidate)
            llm_used = True
            llm_raw = slot_result.llm_raw
            llm_schema_errors = list(slot_result.schema_errors)
            semantic_missing_paths = merge_v2_missing_paths(
                list(semantic_missing_paths),
                slot_debug,
                include_spatial=spatial_v2_meta is not None,
            )
            fallback_reason = None
            normalization_payload = {
                "intent": user_candidate.intent.value,
                "confidence": slot_result.confidence,
                "backend": "llm_slot_frame",
            }
            debug["inference_backend"] = "llm_slot_frame+runtime_semantic"
        else:
            slot_debug = dict(slot_result.stage_trace or {})
            if slot_result.fallback_reason:
                llm_stage_failures.append(slot_result.fallback_reason)
            _progress(progress_cb, "semantic_frame", "Falling back to semantic frame", "Slot frame fallback triggered; trying semantic-frame route.")
            semantic_result = build_llm_semantic_frame(
                text,
                context_summary=context_summary,
                config_path=ollama_config_path,
                turn_id=state.turn_id,
            )
            if semantic_result.ok and semantic_result.candidate and semantic_result.user_candidate:
                user_candidate = _apply_explicit_user_controls(semantic_result.user_candidate, explicit_controls)
                normalized_text = semantic_result.normalized_text or text
                _progress(progress_cb, "semantic_extract", "Extracting semantic candidates", "Runtime semantic extraction from semantic-frame text.")
                extracted_candidate, debug = extract_candidates_from_normalized_text(
                    normalized_text,
                    raw_text=text,
                    turn_id=state.turn_id,
                    min_confidence=min_confidence,
                    context_summary=context_summary,
                    config_path=ollama_config_path,
                    apply_autofix=apply_autofix,
                )
                semantic_candidate = filter_candidate_by_explicit_targets(
                    semantic_result.candidate,
                    list(user_candidate.target_paths),
                )
                bridge_updates = list(semantic_candidate.updates)
                bridge_source_candidate = CandidateUpdate(
                    producer=Producer.BERT_EXTRACTOR,
                    intent=user_candidate.intent,
                    target_paths=list(user_candidate.target_paths),
                    updates=bridge_updates,
                    confidence=max(float(semantic_result.confidence or 0.0), float(user_candidate.confidence or 0.0), 0.8),
                    rationale="v2_bridge_seed_from_semantic_frame",
                )
                if pipeline_selection.geometry == "v2":
                    semantic_candidate = _strip_geometry_updates(semantic_candidate)
                if pipeline_selection.source == "v2":
                    semantic_candidate = _strip_source_updates(semantic_candidate)
                if user_candidate.target_paths:
                    extracted_candidate = filter_candidate_by_explicit_targets(extracted_candidate, list(user_candidate.target_paths))
                bridge_source_candidate = CandidateUpdate(
                    producer=bridge_source_candidate.producer,
                    intent=bridge_source_candidate.intent,
                    target_paths=list(bridge_source_candidate.target_paths),
                    updates=list(bridge_source_candidate.updates) + list(extracted_candidate.updates),
                    confidence=bridge_source_candidate.confidence,
                    rationale=bridge_source_candidate.rationale,
                )
                extracted_candidate = drop_updates_shadowed_by_anchor(extracted_candidate, semantic_candidate)
                bridge_candidates, bridge_meta = _build_v2_bridge_candidates(
                    base_config=draft.config,
                    bridge_source=bridge_source_candidate,
                    pipeline_selection=pipeline_selection,
                    turn_id=state.turn_id,
                    intent=user_candidate.intent,
                    confidence=max(float(semantic_result.confidence or 0.0), float(user_candidate.confidence or 0.0), 0.8),
                )
                if bridge_candidates:
                    bridge_paths = [update.path for candidate in bridge_candidates for update in candidate.updates]
                    user_candidate = _augment_user_targets(user_candidate, bridge_paths)
                for key, meta in bridge_meta.items():
                    slot_debug[key] = merge_v2_meta(slot_debug.get(key), meta)
                if semantic_candidate.updates:
                    content_candidates.append(semantic_candidate)
                content_candidates.extend(bridge_candidates)
                if extracted_candidate.updates:
                    content_candidates.append(extracted_candidate)
                llm_used = True
                llm_raw = semantic_result.llm_raw
                llm_schema_errors = list(semantic_result.schema_errors)
                fallback_reason = None
                normalization_payload = {
                    "intent": user_candidate.intent.value,
                    "confidence": semantic_result.confidence,
                    "backend": "llm_semantic_frame",
                }
                debug["inference_backend"] = "llm_semantic_frame+runtime_semantic"
            else:
                llm_raw = semantic_result.llm_raw or slot_result.llm_raw
                llm_schema_errors = list(semantic_result.schema_errors)
                if semantic_result.fallback_reason:
                    llm_stage_failures.append(semantic_result.fallback_reason)
                    fallback_reason = semantic_result.fallback_reason
                else:
                    fallback_reason = slot_result.fallback_reason

    if not llm_used:
        _progress(progress_cb, "normalize", "Normalizing request", "Using fallback normalizer and deterministic extraction.")
        norm = normalize_user_turn(
            text,
            context_summary=context_summary,
            config_path=ollama_config_path,
            enable_llm=bool(normalize_input and llm_router),
        )
        user_candidate = CandidateUpdate(
            producer=Producer.USER_EXPLICIT,
            intent=norm["intent"],
            target_paths=list(norm["target_paths"]),
            updates=[],
            confidence=float(norm["confidence"]),
            rationale="fallback_user_explicit",
        )
        primary_candidate, debug = extract_candidates_from_normalized_text(
            norm["normalized_text"],
            raw_text=text,
            turn_id=state.turn_id,
            min_confidence=min_confidence,
            context_summary=context_summary,
            config_path=ollama_config_path,
            apply_autofix=apply_autofix,
        )
        bridge_source_candidate = primary_candidate
        if pipeline_selection.geometry == "v2":
            primary_candidate = _strip_geometry_updates(primary_candidate)
        if pipeline_selection.source == "v2":
            primary_candidate = _strip_source_updates(primary_candidate)
        normalized_text = norm["normalized_text"]
        primary_candidate = filter_candidate_by_explicit_targets(primary_candidate, list(user_candidate.target_paths))
        bridge_candidates, bridge_meta = _build_v2_bridge_candidates(
            base_config=draft.config,
            bridge_source=bridge_source_candidate,
            pipeline_selection=pipeline_selection,
            turn_id=state.turn_id,
            intent=user_candidate.intent,
            confidence=max(float(norm["confidence"] or 0.0), float(user_candidate.confidence or 0.0), 0.8),
        )
        if bridge_candidates:
            bridge_paths = [update.path for candidate in bridge_candidates for update in candidate.updates]
            user_candidate = _augment_user_targets(user_candidate, bridge_paths)
        for key, meta in bridge_meta.items():
            slot_debug[key] = merge_v2_meta(slot_debug.get(key), meta)
        content_candidates = [*bridge_candidates, primary_candidate]
        normalization_payload = {
            "intent": norm["intent"].value,
            "confidence": norm["confidence"],
            "backend": "fallback_normalizer",
        }

    if user_candidate is None:
        return {"error": "user intent unavailable"}
    dialogue_user_intent = user_candidate.intent.value
    _progress(progress_cb, "candidate_merge", "Merging candidates", "Combining user intent, semantic candidates, and recommendation paths.")

    if state.pending_overwrite:
        if user_candidate.intent == Intent.CONFIRM:
            staged_pending_overwrite = list(state.pending_overwrite)
            confirmed_candidate = build_candidate_from_pending_confirmation(staged_pending_overwrite, turn_id=state.turn_id)
            content_candidates = [confirmed_candidate]
            user_candidate = confirmed_candidate
            applying_pending_overwrite = True
            debug["inference_backend"] = f"{debug.get('inference_backend', 'orchestrated')}+confirmed_pending_overwrite"
            normalization_payload = {
                "intent": Intent.CONFIRM.value,
                "confidence": 1.0,
                "backend": "pending_overwrite_confirmation",
            }
        elif user_candidate.intent == Intent.REJECT:
            rejected_overwrite_preview = list(state.pending_overwrite)
            staged_pending_overwrite = []
            content_candidates = []
            debug["inference_backend"] = f"{debug.get('inference_backend', 'orchestrated')}+rejected_pending_overwrite"
            normalization_payload = {
                "intent": Intent.REJECT.value,
                "confidence": 1.0,
                "backend": "pending_overwrite_rejection",
            }
        else:
            staged_pending_overwrite = list(state.pending_overwrite)

    if user_candidate.target_paths:
        content_candidates = [
            filter_candidate_by_explicit_targets(candidate, list(user_candidate.target_paths))
            for candidate in content_candidates
        ]

    suppress_content_generation = bool(rejected_overwrite_preview and user_candidate.intent == Intent.REJECT)
    candidates: list[CandidateUpdate] = list(content_candidates)
    if not suppress_content_generation:
        forced_explicit = _build_forced_explicit_candidate(
            text=text,
            normalized_text=normalized_text,
            user_candidate=user_candidate,
            turn_id=state.turn_id,
        )
        if forced_explicit is not None:
            candidates.append(forced_explicit)

    merged_user_text = f"{text} ; {normalized_text}"
    has_explicit_physics = _has_explicit_physics_choice(merged_user_text, KNOWLEDGE["physics_lists"])
    allow_recommender = not has_explicit_physics
    reco_candidate = recommend_physics_list(
        text,
        normalized_text,
        context_summary=context_summary,
        allowed_lists=KNOWLEDGE["physics_lists"],
        turn_id=state.turn_id,
        config_path=ollama_config_path,
    )
    if not suppress_content_generation and allow_recommender and reco_candidate is not None:
        candidates.append(reco_candidate)

    pending_conflict_rejected: list[dict[str, Any]] = []
    if staged_pending_overwrite and not applying_pending_overwrite:
        pending_paths = [str(item.get("path", "")) for item in staged_pending_overwrite if str(item.get("path", ""))]
        replacement_targets = list(user_candidate.target_paths) if user_candidate.intent in {Intent.SET, Intent.MODIFY} else []
        filtered_candidates: list[CandidateUpdate] = []
        refreshed_pending: list[dict[str, Any]] = []
        for candidate in candidates:
            filtered_candidate, blocked_updates, replacement_updates = partition_candidate_by_pending_paths(
                candidate,
                pending_paths,
                replace_target_paths=replacement_targets,
            )
            for update in blocked_updates:
                pending_conflict_rejected.append(
                    {
                        "path": update.path,
                        "producer": candidate.producer.value,
                        "reason_code": E_PENDING_OVERWRITE_CONFLICT,
                        "detail": "update blocked: waiting for explicit overwrite confirmation on this field",
                    }
                )
            for update in replacement_updates:
                refreshed_pending.append(
                    build_pending_confirmation_item(
                        update,
                        draft=draft,
                        lang=lang,
                        producer=candidate.producer.value,
                    )
                )
            if filtered_candidate.updates:
                filtered_candidates.append(filtered_candidate)
        candidates = filtered_candidates
        if refreshed_pending:
            staged_pending_overwrite = merge_pending_confirmations(staged_pending_overwrite, refreshed_pending)

    confirmation_policy_result = evaluate_confirmation_requirements(
        draft,
        user_candidate,
        candidates,
        lang=lang,
        min_confidence=min_confidence if not applying_pending_overwrite else 0.0,
        enforce_no_implicit_overwrite=True,
        low_confidence_first=False,
        evaluate_pending=not applying_pending_overwrite,
    )
    candidates = confirmation_policy_result.filtered_candidates
    policy_rejected = confirmation_policy_result.rejected
    if not applying_pending_overwrite:
        if confirmation_policy_result.pending:
            staged_pending_overwrite = merge_pending_confirmations(staged_pending_overwrite, confirmation_policy_result.pending)
        candidates, staged_pending_overwrite = _stage_dependent_geometry_updates_for_pending(
            draft,
            candidates,
            staged_pending_overwrite,
            lang=lang,
        )
        candidates, staged_pending_overwrite = _stage_dependent_source_updates_for_pending(
            draft,
            candidates,
            staged_pending_overwrite,
            lang=lang,
        )
    accepted_updates, rejected_updates, applied_rules = arbitrate_candidates(draft, candidates)
    _progress(progress_cb, "arbitration", "Arbitrating updates", "Selecting updates that survive constraints and overwrite policy.")
    rejected_updates = pending_conflict_rejected + policy_rejected + rejected_updates
    if staged_pending_overwrite and not applying_pending_overwrite:
        applied_rules = [{"rule": "pending_overwrite_confirmation_required", "count": len(staged_pending_overwrite)}] + applied_rules
    committed_updates = list(accepted_updates)

    working = deep_copy(draft.config)
    _apply_updates(working, accepted_updates)
    # synchronize deterministic derived fields after primary updates
    post_default = build_semantic_sync_candidate(working, turn_id=state.turn_id, recent_updates=committed_updates)
    if post_default:
        _apply_updates(working, post_default.updates)
        committed_updates.extend(post_default.updates)
        applied_rules.append({"rule": "post_commit_semantic_sync", "count": len(post_default.updates)})

    report = validate_all(working)
    _progress(progress_cb, "validation", "Validating config", "Running completeness and rule validation on the working config.")
    hard_errors = [e for e in report.errors if e.get("code") != "E_REQUIRED_MISSING"]

    if hard_errors:
        # rollback
        if applying_pending_overwrite and staged_pending_overwrite:
            confirm_apply_failed = True
        rejected_updates.extend(
            {
                "path": upd.path,
                "producer": upd.producer.value,
                "reason_code": E_CANDIDATE_REJECTED_BY_GATE,
                "detail": "rollback due to hard gate errors",
            }
            for upd in accepted_updates
        )
        accepted_updates = []
        committed_updates = []
        working = deep_copy(draft.config)
        report = validate_all(working)
        if confirm_apply_failed:
            applied_rules = [{"rule": "pending_overwrite_confirm_failed_rollback", "count": len(staged_pending_overwrite)}] + applied_rules
    else:
        draft.config = working
        for upd in committed_updates:
            draft.field_sources[upd.path] = upd.producer.value
        lock_from_candidate(draft.constraint_ledger, user_candidate, draft.config, state.turn_id)

    # phase transition: local uses same report in v0.2 prototype
    phase_config = working if not hard_errors else state.config
    draft.phase = decide_phase_transition(draft.phase, report, report, config=phase_config)
    final_report = validate_layer_c_completeness(draft.config if not hard_errors else state.config)
    if rejected_overwrite_preview:
        semantic_missing_paths = []
    elif not hard_errors and _has_geometry_signal(
        debug=debug,
        committed_updates=committed_updates,
        user_candidate=user_candidate,
    ):
        semantic_missing_paths = _semantic_missing_from_debug(debug)
    semantic_missing_paths = merge_v2_missing_paths(
        list(semantic_missing_paths),
        slot_debug,
        include_spatial=True,
    )
    final_missing_paths = _dedupe_paths(final_report.missing_required_paths + list(semantic_missing_paths))
    pending_overwrite_required = bool(staged_pending_overwrite and (not applying_pending_overwrite or confirm_apply_failed))
    confirmation_payload = _build_confirmation_payload_with_staged_reference(
        staged_pending_overwrite if pending_overwrite_required else [],
        lang=lang,
        session_id=state.session_id,
        turn_id=state.turn_id,
        config=draft.config,
    )
    is_complete = bool(final_report.ok and not final_missing_paths and not pending_overwrite_required and not confirm_apply_failed)
    dialogue_pending_preview = list(staged_pending_overwrite) if pending_overwrite_required else []
    if is_complete:
        # lock subtree on complete
        # exact locks are already maintained by constraint ledger.
        pass

    if not hard_errors:
        commit_turn(state, draft)
        state.semantic_missing_paths = list(semantic_missing_paths)
        if applying_pending_overwrite:
            state.pending_overwrite = []
        elif rejected_overwrite_preview:
            state.pending_overwrite = []
        elif staged_pending_overwrite:
            state.pending_overwrite = staged_pending_overwrite
        else:
            state.pending_overwrite = []
        state.slot_memory = _refresh_slot_memory(
            getattr(state, "slot_memory", {}),
            config=state.config,
            slot_frame=slot_frame_memory_snapshot,
            allow_frame_overlay=not pending_overwrite_required,
        )
    elif staged_pending_overwrite:
        state.pending_overwrite = staged_pending_overwrite

    append_audit_entry(
        state=state,
        before_config=before_config,
        after_config=state.config,
        accepted_updates=committed_updates,
        rejected_updates=rejected_updates,
        validation_report=report,
        applied_rules=applied_rules,
    )

    state.open_questions, answered_this_turn = advance_question_state(
        previous_missing_paths=previous_missing_paths,
        current_missing_paths=final_missing_paths,
        open_questions=state.open_questions,
    )
    asked_fields = plan_questions(
        final_missing_paths,
        state.phase,
        open_questions=state.open_questions,
        last_asked_paths=state.last_asked_paths,
        question_attempts=state.question_attempts,
    )
    asked_fields = prioritize_v2_compile_questions(asked_fields, final_missing_paths, slot_debug)
    asked_fields = prioritize_spatial_questions(asked_fields, final_missing_paths, slot_debug)
    if asked_fields:
        for path in asked_fields:
            if path in final_missing_paths and path not in state.open_questions:
                state.open_questions.append(path)
    else:
        state.open_questions = []
    state.last_asked_paths = list(asked_fields)
    state.question_attempts = update_question_attempts(
        previous_attempts=state.question_attempts,
        current_missing_paths=final_missing_paths,
        answered_paths=answered_this_turn,
        asked_paths=asked_fields,
    )
    asked_fields_friendly = to_friendly_labels(asked_fields, lang)
    updated_paths = [upd.path for upd in committed_updates]
    available_explanations = collect_available_explanations(state.config, lang=lang)
    dialogue_decision = decide_dialogue_action(
        user_intent=dialogue_user_intent,
        is_complete=is_complete,
        asked_fields=asked_fields,
        missing_fields=final_missing_paths,
        updated_paths=updated_paths,
        answered_this_turn=answered_this_turn,
        pending_overwrite_preview=dialogue_pending_preview,
        rejected_overwrite_preview=rejected_overwrite_preview,
        available_explanations=available_explanations,
        last_dialogue_action=state.last_dialogue_action,
    )
    dialogue_trace = build_dialogue_trace(dialogue_decision)
    _progress(progress_cb, "dialogue", "Rendering response", "Building grounded reply and dialogue summary.")
    dialogue_summary, raw_dialogue_before_reply, dialogue_memory = sync_dialogue_state(
        state,
        decision=dialogue_decision,
        lang=lang,
        is_complete=is_complete,
    )
    question = render_dialogue_message(
        dialogue_decision,
        lang=lang,
        use_llm_question=llm_question,
        ollama_config=ollama_config_path,
        user_temperature=user_temperature,
        dialogue_summary=dialogue_summary,
        raw_dialogue=raw_dialogue_before_reply,
    )
    question = enforce_message_grounding(
        question,
        config=state.config,
        action=dialogue_decision.action.value,
        lang=lang,
    )
    state.last_dialogue_action = dialogue_decision.action.value
    state.history.append({"role": "assistant", "content": question})
    _progress(progress_cb, "finalize", "Finalizing turn", "Persisting turn state and assembling response payload.")
    raw_dialogue = build_raw_dialogue(state.history)
    candidate_patch_paths = _dedupe_paths([update.path for candidate in candidates for update in candidate.updates])
    applied_paths = _dedupe_paths([update.path for update in committed_updates])
    rejected_paths = _dedupe_paths([str(item.get("path", "")) for item in rejected_updates if str(item.get("path", ""))])
    trace_intent = intent_decision.intent
    if composite_intent.requires_staged_runtime_guard:
        trace_intent = "config_mutation"
    elif intent_decision.intent == "config_mutation" or applied_paths or pending_overwrite_required or rejected_overwrite_preview:
        trace_intent = "config_mutation"
    trace_safety = intent_decision.safety_class
    if trace_intent == "config_mutation":
        trace_safety = ActionSafetyClass.CONFIG_MUTATION
    trace_terminal = terminal_state_for_intent(
        trace_intent,
        mutation_applied=bool(applied_paths),
        waiting_confirmation=pending_overwrite_required,
        rejected=bool(rejected_overwrite_preview and not applied_paths),
        unsupported=bool(hard_errors and not applied_paths),
    )
    trace_nodes = graph_path_for_intent(
        trace_intent,
        mutation_applied=bool(applied_paths),
        waiting_confirmation=pending_overwrite_required,
        rejected=bool(rejected_overwrite_preview and not applied_paths),
        unsupported=bool(hard_errors and not applied_paths),
    )
    trace_blocked_tools: list[str] = []
    if trace_safety == ActionSafetyClass.CONFIG_MUTATION or composite_intent.requires_staged_runtime_guard:
        trace_blocked_tools = ["run_beam", "viewer_open"]
    elif trace_intent == "run_requested":
        trace_blocked_tools = ["run_beam"]
    elif trace_intent == "viewer_requested":
        trace_blocked_tools = ["viewer_open"]
    nlu_turn_trace = NluTurnTrace(
        turn_id_before=turn_id_before,
        turn_id_after=state.turn_id,
        intent=trace_intent,
        action_safety_class=trace_safety,
        terminal_state=trace_terminal,
        node_sequence=trace_nodes,
        prompt_profile_id=slot_debug.get("prompt_profile_id") or intent_decision.prompt_profile_id,
        llm_used=llm_used,
        fallback_reason=fallback_reason,
        candidate_patch_paths=candidate_patch_paths,
        confirmation_required=pending_overwrite_required,
        applied_paths=applied_paths,
        rejected_paths=rejected_paths,
        context_pack_hash=context_pack.context_pack_hash,
        patch_hash=stable_hash({"applied": applied_paths, "pending": staged_pending_overwrite}),
        confirmation_id=str(confirmation_payload.get("confirmation_id") or ""),
        confirmation_patch_hash=str(confirmation_payload.get("patch_hash") or ""),
        grounding_status="legacy_validated",
        interrupt_status="waiting_confirmation" if pending_overwrite_required else "none",
        idempotency_key=stable_hash({"session_id": state.session_id, "turn_id": state.turn_id, "patch": applied_paths}),
        runtime_payload_ready=is_complete,
        tool_calls_allowed=[],
        tool_calls_blocked=trace_blocked_tools,
        composite_intent=composite_intent.to_dict(),
        guarded_runtime_intent_pending=composite_intent.requires_staged_runtime_guard,
    )
    final_context_pack = context_pack
    if trace_intent != context_pack.intent:
        final_context_pack = build_context_pack(
            user_turn=text,
            intent_decision=intent_decision,
            config=before_config,
            staged_patch_summary={"pending_overwrite_count": len(state.pending_overwrite)}
            if state.pending_overwrite
            else None,
        )
        object.__setattr__(final_context_pack, "intent", trace_intent)
        object.__setattr__(
            final_context_pack,
            "context_pack_hash",
            stable_hash({"base": final_context_pack.to_dict(), "final_intent": trace_intent}),
        )
        nlu_turn_trace.context_pack_hash = final_context_pack.context_pack_hash
    internal_trace = {
        "agent": {
            "intent_decision": intent_decision.to_dict(),
            "composite_intent": composite_intent.to_dict(),
            "context_pack": final_context_pack.to_dict(),
            "initial_context_pack": context_pack.to_dict(),
            "nlu_turn_trace": nlu_turn_trace.to_dict(),
        },
        "nlu": {
            "llm_used": llm_used,
            "fallback_reason": fallback_reason,
            "inference_backend": debug.get("inference_backend", "orchestrated"),
            "normalization": normalization_payload,
            "pipelines": {"geometry": pipeline_selection.geometry, "source": pipeline_selection.source},
            "slot_debug": slot_debug,
            "geometry_compare": geometry_compare,
            "source_compare": source_compare,
            "interpreter_debug": interpreter_debug,
            "llm_stage_failures": list(llm_stage_failures),
            "llm_schema_errors": list(llm_schema_errors),
            "llm_raw_preview": str(llm_raw or "")[:2000],
        },
        "arbitration": {
            "candidate_count": len(candidates),
            "accepted_update_count": len(committed_updates),
            "rejected_update_count": len(rejected_updates),
            "pending_overwrite_count": len(staged_pending_overwrite),
            "applied_rules": list(applied_rules),
        },
        "validation": {
            "is_complete": is_complete,
            "pending_overwrite_required": pending_overwrite_required,
            "confirmation": confirmation_payload,
            "missing_fields": list(final_missing_paths),
            "schema_missing_fields": list(final_report.missing_required_paths),
            "semantic_missing_fields": list(semantic_missing_paths),
            "violations": list(report.errors),
            "warnings": list(report.warnings),
        },
        "dialogue": {
            "action": dialogue_decision.action.value,
            "asked_fields": list(asked_fields),
            "asked_fields_friendly": list(asked_fields_friendly),
            "answered_this_turn": list(answered_this_turn),
            "summary": dialogue_summary,
            "trace": dialogue_trace,
        },
    }

    return {
        "session_id": state.session_id,
        "phase": state.phase.value,
        "phase_title": phase_title(state.phase.value, lang),
        "dialogue_action": dialogue_decision.action.value,
        "dialogue_trace": dialogue_trace,
        "dialogue_summary": dialogue_summary,
        "dialogue_memory": dialogue_memory,
        "raw_dialogue": raw_dialogue,
        "is_complete": is_complete,
        "pending_overwrite_required": pending_overwrite_required,
        "confirmation": confirmation_payload,
        "assistant_message": question,
        "missing_fields": final_missing_paths,
        "answered_this_turn": answered_this_turn,
        "asked_fields": asked_fields,
        "asked_fields_friendly": asked_fields_friendly,
        "open_questions": state.open_questions,
        "question_attempts": state.question_attempts,
        "normalized_text": normalized_text,
        "normalization": normalization_payload,
        "pipelines": {"geometry": pipeline_selection.geometry, "source": pipeline_selection.source},
        "llm_used": llm_used,
        "fallback_reason": fallback_reason,
        "llm_raw": llm_raw,
        "llm_schema_errors": llm_schema_errors,
        "llm_stage_failures": llm_stage_failures,
        "slot_debug": slot_debug,
        "geometry_compare": geometry_compare,
        "source_compare": source_compare,
        "interpreter_debug": interpreter_debug,
        "temperatures": {
            "internal": internal_temperature,
            "user": user_temperature if llm_question else None,
        },
        "context_summary_used": context_summary,
        "config": state.config,
        "field_sources": state.field_sources,
        "applied_rules": applied_rules,
        "rejected_updates": rejected_updates,
        "violations": report.errors,
        "warnings": report.warnings,
        "graph_candidates": debug.get("graph_candidates", []),
        "graph_choice": debug.get("graph_choice", {}),
        "inference_backend": debug.get("inference_backend", "orchestrated"),
        "nlu_turn_trace": nlu_turn_trace.to_dict(),
        "context_pack": final_context_pack.to_dict(),
        "internal_trace": internal_trace,
        "history": state.history[-10:],
        "audit_size": len(state.audit_trail),
    }


def get_session_audit(session_id: str) -> list[dict]:
    with _SESSIONS_LOCK:
        state = SESSIONS.get(session_id)
    if not state:
        return []
    return state.audit_trail


def get_session_config_summary(session_id: str, *, lang: str = "zh") -> dict[str, Any]:
    sid = str(session_id or "").strip()
    with _SESSIONS_LOCK:
        state = SESSIONS.get(sid)
        if state is None:
            return {
                "ok": False,
                "error": "no_session_available",
                "session_id": sid,
            }
        config = deep_copy(state.config)
        phase = state.phase.value
        pending_overwrite = [dict(item) for item in state.pending_overwrite]
        confirmation_payload = _build_confirmation_payload_with_staged_reference(
            pending_overwrite,
            lang=lang,
            session_id=state.session_id,
            turn_id=state.turn_id,
            config=config,
        )
        semantic_missing = list(state.semantic_missing_paths)
        dialogue_summary = dict(state.dialogue_summary or {})
        last_asked_paths = list(state.last_asked_paths)

    report = validate_layer_c_completeness(config)
    missing_paths = _dedupe_paths(list(report.missing_required_paths) + semantic_missing)
    missing_labels = to_friendly_labels(missing_paths, lang)
    asked_labels = to_friendly_labels(last_asked_paths, lang)
    config_identity = {
        "geometry_structure": get_path(config, "geometry.structure"),
        "materials": get_path(config, "materials.selected_materials", []),
        "source_type": get_path(config, "source.type"),
        "particle": get_path(config, "source.particle"),
        "source_energy": get_path(config, "source.energy"),
        "physics_list": get_path(config, "physics.physics_list"),
        "output_format": get_path(config, "output.format"),
        "output_path": get_path(config, "output.path"),
    }
    if lang == "zh":
        message = (
            f"当前配置阶段：{phase_title(phase, lang)}。"
            f"几何：{config_identity['geometry_structure'] or '未设置'}；"
            f"材料：{', '.join(config_identity['materials']) if config_identity['materials'] else '未设置'}；"
            f"源：{config_identity['source_type'] or '未设置'} / {config_identity['particle'] or '未设置'}；"
            f"物理：{config_identity['physics_list'] or '未设置'}；"
            f"输出：{config_identity['output_format'] or '未设置'}。"
        )
        if missing_labels:
            message += f" 仍缺少：{', '.join(missing_labels[:4])}。"
    else:
        message = (
            f"Current phase: {phase_title(phase, lang)}. "
            f"Geometry: {config_identity['geometry_structure'] or 'not set'}; "
            f"materials: {', '.join(config_identity['materials']) if config_identity['materials'] else 'not set'}; "
            f"source: {config_identity['source_type'] or 'not set'} / {config_identity['particle'] or 'not set'}; "
            f"physics: {config_identity['physics_list'] or 'not set'}; "
            f"output: {config_identity['output_format'] or 'not set'}."
        )
        if missing_labels:
            message += f" Still missing: {', '.join(missing_labels[:4])}."
    return {
        "ok": True,
        "session_id": sid,
        "action_safety_class": "read_only",
        "phase": phase,
        "phase_title": phase_title(phase, lang),
        "is_complete": bool(report.ok and not missing_paths and not pending_overwrite),
        "message": message,
        "config_identity": config_identity,
        "missing_fields": missing_paths,
        "missing_fields_friendly": missing_labels,
        "last_asked_paths": last_asked_paths,
        "last_asked_fields_friendly": asked_labels,
        "pending_overwrite": pending_overwrite,
        "confirmation": confirmation_payload,
        "dialogue_summary": dialogue_summary,
        "config": config,
    }

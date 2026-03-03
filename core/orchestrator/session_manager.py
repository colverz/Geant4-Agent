from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from core.audit.audit_log import append_audit_entry
from core.config.defaults import build_strict_default_config
from core.config.field_registry import friendly_label
from core.config.phase_registry import phase_title
from core.dialogue.policy import decide_dialogue_action
from core.dialogue.renderer import render_dialogue_message
from core.dialogue.state import build_raw_dialogue, collect_available_explanations, sync_dialogue_state
from core.dialogue.types import build_dialogue_trace
from core.orchestrator.arbiter import arbitrate_candidates
from core.orchestrator.candidate_preprocess import (
    drop_updates_shadowed_by_anchor,
    filter_candidate_by_target_scopes,
)
from core.orchestrator.constraint_ledger import lock_from_candidate
from core.orchestrator.path_ops import deep_copy, get_path, set_path
from core.orchestrator.phase_machine import decide_phase_transition
from core.orchestrator.semantic_sync import build_semantic_sync_candidate
from core.orchestrator.turn_transaction import begin_turn, commit_turn
from core.orchestrator.types import CandidateUpdate, Intent, Phase, Producer, SessionState, UpdateOp
from core.slots.slot_mapper import slot_frame_to_candidates
from core.validation.error_codes import (
    E_CANDIDATE_REJECTED_BY_GATE,
    E_LLM_ROUTER_DISABLED,
    E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT,
)
from core.validation.validator_gate import (
    validate_all,
    validate_layer_c_completeness,
)
from nlu.bert.extractor import extract_candidates_from_normalized_text
from nlu.llm.slot_frame import build_llm_slot_frame
from nlu.llm.semantic_frame import build_llm_semantic_frame
from nlu.llm.normalizer import normalize_user_turn
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


def _load_knowledge() -> dict[str, list[str]]:
    physics_lists = json.loads((KNOWLEDGE_DIR / "physics_lists.json").read_text(encoding="utf-8")).get("items", [])
    return {"physics_lists": physics_lists}


KNOWLEDGE = _load_knowledge()


def default_config() -> dict[str, Any]:
    return build_strict_default_config()


def get_or_create_session(session_id: str | None) -> SessionState:
    sid = session_id or str(uuid.uuid4())
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
    return "; ".join(parts)

def _apply_updates(config: dict, updates: list) -> None:
    for upd in updates:
        if upd.op == "remove":
            # remove-path is intentionally omitted in v0.2 prototype to keep state trace stable.
            continue
        set_path(config, upd.path, upd.value)


def _path_explicitly_requested(user_candidate: CandidateUpdate, path: str) -> bool:
    for target in user_candidate.target_paths:
        if path == target:
            return True
        if path.startswith(target + "."):
            return True
    return False


def _is_unset_for_overwrite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _enforce_no_implicit_overwrite(
    state_like: Any,
    user_candidate: CandidateUpdate,
    candidates: list[CandidateUpdate],
) -> tuple[list[CandidateUpdate], list[dict]]:
    policy_rejected: list[dict] = []
    allow_overwrite = user_candidate.intent in {Intent.SET, Intent.MODIFY}
    filtered_candidates: list[CandidateUpdate] = []
    for candidate in candidates:
        kept = []
        for upd in candidate.updates:
            old = get_path(state_like.config, upd.path)
            if _is_unset_for_overwrite(old) or old == upd.value:
                kept.append(upd)
                continue
            explicitly_requested = _path_explicitly_requested(user_candidate, upd.path)
            if allow_overwrite and explicitly_requested:
                kept.append(upd)
                continue
            policy_rejected.append(
                {
                    "path": upd.path,
                    "producer": candidate.producer.value,
                    "reason_code": E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT,
                    "detail": "overwrite blocked: user did not explicitly request this field update",
                }
            )
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
                rationale=f"{candidate.rationale}_overwrite_guarded",
            )
        )
    return filtered_candidates, policy_rejected


def _extract_pending_overwrites(
    state_like: Any,
    user_candidate: CandidateUpdate,
    candidates: list[CandidateUpdate],
    *,
    lang: str,
) -> tuple[list[CandidateUpdate], list[dict[str, Any]]]:
    if user_candidate.intent not in {Intent.SET, Intent.MODIFY}:
        return candidates, []
    pending: list[dict[str, Any]] = []
    filtered_candidates: list[CandidateUpdate] = []
    for candidate in candidates:
        kept: list[UpdateOp] = []
        for upd in candidate.updates:
            old = get_path(state_like.config, upd.path)
            if _is_unset_for_overwrite(old) or old == upd.value:
                kept.append(upd)
                continue
            if _path_explicitly_requested(user_candidate, upd.path):
                pending.append(
                    {
                        "path": upd.path,
                        "field": friendly_label(upd.path, lang),
                        "old": old,
                        "new": upd.value,
                        "producer": candidate.producer.value,
                    }
                )
                continue
            kept.append(upd)
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
                rationale=f"{candidate.rationale}_overwrite_staged",
            )
        )
    return filtered_candidates, pending


def _candidate_from_pending_overwrite(items: list[dict[str, Any]], *, turn_id: int) -> CandidateUpdate:
    updates = [
        UpdateOp(
            path=str(item["path"]),
            op="set",
            value=item.get("new"),
            producer=Producer.USER_EXPLICIT,
            confidence=1.0,
            turn_id=turn_id,
        )
        for item in items
    ]
    return CandidateUpdate(
        producer=Producer.USER_EXPLICIT,
        intent=Intent.MODIFY,
        target_paths=sorted({str(item["path"]) for item in items}),
        updates=updates,
        confidence=1.0,
        rationale="confirmed_pending_overwrite",
    )
def process_turn(payload: dict, *, ollama_config_path: str, min_confidence: float = 0.6, lang: str = "zh") -> dict:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"error": "missing text"}
    state = get_or_create_session(payload.get("session_id"))
    previous_missing_paths = validate_layer_c_completeness(state.config).missing_required_paths
    state.turn_id += 1
    state.history.append({"role": "user", "content": text})

    before_config = deep_copy(state.config)
    draft = begin_turn(state)
    context_summary = _build_context_summary(state)
    llm_router = bool(payload.get("llm_router", True))
    llm_question = bool(payload.get("llm_question", True))
    internal_temperature = 0.0
    user_temperature = float(payload.get("user_temperature", 1.0))
    normalize_input = bool(payload.get("normalize_input", True))
    enable_llm_first = bool(llm_router and normalize_input)
    llm_used = False
    fallback_reason = E_LLM_ROUTER_DISABLED if (not llm_router and normalize_input) else "E_LLM_DISABLED"
    llm_raw = ""
    llm_schema_errors: list[str] = []
    llm_stage_failures: list[str] = []
    slot_debug: dict[str, Any] = {}
    debug: dict[str, Any] = {"graph_candidates": []}
    normalized_text = text
    content_candidates: list[CandidateUpdate] = []
    user_candidate: CandidateUpdate | None = None
    applying_pending_overwrite = False
    confirmation_gate_active = False
    staged_pending_overwrite: list[dict[str, Any]] = []

    if enable_llm_first:
        slot_result = build_llm_slot_frame(
            text,
            context_summary=context_summary,
            config_path=ollama_config_path,
        )
        if slot_result.ok and slot_result.frame:
            slot_candidate, user_candidate = slot_frame_to_candidates(slot_result.frame, turn_id=state.turn_id)
            normalized_text = slot_result.normalized_text or text
            slot_debug = dict(slot_result.stage_trace or {})
            slot_debug.setdefault("final_status", "ok")
            extracted_candidate, debug = extract_candidates_from_normalized_text(
                normalized_text,
                raw_text=text,
                turn_id=state.turn_id,
                min_confidence=min_confidence,
                context_summary=context_summary,
                config_path=ollama_config_path,
            )
            if user_candidate.target_paths:
                extracted_candidate = filter_candidate_by_target_scopes(extracted_candidate, list(user_candidate.target_paths))
            extracted_candidate = drop_updates_shadowed_by_anchor(extracted_candidate, slot_candidate)
            if slot_candidate is not None and slot_candidate.updates:
                content_candidates.append(slot_candidate)
            if extracted_candidate.updates:
                content_candidates.append(extracted_candidate)
            llm_used = True
            llm_raw = slot_result.llm_raw
            llm_schema_errors = list(slot_result.schema_errors)
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
            semantic_result = build_llm_semantic_frame(
                text,
                context_summary=context_summary,
                config_path=ollama_config_path,
                turn_id=state.turn_id,
            )
            if semantic_result.ok and semantic_result.candidate and semantic_result.user_candidate:
                user_candidate = semantic_result.user_candidate
                normalized_text = semantic_result.normalized_text or text
                extracted_candidate, debug = extract_candidates_from_normalized_text(
                    normalized_text,
                    raw_text=text,
                    turn_id=state.turn_id,
                    min_confidence=min_confidence,
                    context_summary=context_summary,
                    config_path=ollama_config_path,
                )
                if user_candidate.target_paths:
                    extracted_candidate = filter_candidate_by_target_scopes(extracted_candidate, list(user_candidate.target_paths))
                extracted_candidate = drop_updates_shadowed_by_anchor(extracted_candidate, semantic_result.candidate)
                if semantic_result.candidate.updates:
                    content_candidates.append(semantic_result.candidate)
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
        norm = normalize_user_turn(text, context_summary=context_summary, config_path=ollama_config_path)
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
        )
        normalized_text = norm["normalized_text"]
        content_candidates = [primary_candidate]
        normalization_payload = {
            "intent": norm["intent"].value,
            "confidence": norm["confidence"],
            "backend": "fallback_normalizer",
        }

    if user_candidate is None:
        return {"error": "user intent unavailable"}
    dialogue_user_intent = user_candidate.intent.value

    if state.pending_overwrite:
        if user_candidate.intent == Intent.CONFIRM:
            confirmed_candidate = _candidate_from_pending_overwrite(state.pending_overwrite, turn_id=state.turn_id)
            content_candidates = [confirmed_candidate]
            user_candidate = confirmed_candidate
            applying_pending_overwrite = True
            debug["inference_backend"] = f"{debug.get('inference_backend', 'orchestrated')}+confirmed_pending_overwrite"
            normalization_payload = {
                "intent": Intent.CONFIRM.value,
                "confidence": 1.0,
                "backend": "pending_overwrite_confirmation",
            }
        else:
            confirmation_gate_active = True
            staged_pending_overwrite = list(state.pending_overwrite)
            content_candidates = []

    if not confirmation_gate_active and get_path(draft.config, "geometry.structure") is not None:
        content_candidates = [
            filter_candidate_by_target_scopes(candidate, list(user_candidate.target_paths))
            for candidate in content_candidates
        ]

    if confirmation_gate_active:
        accepted_updates = []
        rejected_updates = []
        applied_rules = [{"rule": "pending_overwrite_confirmation_required", "count": len(staged_pending_overwrite)}]
        committed_updates = []
        working = deep_copy(draft.config)
        report = validate_all(working)
        hard_errors = []
    else:
        candidates: list[CandidateUpdate] = list(content_candidates)
        reco_candidate = recommend_physics_list(
            text,
            normalized_text,
            context_summary=context_summary,
            allowed_lists=KNOWLEDGE["physics_lists"],
            turn_id=state.turn_id,
            config_path=ollama_config_path,
        )
        if reco_candidate is not None:
            candidates.append(reco_candidate)

        candidates, policy_rejected = _enforce_no_implicit_overwrite(draft, user_candidate, candidates)
        if not applying_pending_overwrite:
            candidates, staged_pending_overwrite = _extract_pending_overwrites(
                draft,
                user_candidate,
                candidates,
                lang=lang,
            )
        accepted_updates, rejected_updates, applied_rules = arbitrate_candidates(draft, candidates)
        rejected_updates = policy_rejected + rejected_updates
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
        hard_errors = [e for e in report.errors if e.get("code") != "E_REQUIRED_MISSING"]

        if hard_errors:
            # rollback
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
        else:
            draft.config = working
            for upd in committed_updates:
                draft.field_sources[upd.path] = upd.producer.value
            lock_from_candidate(draft.constraint_ledger, user_candidate, draft.config, state.turn_id)

    # phase transition: local uses same report in v0.2 prototype
    phase_config = working if not hard_errors else state.config
    draft.phase = decide_phase_transition(draft.phase, report, report, config=phase_config)
    final_report = validate_layer_c_completeness(draft.config if not hard_errors else state.config)
    is_complete = final_report.ok
    if is_complete:
        # lock subtree on complete
        # exact locks are already maintained by constraint ledger.
        pass

    if not hard_errors and not confirmation_gate_active:
        commit_turn(state, draft)
        if applying_pending_overwrite:
            state.pending_overwrite = []
        elif staged_pending_overwrite:
            state.pending_overwrite = staged_pending_overwrite
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

    if staged_pending_overwrite:
        answered_this_turn = []
        asked_fields = []
    else:
        state.open_questions, answered_this_turn = advance_question_state(
            previous_missing_paths=previous_missing_paths,
            current_missing_paths=final_report.missing_required_paths,
            open_questions=state.open_questions,
        )
        asked_fields = plan_questions(
            final_report.missing_required_paths,
            state.phase,
            open_questions=state.open_questions,
            last_asked_paths=state.last_asked_paths,
            question_attempts=state.question_attempts,
        )
        if asked_fields:
            for path in asked_fields:
                if path in final_report.missing_required_paths and path not in state.open_questions:
                    state.open_questions.append(path)
        else:
            state.open_questions = []
        state.last_asked_paths = list(asked_fields)
        state.question_attempts = update_question_attempts(
            previous_attempts=state.question_attempts,
            current_missing_paths=final_report.missing_required_paths,
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
        missing_fields=final_report.missing_required_paths,
        updated_paths=updated_paths,
        answered_this_turn=answered_this_turn,
        pending_overwrite_preview=staged_pending_overwrite,
        available_explanations=available_explanations,
        last_dialogue_action=state.last_dialogue_action,
    )
    dialogue_trace = build_dialogue_trace(dialogue_decision)
    dialogue_summary, _, dialogue_memory = sync_dialogue_state(
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
    )
    state.last_dialogue_action = dialogue_decision.action.value
    state.history.append({"role": "assistant", "content": question})
    raw_dialogue = build_raw_dialogue(state.history)

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
        "assistant_message": question,
        "missing_fields": final_report.missing_required_paths,
        "answered_this_turn": answered_this_turn,
        "asked_fields": asked_fields,
        "asked_fields_friendly": asked_fields_friendly,
        "open_questions": state.open_questions,
        "question_attempts": state.question_attempts,
        "normalized_text": normalized_text,
        "normalization": normalization_payload,
        "llm_used": llm_used,
        "fallback_reason": fallback_reason,
        "llm_raw": llm_raw,
        "llm_schema_errors": llm_schema_errors,
        "llm_stage_failures": llm_stage_failures,
        "slot_debug": slot_debug,
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
        "inference_backend": debug.get("inference_backend", "orchestrated"),
        "history": state.history[-10:],
        "audit_size": len(state.audit_trail),
    }


def get_session_audit(session_id: str) -> list[dict]:
    state = SESSIONS.get(session_id)
    if not state:
        return []
    return state.audit_trail

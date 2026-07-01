from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.geant4.runtime_discovery import discover_local_geant4_runtime

from .contracts import V3AgentResult, V3AgentState, V3Answer, V3TurnInput
from .controller import AgentController
from .context import (
    build_v3_context_pack,
    build_v3_state_summary,
    resolve_open_questions_if_answered,
    update_v3_workflow_state,
)
from .dialogue_composer import compose_v3_dialogue
from .patches import apply_patches_to_state, build_patches_from_config_overrides, build_patches_from_requested_changes
from .pending_action import V3PendingAction, V3PendingActionManager
from .proposal_critic import review_v3_proposal
from .reasoners import BasicGeant4Reasoner, LLMGeant4Reasoner
from .response_naturalizer import V3ResponseNaturalizer
from .response_quality import evaluate_v3_response_quality
from .runtime_policy import V3RuntimePolicy, runtime_policy_from_turn, set_runtime_policy
from .session import V3SessionStore
from .tools import (
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
    build_default_geant4_tool_registry,
)
from .turn_understanding import LLMTurnUnderstandingProvider, build_v3_turn_understanding, normalize_confirmation_event


V3_AGENT_TURN_SCHEMA_VERSION = "geant4_agent_v3_turn.v1"
_DEFAULT_SESSIONS_DIR = Path("runtime_artifacts/sessions")


def build_v3_agent_controller(*, llm_config: str = "", lang: str = "zh-CN") -> AgentController:
    if llm_config and llm_config.strip():
        reasoner = LLMGeant4Reasoner(llm_config_path=llm_config, lang=lang)
    else:
        reasoner = BasicGeant4Reasoner()
    return AgentController(
        reasoner=reasoner,
        tools=build_default_geant4_tool_registry(),
        constraint_reviewer=review_v3_proposal,
    )


@dataclass(slots=True)
class V3AgentTurnService:
    states: dict[str, V3AgentState] = field(default_factory=dict)
    sessions_dir: Path = _DEFAULT_SESSIONS_DIR
    store: V3SessionStore | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.store = V3SessionStore(self.sessions_dir)

    def _state_path(self, session_id: str) -> Path:
        assert self.store is not None
        return self.store.state_path(session_id)

    def _save_state(self, state: V3AgentState) -> None:
        assert self.store is not None
        self.store.save(state, last_turn_id=str(state.metadata.get("last_turn_id") or ""))

    def _load_state(self, session_id: str) -> V3AgentState | None:
        assert self.store is not None
        return self.store.load(session_id)

    def reset(self, session_id: str | None = None) -> None:
        assert self.store is not None
        if session_id:
            self.states.pop(session_id, None)
            self.store.reset(session_id)
            return
        self.states.clear()
        self.store.reset()

    def run_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        turn, runtime_discovery = build_turn_input(payload)
        previous_state = self.states.get(turn.session_id)
        if previous_state is None:
            previous_state = self._load_state(turn.session_id)

        understanding = _build_mainline_turn_understanding(turn, previous_state)
        _apply_understanding_intent_to_turn(turn, understanding)
        patch_result = _patch_result_from_understanding(understanding)

        if (
            not patch_result.patches
            and not patch_result.errors
            and previous_state is not None
            and not turn.metadata.get("config_overrides")
            and understanding.source in {"fallback", "fallback_after_llm_error", "llm_uncertain"}
        ):
            inferred_overrides = _infer_config_overrides_from_text(turn.user_text)
            if inferred_overrides:
                turn.metadata["config_overrides"] = inferred_overrides
                turn.metadata["accept_defaults"] = True
                if "run_events" in inferred_overrides:
                    turn.metadata["events"] = _positive_int(
                        inferred_overrides["run_events"],
                        default=_positive_int(turn.metadata.get("events"), default=1000),
                    )
                understanding = build_v3_turn_understanding(turn, previous_state)
                patch_result = _patch_result_from_understanding(understanding)

        if not patch_result.patches and not patch_result.errors:
            patch_result = build_patches_from_config_overrides(
                turn.metadata.get("config_overrides"),
                evidence="user_turn" if turn.metadata.get("config_overrides") else "none",
            )
        if patch_result.patches or patch_result.errors:
            turn.metadata["state_patch"] = patch_result.to_dict()
            turn.metadata["config_overrides"] = patch_result.config_overrides
            if patch_result.config_overrides:
                turn.metadata["accept_defaults"] = True
                if "run_events" in patch_result.config_overrides:
                    turn.metadata["events"] = _positive_int(
                        patch_result.config_overrides["run_events"],
                        default=_positive_int(turn.metadata.get("events"), default=1000),
                    )
        turn.metadata["turn_understanding"] = understanding.to_dict()
        pending_action = V3PendingActionManager.get(previous_state)
        cancelled_pending_action: dict[str, Any] | None = None
        if previous_state is not None and patch_result.config_overrides:
            patch_apply_result = apply_patches_to_state(previous_state, patch_result)
            turn.metadata["state_patch_apply"] = patch_apply_result.to_dict()
        has_explicit_confirmation_event = bool(turn.metadata.get("confirmation_event"))
        event_matches_pending = V3PendingActionManager.confirmation_event_matches(
            turn.metadata.get("confirmation_event"),
            pending_action,
        )
        understanding_confirms_pending = (
            understanding.confirmation == "confirmed"
            and pending_action is not None
            and understanding.referenced_state == "pending_action"
            and event_matches_pending
        )
        is_confirmed = understanding_confirms_pending or (
            not has_explicit_confirmation_event and _looks_like_confirmation(turn.user_text)
        )
        is_rejected = (understanding.confirmation == "rejected" and event_matches_pending) or (
            not has_explicit_confirmation_event and _looks_like_cancellation(turn.user_text)
        )
        if is_confirmed and understanding.confirmation != "confirmed":
            turn.metadata["turn_understanding"] = {
                **turn.metadata["turn_understanding"],
                "dialogue_act": "confirm",
                "confirmation": "confirmed",
                "referenced_state": "pending_action" if pending_action else turn.metadata["turn_understanding"].get("referenced_state", "none"),
                "confidence": 0.7,
                "reason": "free-text confirmation compatibility fallback",
                "source": "text_fallback",
            }
        if is_rejected and understanding.confirmation != "rejected":
            turn.metadata["turn_understanding"] = {
                **turn.metadata["turn_understanding"],
                "dialogue_act": "reject",
                "confirmation": "rejected",
                "referenced_state": "pending_action" if pending_action else turn.metadata["turn_understanding"].get("referenced_state", "none"),
                "confidence": 0.7,
                "reason": "free-text cancellation compatibility fallback",
                "source": "text_fallback",
            }
        if pending_action and is_confirmed:
            confirmed_by = "confirmation_event" if has_explicit_confirmation_event else "user_text"
            V3PendingActionManager.apply_to_turn(turn, pending_action, confirmed_by=confirmed_by)
        if pending_action and is_rejected:
            cancelled_pending_action = pending_action.to_dict()
            V3PendingActionManager.clear(previous_state)
            turn.metadata["suppress_run"] = True
        if pending_action is None and (
            is_confirmed or (is_rejected and _looks_like_standalone_cancellation(turn.user_text))
        ):
            result = _no_pending_confirmation_result(turn, previous_state, confirmed=is_confirmed)
        else:
            # Use BasicGeant4Reasoner for confirm/cancel; LLM for understanding/analysis
            use_basic = bool(pending_action and (is_confirmed or is_rejected))
            controller = build_v3_agent_controller(
                llm_config="" if use_basic else turn.metadata.get("llm_config_path", ""),
                lang=turn.locale,
            )
            # Detect sweep: user-requested or LLM auto-suggested
            sweep_spec = _detect_sweep_request(turn, previous_state)
            if sweep_spec and previous_state is not None:
                previous_state.metadata["suggested_next_actions"] = [_suggestion_from_sweep(sweep_spec)]
                turn.metadata["suppress_run"] = True
            result = controller.run(turn, state=previous_state)
        if cancelled_pending_action is not None:
            result.terminated_reason = "final_answer"
        if isinstance(turn.metadata.get("execution_authorization"), dict):
            result.state.metadata["last_execution_authorization"] = turn.metadata["execution_authorization"]
        result.state.metadata["turn_understanding"] = turn.metadata["turn_understanding"]
        if isinstance(turn.metadata.get("state_patch"), dict):
            result.state.metadata["last_state_patch"] = turn.metadata["state_patch"]
        if isinstance(turn.metadata.get("state_patch_apply"), dict):
            result.state.metadata["last_state_patch_apply"] = turn.metadata["state_patch_apply"]
        result.state.metadata["last_turn_id"] = str(uuid.uuid4())
        pending_action = _extract_pending_action(result)
        if pending_action:
            V3PendingActionManager.store(result.state, pending_action)
        elif result.terminated_reason in {"observed", "final_answer", "blocked"}:
            V3PendingActionManager.clear(result.state)
        resolve_open_questions_if_answered(result.state, config_overrides=turn.metadata.get("config_overrides"))
        _persist_suggested_next_actions(turn, result.state)
        update_v3_workflow_state(result.state, last_user_turn=turn.user_text)
        self.states[turn.session_id] = result.state
        self._save_state(result.state)
        response = serialize_turn_result(result)
        response["pending_action"] = V3PendingActionManager.get_dict(result.state)
        if cancelled_pending_action is not None:
            response["cancelled_pending_action"] = cancelled_pending_action
        # Pass LLM suggestions to UI (from turn metadata, stored by reasoner)
        llm_suggestions = turn.metadata.get("suggestions")
        if isinstance(llm_suggestions, list) and llm_suggestions:
            response["suggestions"] = llm_suggestions
            result.state.metadata["suggestions"] = llm_suggestions
        elif isinstance(result.state.metadata.get("suggested_next_actions"), list):
            response["suggestions"] = result.state.metadata["suggested_next_actions"]
        if runtime_discovery is not None:
            response["runtime_discovery"] = runtime_discovery.to_dict()
        response["context"] = build_v3_context_pack(result.state, last_user_turn=turn.user_text)
        response["summary"] = build_v3_state_summary(result.state)
        dialogue = compose_v3_dialogue(response, locale=turn.locale).to_dict()
        response["dialogue"] = dialogue
        response["display_message"] = dialogue["display_message"]
        response["raw_message"] = dialogue["raw_message"]
        response["dialogue_act"] = dialogue["dialogue_act"]
        response["evidence_used"] = dialogue["evidence_used"]
        response["answer_parts"] = dialogue.get("answer_parts") or []
        if isinstance(response.get("answer"), dict):
            response["answer"]["display_message"] = dialogue["display_message"]
            response["answer"]["raw_message"] = dialogue["raw_message"]
            response["answer"]["answer_parts"] = response["answer_parts"]
        if bool(turn.metadata.get("llm_naturalize_enabled")):
            naturalization = V3ResponseNaturalizer(str(turn.metadata.get("llm_config_path") or "")).naturalize(
                response,
                locale=turn.locale,
            )
            response["naturalization"] = naturalization.to_dict()
            result.state.metadata["last_response_naturalization"] = response["naturalization"]
            if naturalization.ok and naturalization.display_message:
                response["display_message"] = naturalization.display_message
                response["dialogue"]["display_message"] = naturalization.display_message
                if isinstance(response.get("answer"), dict):
                    response["answer"]["display_message"] = naturalization.display_message
        if isinstance(response.get("state"), dict) and isinstance(response["state"].get("metadata"), dict) and response.get("naturalization"):
            response["state"]["metadata"]["last_response_naturalization"] = response["naturalization"]
        response["dialogue_quality"] = evaluate_v3_response_quality(response)
        result.state.metadata["last_dialogue_quality"] = response["dialogue_quality"]
        if isinstance(response.get("state"), dict) and isinstance(response["state"].get("metadata"), dict):
            response["state"]["metadata"]["last_dialogue_quality"] = response["dialogue_quality"]
        self._save_state(result.state)
        return response

    def get_state_payload(self, session_id: str, *, lang: str = "zh") -> tuple[int, dict[str, Any]]:
        state = self.states.get(session_id) or self._load_state(session_id)
        if state is None:
            return 404, {"ok": False, "error": "session_not_found", "session_id": session_id}
        self.states[session_id] = state
        return 200, {
            "ok": True,
            "session_id": session_id,
            "lang": lang,
            "state": state.to_dict(),
            "summary": build_v3_state_summary(state),
            "context": build_v3_context_pack(state),
        }


def build_turn_input(payload: dict[str, Any]) -> tuple[V3TurnInput, Any | None]:
    session_id = str(payload.get("session_id") or "").strip() or _new_session_id()
    text = str(payload.get("text") or payload.get("user_text") or "").strip()
    locale = str(payload.get("locale") or payload.get("lang") or "zh-CN").strip() or "zh-CN"
    events = _positive_int(payload.get("events"), default=1000)
    auto_discover_runtime = bool(payload.get("auto_discover_runtime"))
    runtime_discovery = discover_local_geant4_runtime() if auto_discover_runtime else None
    runtime_env = runtime_discovery.env() if runtime_discovery is not None and runtime_discovery.found else {}
    runtime_policy = V3RuntimePolicy.from_payload(payload, runtime_env=runtime_env)
    accept_defaults = bool(payload.get("accept_defaults"))
    config_overrides: dict[str, Any] = dict(payload.get("config_overrides") or {})  # LLMGeant4Reasoner injects these
    if "run_events" in config_overrides:
        events = _positive_int(config_overrides["run_events"], default=events)
    metadata = {
        "accept_defaults": accept_defaults,
        "events": events,
        "run": bool(payload.get("run")),
        "run_confirmed": False,
        "llm_design_enabled": bool(payload.get("llm_design_enabled")),
        "llm_result_enabled": bool(payload.get("llm_result_enabled")),
        "llm_naturalize_enabled": bool(payload.get("llm_naturalize_enabled")),
        "llm_config_path": str(payload.get("llm_config_path") or "").strip(),
        "config_overrides": config_overrides,
        "confirmation_event": normalize_confirmation_event(payload.get("confirmation_event")),
    }
    set_runtime_policy(metadata, runtime_policy)
    if metadata["config_overrides"]:
        metadata["accept_defaults"] = True
    if payload.get("run_id"):
        metadata["run_id"] = str(payload["run_id"])
    return (
        V3TurnInput(
            session_id=session_id,
            user_text=text,
            locale=locale,
            attachments=list(payload.get("attachments") or []),
            metadata=metadata,
        ),
        runtime_discovery,
    )


def _build_mainline_turn_understanding(turn: V3TurnInput, state: V3AgentState | None) -> Any:
    llm_config = str(turn.metadata.get("llm_config_path") or "").strip()
    if llm_config and not turn.metadata.get("config_overrides"):
        return LLMTurnUnderstandingProvider(llm_config).understand(turn, state)
    return build_v3_turn_understanding(turn, state)


def _apply_understanding_intent_to_turn(turn: V3TurnInput, understanding: Any) -> None:
    if getattr(understanding, "risk_intent", "") == "run_requested" or getattr(understanding, "dialogue_act", "") == "request_run":
        turn.metadata["run"] = True
    if getattr(understanding, "dialogue_act", "") == "reject":
        turn.metadata["suppress_run"] = True


def _patch_result_from_understanding(understanding: Any) -> Any:
    changes = getattr(understanding, "requested_changes", None)
    if changes:
        return build_patches_from_requested_changes(
            [change.to_dict() if hasattr(change, "to_dict") else change for change in changes],
            evidence=f"{getattr(understanding, 'source', 'turn')}_turn_understanding",
        )
    return build_patches_from_config_overrides({}, evidence="none")


def _new_session_id() -> str:
    return f"v3-{uuid.uuid4().hex}"


def serialize_turn_result(result: V3AgentResult) -> dict[str, Any]:
    return {
        "schema_version": V3_AGENT_TURN_SCHEMA_VERSION,
        "ok": result.terminated_reason in {"final_answer", "waiting_user", "waiting_confirmation", "observed", "blocked"},
        "terminated_reason": result.terminated_reason,
        "answer": result.answer.to_dict(),
        "state": result.state.to_dict(),
        "trace": result.trace,
        "observations": [item.to_dict() for item in result.observations],
    }


def _extract_pending_action(result: V3AgentResult) -> V3PendingAction | None:
    return V3PendingAction.from_commit_gate_result(result)


def _persist_suggested_next_actions(turn: V3TurnInput, state: V3AgentState) -> None:
    suggestions: list[dict[str, Any]] = []
    sweep = turn.metadata.get("auto_sweep")
    if isinstance(sweep, dict):
        suggestions.append(_suggestion_from_sweep(sweep))
    opt = turn.metadata.get("optimization")
    if isinstance(opt, dict) and opt.get("next_value") is not None:
        parameter = str(opt.get("parameter") or "target_thickness_mm")
        next_value = opt.get("next_value")
        suggestions.append({
            "text": f"Try {parameter}={next_value}",
            "prefill": f"change {parameter} to {next_value} and run again",
            "kind": "optimization",
            "proposal": dict(opt),
        })
    raw = turn.metadata.get("suggestions")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("label") or item.get("prefill") or "").strip()
                prefill = str(item.get("prefill") or item.get("text") or "").strip()
                if text and prefill:
                    suggestions.append({"text": text, "prefill": prefill})
            else:
                text = str(item or "").strip()
                if text:
                    suggestions.append({"text": text, "prefill": text})
    if suggestions:
        state.metadata["suggested_next_actions"] = _dedupe_suggestions(suggestions)


def _suggestion_from_sweep(sweep_spec: dict[str, Any]) -> dict[str, Any]:
    parameter = str(sweep_spec.get("parameter") or "source_energy_mev")
    values = sweep_spec.get("values") if isinstance(sweep_spec.get("values"), list) else []
    label = str(sweep_spec.get("label") or parameter)
    values_text = " ".join(str(value) for value in values)
    unit = " MeV" if "energy" in parameter else ""
    return {
        "text": f"Sweep {label}",
        "prefill": f"run sweep {values_text}{unit}".strip(),
        "kind": "sweep",
        "proposal": dict(sweep_spec),
    }


def _dedupe_suggestions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("prefill") or item.get("text") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= 8:
            break
    return out


def _prepare_state_for_modified_draft(state: V3AgentState) -> None:
    state.metadata.pop("pending_action", None)
    state.metadata.pop("suggested_next_actions", None)
    stale_sources = {GEANT4_PAYLOAD_BUILDER_TOOL, GEANT4_RUNTIME_PREFLIGHT_TOOL, GEANT4_RUNTIME_TOOL, "commit_gate"}
    state.observations = [observation for observation in state.observations if observation.source not in stale_sources]
    for artifact_id in ("geant4_runtime_payload_draft", "draft_geant4_runtime_payload", "preflight_geant4_runtime"):
        state.artifacts.pop(artifact_id, None)


def _no_pending_confirmation_result(
    turn: V3TurnInput,
    state: V3AgentState | None,
    *,
    confirmed: bool,
) -> V3AgentResult:
    turn.metadata["run"] = False
    turn.metadata["run_confirmed"] = False
    turn.metadata["suppress_run"] = True
    agent_state = state or V3AgentState(session_id=turn.session_id, goal=turn.user_text)
    if not agent_state.goal:
        agent_state.goal = turn.user_text
    if confirmed:
        message = (
            "当前没有等待确认的运行行动。请先告诉我要运行哪个方案，或点击运行按钮生成待确认行动。"
            if not _is_en_locale(turn.locale)
            else "There is no pending run action to confirm. Ask me to run a specific draft first, or use the run action to create a confirmation step."
        )
    else:
        message = (
            "当前没有等待取消的运行行动。我不会启动 Geant4。"
            if not _is_en_locale(turn.locale)
            else "There is no pending run action to cancel. I will not start Geant4."
        )
    return V3AgentResult(
        answer=V3Answer(
            message=message,
            understanding=agent_state.goal,
            evidence=[
                {
                    "source": "pending_action",
                    "status": "missing",
                    "reason": "no_pending_action_to_confirm" if confirmed else "no_pending_action_to_cancel",
                }
            ],
            next_options=(
                ["运行当前方案", "查看当前配置", "修改方案"]
                if not _is_en_locale(turn.locale)
                else ["Run current draft", "View current configuration", "Modify design"]
            ),
            runtime_status="not_started",
        ),
        state=agent_state,
        trace=[],
        observations=[],
        terminated_reason="final_answer",
    )


def _is_en_locale(locale: str) -> bool:
    return str(locale).lower().startswith("en")


def _looks_like_cancellation(text: str) -> bool:
    if any(token in text for token in ("取消", "先不", "不要运行", "不运行", "别跑", "暂停")):
        return True
    normalized = text.strip().lower()
    if not normalized:
        return False
    english_tokens = ("cancel", "stop", "do not run", "don't run", "never mind")
    chinese_tokens = ("取消", "先不", "不要运行", "不运行", "别跑", "暂停")
    english_pattern = r"\b(" + "|".join(re.escape(token) for token in english_tokens) + r")\b"
    return bool(re.search(english_pattern, normalized)) or any(token in text for token in chinese_tokens)


def _looks_like_standalone_cancellation(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    english_exact = {"cancel", "cancel run", "stop", "stop run", "do not run", "don't run", "never mind"}
    chinese_exact = {"取消", "取消运行", "先不", "先不要", "不要运行", "不运行", "别跑", "暂停"}
    return normalized in english_exact or text.strip() in chinese_exact


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _infer_config_overrides_from_text(text: str) -> dict[str, Any]:
    normalized = str(text or "").strip()
    if not normalized:
        return {}
    lowered = normalized.lower()
    if not _looks_like_modification_request(normalized):
        return {}

    overrides: dict[str, Any] = {}
    energy = _extract_energy_mev(lowered)
    if energy is not None:
        overrides["source_energy_mev"] = energy
    events = _extract_requested_events(lowered)
    if events is not None:
        overrides["run_events"] = events
    thickness = _extract_thickness_mm(lowered)
    if thickness is not None:
        overrides["target_thickness_mm"] = thickness
    if _extract_downstream_scoring_request(lowered):
        overrides["enable_downstream_scoring"] = True
    material = _extract_material_id(normalized)
    if material:
        overrides["target_material"] = material
    return overrides


def _looks_like_modification_request(text: str) -> bool:
    if any(token in text for token in ("改", "换", "设为", "设置", "调整", "增加", "降低", "再跑", "再运行")):
        return True
    lowered = text.lower()
    english_tokens = (
        "change",
        "set ",
        "use ",
        "rerun",
        "run again",
        "modify",
        "adjust",
        "increase",
        "decrease",
        "add",
        "enable",
    )
    chinese_tokens = ("改", "换", "设为", "设置", "调整", "增加", "降低", "再跑", "再运行")
    return any(token in lowered for token in english_tokens) or any(token in text for token in chinese_tokens)


def _extract_energy_mev(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(kev|mev|gev)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "kev":
        return value / 1000.0
    if unit == "gev":
        return value * 1000.0
    return value


def _extract_requested_events(text: str) -> int | None:
    clean_match = re.search(r"(\d+)\s*(?:events?|个事件|事件|次)", text, flags=re.IGNORECASE)
    if clean_match:
        return _positive_int(clean_match.group(1), default=1000)
    match = re.search(r"(\d+)\s*(?:events?|个事件|事件|次)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return _positive_int(match.group(1), default=1000)


def _extract_thickness_mm(text: str) -> float | None:
    if any(token in text for token in ("厚", "厚度")):
        match = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", text, flags=re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value * 10.0 if match.group(2).lower() == "cm" else value
    if not any(token in text for token in ("thick", "thickness", "厚", "厚度")):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return value * 10.0 if match.group(2).lower() == "cm" else value


def _extract_downstream_scoring_request(text: str) -> bool:
    has_downstream_target = "downstream" in text and any(token in text for token in ("scoring", "detector", "plane"))
    has_scoring_action = any(token in text for token in ("add", "enable", "include", "with"))
    return has_downstream_target and has_scoring_action


def _extract_material_id(text: str) -> str:
    canonical_ids = {
        "g4_pb": "G4_Pb",
        "g4_cu": "G4_Cu",
        "g4_al": "G4_Al",
        "g4_water": "G4_WATER",
        "g4_polyethylene": "G4_POLYETHYLENE",
        "g4_concrete": "G4_CONCRETE",
        "g4_si": "G4_Si",
    }
    lowered_for_explicit = text.lower()
    explicit = re.search(r"\b(g4_[a-z0-9_]+)\b", lowered_for_explicit, flags=re.IGNORECASE)
    if explicit:
        material_id = explicit.group(1).lower()
        return canonical_ids.get(material_id, explicit.group(1))

    alias_tokens = (
        ("G4_POLYETHYLENE", ("polyethylene", "聚乙烯")),
        ("G4_CONCRETE", ("concrete", "混凝土")),
        ("G4_WATER", ("water", "水")),
        ("G4_Pb", ("lead", "pb", "铅")),
        ("G4_Cu", ("copper", "cu", "铜")),
        ("G4_Al", ("aluminum", "aluminium", "al", "铝")),
        ("G4_Si", ("silicon", "si", "硅")),
    )
    for material_id, tokens in alias_tokens:
        for token in tokens:
            if token.isascii():
                if re.search(rf"\b{re.escape(token.lower())}\b", lowered_for_explicit):
                    return material_id
            elif token in text:
                return material_id

    return ""
    material_tokens = (
        ("G4_Pb", ("g4_pb", "lead", "pb", "铅")),
        ("G4_Cu", ("g4_cu", "copper", "cu", "铜")),
        ("G4_Al", ("g4_al", "aluminum", "aluminium", "al", "铝")),
        ("G4_WATER", ("g4_water", "water", "水")),
        ("G4_POLYETHYLENE", ("g4_polyethylene", "polyethylene", "聚乙烯")),
        ("G4_CONCRETE", ("g4_concrete", "concrete", "混凝土")),
        ("G4_Si", ("g4_si", "silicon", "si", "硅")),
    )
    lowered = text.lower()
    for material_id, tokens in material_tokens:
        if any(token in lowered or token in text for token in tokens):
            return material_id
    return ""


def _looks_like_confirmation(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    english_tokens = ("confirm", "approved", "approve", "go ahead", "yes run")
    chinese_tokens = ("确认", "可以运行", "同意运行", "开始运行", "执行吧", "跑吧")
    english_pattern = r"\b(" + "|".join(re.escape(token) for token in english_tokens) + r")\b"
    return bool(re.search(english_pattern, normalized)) or any(token in text for token in chinese_tokens)


# ── Iterative optimization ───────────────────────────────────────

def _execute_optimization_step(controller: Any, turn: V3TurnInput, state: V3AgentState) -> Any:
    """Execute one step of iterative optimization. LLM suggests next parameter value."""
    opt = turn.metadata.get("optimization")
    if not isinstance(opt, dict) or opt.get("goal_met"):
        return controller.run(turn, state=state)

    param = opt.get("parameter", "target_thickness_mm")
    next_val = opt.get("next_value")
    if next_val is None:
        return controller.run(turn, state=state)

    # Run simulation with the suggested next value
    turn.metadata["config_overrides"] = {param: next_val}
    turn.metadata["accept_defaults"] = True
    turn.metadata["run"] = True
    turn.metadata["run_confirmed"] = False
    runtime_policy_from_turn(turn)
    _prepare_state_for_modified_draft(state)
    result = controller.run(turn, state=state)

    # Present the result with iteration info
    iter_msg = (f"Iteration: tried {param}={next_val}. "
                f"Goal: {opt.get('current_metric', 'optimizing')}. "
                f"Reasoning: {opt.get('reasoning', 'adjusting parameter')}")
    result.answer = V3Answer(message=iter_msg, understanding=state.goal)
    return result


# ── Sweep trend analysis ─────────────────────────────────────────

def _analyze_sweep_trends(sweep_results: list[dict], label: str, turn: V3TurnInput, state: V3AgentState | None) -> str | None:
    """Call LLM to analyze trends in sweep data."""
    llm_config = turn.metadata.get("llm_config_path", "")
    if not llm_config or len(sweep_results) < 2:
        return None
    try:
        from nlu.llm_support.ollama_client import chat, extract_json
        data_str = "\n".join(
            f"  {sr.get('value')}: edep={sr.get('target_edep')} MeV, crossings={sr.get('detector_crossings')}, events={sr.get('events')}"
            for sr in sweep_results
        )
        prompt = f"""You are a physicist analyzing a parameter sweep.

Sweep parameter: {label}
Goal: {state.goal if state else 'physics simulation'}
Data:
{data_str}

Your task:
1. Identify the TREND: is target_edep increasing or decreasing with the parameter? Crossings?
2. Explain the PHYSICS: what interaction mechanism explains this trend?
3. Give a CONCLUSION: what does this mean for the user's goal?

Return JSON:
"trend": one sentence describing the trend.
"physics": one sentence explaining the physics.
"conclusion": one sentence practical takeaway.
"message": combined natural language analysis in Chinese.

Return JSON only. No markdown."""

        response = chat(prompt, config_path=llm_config)
        raw = str(response.get("response") or "")
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and parsed.get("message"):
            return parsed["message"]
    except Exception:
        pass
    return None


# ── Parameter sweep ──────────────────────────────────────────────

def _detect_sweep_request(turn: V3TurnInput, state: V3AgentState | None) -> dict[str, Any] | None:
    text = str(turn.user_text or "").lower()
    if not any(kw in text for kw in ("执行扫描", "参数扫描", "对比扫描", "run sweep", "sweep")):
        return None
    if state is None:
        return None
    for obs in reversed(state.observations):
        if obs.source != "geant4_runtime_tool":
            continue
        sweep_raw = state.metadata.get("llm_sweep_suggestion")
        if isinstance(sweep_raw, dict) and sweep_raw.get("values"):
            return sweep_raw
        energies = _extract_energy_list(text)
        if energies:
            return {"parameter": "source_energy_mev", "values": energies, "label": "Energy (MeV)"}
        thicknesses = _extract_thickness_list(text)
        if thicknesses:
            return {"parameter": "target_thickness_mm", "values": thicknesses, "label": "Thickness (mm)"}
    return None


def _extract_energy_list(text: str) -> list[float]:
    import re
    normalized = text.lower().replace(",", " ").replace("，", " ")
    # Match lists like "0.5, 1.0, 2.0 MeV" or "0.5 1.0 2.0 MeV"
    if re.search(r"(?:mev|kev|gev)", normalized):
        matches = re.findall(r"(\d+(?:\.\d+)?)", normalized)
        return [float(m) for m in matches if float(m) < 10000]  # filter non-energy numbers
    return []


def _extract_thickness_list(text: str) -> list[float]:
    import re
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:mm|cm)(?:\s*[,，\s]\s*|\s+(?:and|或)\s+)", text)
    return [float(m) for m in matches] if len(matches) >= 2 else []


def _execute_sweep(controller: Any, turn: V3TurnInput, state: V3AgentState | None, sweep_spec: dict[str, Any]) -> Any:
    import re
    param = sweep_spec.get("parameter", "source_energy_mev")
    values = sweep_spec.get("values") if isinstance(sweep_spec.get("values"), list) else []
    label = sweep_spec.get("label", str(param))
    if not values or state is None:
        return controller.run(turn, state=state)
    sweep_results: list[dict[str, Any]] = []
    for val in values:
        # Clear stale observations to force re-execution
        if state is not None:
            _prepare_state_for_modified_draft(state)
        turn.metadata["config_overrides"] = {param: val, "run_events": 5}
        turn.metadata["accept_defaults"] = True
        turn.metadata["run"] = True
        turn.metadata["run_confirmed"] = False
        runtime_policy_from_turn(turn)
        result = controller.run(turn, state=state)
        rt_data = None
        for obs in result.observations:
            if obs.source == "geant4_runtime_tool":
                data = obs.data if isinstance(obs.data, dict) else {}
                summary = data.get("result_summary", {})
                sc = summary.get("scoring", {})
                rt_data = {
                    "value": val,
                    "events": (summary.get("run", {}) or {}).get("events_completed", "?"),
                    "target_edep": (sc.get("target", {}) or {}).get("target_edep_total_mev", "N/A"),
                    "detector_crossings": (sc.get("detector_crossing", {}) or {}).get("detector_crossing_count", "N/A"),
                }
                break
        sweep_results.append(rt_data or {"value": val, "events": "?", "target_edep": "N/A"})
        state = result.state
    # Build raw table
    lines = [f"Parameter sweep ({label}):"]
    for sr in sweep_results:
        lines.append(f"  {sr.get('value','?')}: edep={sr.get('target_edep','N/A')} MeV, crossings={sr.get('detector_crossings','N/A')}, events={sr.get('events','?')}")
    raw_message = "\n".join(lines)

    # Try LLM trend analysis
    trend_analysis = _analyze_sweep_trends(sweep_results, label, turn, state)
    if trend_analysis:
        raw_message = raw_message + "\n\n" + trend_analysis

    return V3AgentResult(
        answer=V3Answer(message=raw_message, understanding=state.goal if state else ""),
        state=state or V3AgentState(session_id=turn.session_id),
        trace=[], observations=[], terminated_reason="final_answer",
    )


__all__ = [
    "V3_AGENT_TURN_SCHEMA_VERSION",
    "V3AgentTurnService",
    "build_turn_input",
    "build_v3_agent_controller",
    "serialize_turn_result",
]

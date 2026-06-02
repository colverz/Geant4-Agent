from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import V3AgentState, V3Observation
from .tools.geant4_tools import (
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_LLM_DESIGN_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
)


V3_CONTEXT_SCHEMA_VERSION = "geant4_agent_v3_context.v1"
V3_STATE_SUMMARY_SCHEMA_VERSION = "geant4_agent_v3_state_summary.v1"


@dataclass(slots=True)
class V3ContextPack:
    session_id: str
    goal: str = ""
    phase: str = "start"
    latest_design: dict[str, Any] = field(default_factory=dict)
    latest_payload: dict[str, Any] = field(default_factory=dict)
    latest_runtime_facts: dict[str, Any] = field(default_factory=dict)
    pending_action: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    suggested_next_actions: list[dict[str, Any]] = field(default_factory=list)
    last_user_turn: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_CONTEXT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "goal": self.goal,
            "phase": self.phase,
            "latest_design": dict(self.latest_design),
            "latest_payload": dict(self.latest_payload),
            "latest_runtime_facts": dict(self.latest_runtime_facts),
            "pending_action": dict(self.pending_action),
            "open_questions": list(self.open_questions),
            "assumptions": list(self.assumptions),
            "suggested_next_actions": [dict(item) for item in self.suggested_next_actions],
            "last_user_turn": self.last_user_turn,
        }


def build_v3_context_pack(state: V3AgentState, *, last_user_turn: str = "") -> dict[str, Any]:
    phase = infer_v3_phase(state)
    pack = V3ContextPack(
        session_id=state.session_id,
        goal=state.goal,
        phase=phase,
        latest_design=_latest_design_summary(state),
        latest_payload=_latest_payload_summary(state),
        latest_runtime_facts=_latest_runtime_facts(state),
        pending_action=_pending_action_summary(state),
        open_questions=list(state.open_questions),
        assumptions=list(state.assumptions[:8]),
        suggested_next_actions=_suggested_next_actions(state),
        last_user_turn=last_user_turn or str(state.metadata.get("last_user_turn") or ""),
    )
    return pack.to_dict()


def build_v3_state_summary(state: V3AgentState) -> dict[str, Any]:
    phase = infer_v3_phase(state)
    sources = _evidence_sources(state)
    pending = _pending_action_summary(state)
    preflight = _latest_observation(state, {GEANT4_RUNTIME_PREFLIGHT_TOOL})
    preflight_status = preflight.status.value if preflight is not None else ""
    return {
        "schema_version": V3_STATE_SUMMARY_SCHEMA_VERSION,
        "phase": phase,
        "runtime_ready": _runtime_ready(state),
        "runtime_ready_reason": _runtime_ready_reason(state),
        "needs_confirmation": bool(pending),
        "has_design": _has_design(state),
        "has_payload": _has_source(state, GEANT4_PAYLOAD_BUILDER_TOOL),
        "has_preflight": preflight is not None,
        "preflight_status": preflight_status,
        "has_runtime_result": _has_source(state, GEANT4_RUNTIME_TOOL),
        "next_action": _next_action_for_phase(phase),
        "risk_level": str(pending.get("risk_level") or ""),
        "evidence_sources": sources,
    }


def infer_v3_phase(state: V3AgentState) -> str:
    if _pending_action_summary(state):
        return "await_confirmation"
    if _has_source(state, GEANT4_RUNTIME_TOOL):
        return "runtime_observed"
    preflight = _latest_observation(state, {GEANT4_RUNTIME_PREFLIGHT_TOOL})
    if preflight is not None and preflight.status.value == "ok":
        return "runtime_preflight"
    if preflight is not None:
        return "runtime_preflight_blocked"
    if _has_source(state, GEANT4_PAYLOAD_BUILDER_TOOL):
        return "payload_ready"
    if _has_design(state):
        return "design_ready"
    return "start"


def update_v3_workflow_state(state: V3AgentState, *, last_user_turn: str = "") -> None:
    state.metadata["last_user_turn"] = last_user_turn
    state.assumptions = _merge_unique([*state.assumptions, *_collect_assumptions(state)], limit=8)
    state.active_plan = _active_plan_for_state(state)


def resolve_open_questions_if_answered(state: V3AgentState, *, config_overrides: dict[str, Any] | None = None) -> None:
    if not state.open_questions:
        return
    if config_overrides or _has_source(state, GEANT4_PAYLOAD_BUILDER_TOOL):
        state.open_questions = []


def _active_plan_for_state(state: V3AgentState) -> list[str]:
    plan: list[str] = []
    if _has_source(state, "geant4_capability_tool"):
        plan.append("inspect_capability")
    if _has_design(state):
        plan.append("draft_design")
    if _has_source(state, GEANT4_PAYLOAD_BUILDER_TOOL):
        plan.append("build_payload")
    preflight = _latest_observation(state, {GEANT4_RUNTIME_PREFLIGHT_TOOL})
    if preflight is not None:
        plan.append("run_preflight" if preflight.status.value == "ok" else "fix_preflight")
    if _pending_action_summary(state):
        plan.append("await_confirmation")
    if _has_source(state, GEANT4_RUNTIME_TOOL):
        plan.append("run_runtime")
        plan.append("answer_result")
    return plan or ["start"]


def _latest_design_summary(state: V3AgentState) -> dict[str, Any]:
    obs = _latest_observation(state, {GEANT4_LLM_DESIGN_TOOL, GEANT4_DESIGN_TEMPLATE_TOOL})
    data = obs.data if obs is not None and isinstance(obs.data, dict) else {}
    design = data.get("design") if isinstance(data.get("design"), dict) else {}
    setup = design.get("recommended_setup") if isinstance(design.get("recommended_setup"), dict) else {}
    return {
        "source": obs.source if obs else "",
        "goal": str(design.get("goal") or data.get("goal") or ""),
        "geometry": setup.get("geometry"),
        "material": setup.get("material"),
        "source_type": setup.get("source"),
        "observables": list(design.get("observables") or []) if isinstance(design.get("observables"), list) else [],
        "next_action": str(design.get("next_action") or ""),
    } if obs else {}


def _latest_payload_summary(state: V3AgentState) -> dict[str, Any]:
    obs = _latest_observation(state, {GEANT4_PAYLOAD_BUILDER_TOOL})
    data = obs.data if obs is not None and isinstance(obs.data, dict) else {}
    spec = data.get("simulation_spec") if isinstance(data.get("simulation_spec"), dict) else {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    run = spec.get("run") if isinstance(spec.get("run"), dict) else {}
    return {
        "source": obs.source if obs else "",
        "material": geometry.get("material"),
        "geometry": geometry.get("structure"),
        "particle": source.get("particle"),
        "source_type": source.get("type"),
        "source_energy_mev": source.get("energy_mev"),
        "events": run.get("events"),
        "applied_overrides": dict(data.get("applied_overrides") or {}) if isinstance(data.get("applied_overrides"), dict) else {},
    } if obs else {}


def _latest_runtime_facts(state: V3AgentState) -> dict[str, Any]:
    obs = _latest_observation(state, {GEANT4_RUNTIME_TOOL})
    data = obs.data if obs is not None and isinstance(obs.data, dict) else {}
    result = data.get("result_summary") if isinstance(data.get("result_summary"), dict) else {}
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    config = result.get("configuration") if isinstance(result.get("configuration"), dict) else {}
    scoring = result.get("scoring") if isinstance(result.get("scoring"), dict) else {}
    target = scoring.get("target") if isinstance(scoring.get("target"), dict) else {}
    detector = scoring.get("detector_crossing") if isinstance(scoring.get("detector_crossing"), dict) else {}
    plane = scoring.get("plane_crossing") if isinstance(scoring.get("plane_crossing"), dict) else {}
    payload = data.get("runtime_payload") if isinstance(data.get("runtime_payload"), dict) else {}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    geometry = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    return {
        "source": obs.source if obs else "",
        "adapter": data.get("adapter"),
        "events_requested": run.get("events_requested"),
        "events_completed": run.get("events_completed"),
        "material": _first_present(config.get("material"), geometry.get("material")),
        "geometry": _first_present(config.get("geometry_structure"), geometry.get("structure")),
        "particle": _first_present(config.get("particle"), source.get("particle")),
        "source_type": _first_present(config.get("source_type"), source.get("type")),
        "source_energy_mev": _first_present(source.get("energy_mev"), payload.get("energy")),
        "physics_list": _first_present(config.get("physics_list"), payload.get("physics_list")),
        "target_edep_total_mev": target.get("target_edep_total_mev"),
        "detector_crossing_count": detector.get("detector_crossing_count"),
        "plane_crossing_count": plane.get("plane_crossing_count"),
    } if obs else {}


def _pending_action_summary(state: V3AgentState) -> dict[str, Any]:
    pending = state.metadata.get("pending_action")
    if not isinstance(pending, dict):
        return {}
    return {
        "action_id": pending.get("action_id"),
        "kind": pending.get("kind"),
        "intent": pending.get("intent"),
        "risk_level": pending.get("risk_level"),
        "requires_confirmation": bool(pending.get("requires_confirmation")),
        "expected_observation": pending.get("expected_observation") or "",
    }


def _suggested_next_actions(state: V3AgentState) -> list[dict[str, Any]]:
    raw = state.metadata.get("suggested_next_actions")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:8]:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("label") or item.get("prefill") or "").strip()
            prefill = str(item.get("prefill") or item.get("text") or "").strip()
            if text and prefill:
                out.append({"text": text, "prefill": prefill})
    return out


def _collect_assumptions(state: V3AgentState) -> list[str]:
    out: list[str] = []
    for obs in state.observations:
        if obs.source in {GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_LLM_DESIGN_TOOL}:
            design = obs.data.get("design") if isinstance(obs.data, dict) and isinstance(obs.data.get("design"), dict) else {}
            out.extend(str(item) for item in design.get("assumptions") or [] if str(item))
        if obs.source == GEANT4_PAYLOAD_BUILDER_TOOL:
            data = obs.data if isinstance(obs.data, dict) else {}
            out.extend(str(item) for item in data.get("recommended_config_assumptions") or [] if str(item))
    return out


def _latest_observation(state: V3AgentState, sources: set[str]) -> V3Observation | None:
    for obs in reversed(state.observations):
        if obs.source in sources:
            return obs
    return None


def _has_source(state: V3AgentState, source: str) -> bool:
    return any(obs.source == source for obs in state.observations)


def _has_design(state: V3AgentState) -> bool:
    return any(obs.source in {GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_LLM_DESIGN_TOOL} for obs in state.observations)


def _runtime_ready(state: V3AgentState) -> bool:
    if _pending_action_summary(state):
        return True
    preflight = _latest_observation(state, {GEANT4_RUNTIME_PREFLIGHT_TOOL})
    return preflight is not None and preflight.status.value == "ok"


def _runtime_ready_reason(state: V3AgentState) -> str:
    if _pending_action_summary(state):
        return "awaiting_user_confirmation"
    preflight = _latest_observation(state, {GEANT4_RUNTIME_PREFLIGHT_TOOL})
    if preflight is None:
        return "preflight_not_run"
    if preflight.status.value == "ok":
        return "preflight_ok"
    reason = str(preflight.not_evaluable_reason or "")
    return reason or f"preflight_{preflight.status.value}"


def _evidence_sources(state: V3AgentState) -> list[str]:
    return _merge_unique([obs.source for obs in state.observations if obs.source], limit=20)


def _merge_unique(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _next_action_for_phase(phase: str) -> str:
    return {
        "start": "describe_simulation_goal",
        "design_ready": "accept_or_modify_design",
        "payload_ready": "confirm_run_or_modify_payload",
        "runtime_preflight": "confirm_run_or_fix_runtime",
        "runtime_preflight_blocked": "fix_runtime_or_modify_payload",
        "await_confirmation": "confirm_or_cancel_pending_action",
        "runtime_observed": "ask_result_question_or_modify_and_rerun",
    }.get(phase, "continue")


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


__all__ = [
    "V3_CONTEXT_SCHEMA_VERSION",
    "V3_STATE_SUMMARY_SCHEMA_VERSION",
    "build_v3_context_pack",
    "build_v3_state_summary",
    "infer_v3_phase",
    "resolve_open_questions_if_answered",
    "update_v3_workflow_state",
]

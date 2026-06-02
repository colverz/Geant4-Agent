from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .context import build_v3_context_pack
from .contracts import V3AgentState, V3TurnInput


V3_TURN_UNDERSTANDING_SCHEMA_VERSION = "geant4_agent_v3_turn_understanding.v1"


@dataclass(slots=True)
class V3RequestedChange:
    field: str
    value: Any
    unit: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "unit": self.unit,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class V3TurnUnderstanding:
    dialogue_act: str = "unknown"
    user_goal: str = ""
    referenced_state: str = "none"
    requested_changes: list[V3RequestedChange] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    confirmation: str = "not_applicable"
    risk_intent: str = "read_only"
    ambiguities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_TURN_UNDERSTANDING_SCHEMA_VERSION,
            "dialogue_act": self.dialogue_act,
            "user_goal": self.user_goal,
            "referenced_state": self.referenced_state,
            "requested_changes": [item.to_dict() for item in self.requested_changes],
            "constraints": dict(self.constraints),
            "confirmation": self.confirmation,
            "risk_intent": self.risk_intent,
            "ambiguities": list(self.ambiguities),
            "confidence": max(0.0, min(1.0, float(self.confidence))),
            "reason": self.reason,
            "source": self.source,
        }


def build_v3_turn_understanding(turn: V3TurnInput, state: V3AgentState | None = None) -> V3TurnUnderstanding:
    event = normalize_confirmation_event(turn.metadata.get("confirmation_event"))
    if event:
        decision = str(event.get("decision") or "").lower()
        return V3TurnUnderstanding(
            dialogue_act="confirm" if decision == "confirm" else "reject",
            user_goal=turn.user_text,
            referenced_state="pending_action",
            confirmation="confirmed" if decision == "confirm" else "rejected",
            risk_intent="run_requested" if decision == "confirm" else "read_only",
            confidence=1.0,
            reason="explicit UI confirmation event",
            source="explicit_event",
        )

    changes = _changes_from_config_overrides(turn)
    if changes:
        return V3TurnUnderstanding(
            dialogue_act="revise",
            user_goal=turn.user_text,
            referenced_state=_referenced_state(state),
            requested_changes=changes,
            confirmation="not_applicable",
            risk_intent="run_requested" if bool(turn.metadata.get("run")) else "draft_only",
            confidence=0.9,
            reason="structured config overrides supplied by v3 service or LLM",
            source="metadata",
        )

    if bool(turn.metadata.get("run")):
        return V3TurnUnderstanding(
            dialogue_act="request_run",
            user_goal=turn.user_text,
            referenced_state=_referenced_state(state),
            confirmation="not_applicable",
            risk_intent="run_requested",
            confidence=0.8,
            reason="explicit run metadata",
            source="metadata",
        )

    return V3TurnUnderstanding(
        dialogue_act="ask",
        user_goal=turn.user_text,
        referenced_state=_referenced_state(state),
        confirmation="not_applicable",
        risk_intent="read_only",
        confidence=0.5,
        reason="no explicit event or structured change was provided",
        source="fallback",
    )


class LLMTurnUnderstandingProvider:
    """Context-pack-only LLM parser for the user's latest turn.

    The provider never receives raw session metadata or full observations. It
    can suggest requested changes, but patch validation remains the authority
    before any state mutation happens.
    """

    def __init__(self, llm_config_path: str, *, min_confidence: float = 0.55) -> None:
        self._llm_config_path = str(llm_config_path or "").strip()
        self._min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def understand(self, turn: V3TurnInput, state: V3AgentState | None = None) -> V3TurnUnderstanding:
        if not self._llm_config_path:
            return build_v3_turn_understanding(turn, state)
        if normalize_confirmation_event(turn.metadata.get("confirmation_event")):
            return build_v3_turn_understanding(turn, state)
        try:
            raw = self._call_llm(self._prompt(turn, state))
            parsed = self._parse_llm_response(raw)
            if not isinstance(parsed, dict):
                return _llm_uncertain_understanding(turn, state, "llm_output_not_json")
            understanding = _understanding_from_mapping(parsed, turn, state, source="llm")
            if understanding.confidence < self._min_confidence:
                return _llm_uncertain_understanding(turn, state, "llm_confidence_below_threshold", parsed)
            return understanding
        except Exception as exc:
            fallback = build_v3_turn_understanding(turn, state)
            fallback.reason = f"llm_turn_understanding_failed:{exc.__class__.__name__}"
            fallback.source = "fallback_after_llm_error"
            return fallback

    def _prompt(self, turn: V3TurnInput, state: V3AgentState | None = None) -> str:
        context_state = state or V3AgentState(session_id=turn.session_id, goal=turn.user_text)
        context_pack = build_v3_context_pack(context_state, last_user_turn=turn.user_text)
        context_json = json.dumps(context_pack, ensure_ascii=False, indent=2)
        return f"""You are the turn-understanding layer for a Geant4 v3 agent.

Read only the V3ContextPack and the latest user turn. Do not infer from hidden
metadata, raw observations, or examples. Produce a conservative structured
understanding; safety gates and patch validation will decide what can execute.

Latest user turn:
{turn.user_text}

V3ContextPack:
{context_json}

Return JSON only:
{{
  "dialogue_act": "ask|answer|revise|request_run|confirm|reject",
  "user_goal": "short restatement of the latest user intent",
  "referenced_state": "none|design|payload|runtime_result|pending_action",
  "requested_changes": [
    {{"field": "source_energy_mev|run_events|target_material|target_thickness_mm|geometry_dimensions_mm", "value": "...", "unit": "", "evidence": "short quote or reason"}}
  ],
  "constraints": {{}},
  "confirmation": "confirmed|rejected|not_applicable",
  "risk_intent": "read_only|draft_only|run_requested",
  "ambiguities": ["question if needed"],
  "confidence": 0.0,
  "reason": "brief explanation"
}}

Rules:
- Do not set confirmation=confirmed unless the user is clearly approving a
  pending action or explicitly says to run.
- Use requested_changes only for user edits to an existing design/payload.
- Prefer Geant4 material ids such as G4_WATER, not common names like water.
- If a value is ambiguous, leave requested_changes empty and put a question in
  ambiguities.
"""

    def _call_llm(self, prompt: str) -> str:
        from nlu.llm_support.ollama_client import chat

        result = chat(prompt, config_path=self._llm_config_path)
        return str(result.get("response") or "") if result else ""

    def _parse_llm_response(self, raw: str) -> dict[str, Any] | None:
        from nlu.llm_support.ollama_client import extract_json

        parsed = extract_json(raw)
        return parsed if isinstance(parsed, dict) else None


def normalize_confirmation_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    decision = str(raw.get("decision") or "").strip().lower()
    if decision not in {"confirm", "cancel", "reject"}:
        return {}
    return {
        "action_id": str(raw.get("action_id") or raw.get("id") or "").strip(),
        "decision": "confirm" if decision == "confirm" else "reject",
    }


def _changes_from_config_overrides(turn: V3TurnInput) -> list[V3RequestedChange]:
    raw = turn.metadata.get("config_overrides")
    if not isinstance(raw, dict):
        return []
    changes: list[V3RequestedChange] = []
    for key, value in raw.items():
        unit = "MeV" if str(key).endswith("_mev") else "mm" if str(key).endswith("_mm") else ""
        changes.append(
            V3RequestedChange(
                field=str(key),
                value=value,
                unit=unit,
                evidence="config_overrides",
            )
        )
    return changes


def _referenced_state(state: V3AgentState | None) -> str:
    if state is None:
        return "none"
    if state.metadata.get("pending_action"):
        return "pending_action"
    sources = [obs.source for obs in state.observations]
    if "geant4_runtime_tool" in sources:
        return "runtime_result"
    if "geant4_payload_builder_tool" in sources:
        return "payload"
    if "geant4_llm_design_tool" in sources or "geant4_design_template_tool" in sources:
        return "design"
    return "none"


def _understanding_from_mapping(
    data: dict[str, Any],
    turn: V3TurnInput,
    state: V3AgentState | None,
    *,
    source: str,
) -> V3TurnUnderstanding:
    dialogue_act = _allowed_value(data.get("dialogue_act"), {"ask", "answer", "revise", "request_run", "confirm", "reject"}, "ask")
    referenced_state = _allowed_value(
        data.get("referenced_state"),
        {"none", "design", "payload", "runtime_result", "pending_action"},
        _referenced_state(state),
    )
    confirmation = _allowed_value(
        data.get("confirmation"),
        {"confirmed", "rejected", "not_applicable"},
        "not_applicable",
    )
    risk_intent = _allowed_value(data.get("risk_intent"), {"read_only", "draft_only", "run_requested"}, "read_only")
    confidence = _float_between(data.get("confidence"), default=0.0)
    return V3TurnUnderstanding(
        dialogue_act=dialogue_act,
        user_goal=str(data.get("user_goal") or turn.user_text),
        referenced_state=referenced_state,
        requested_changes=_requested_changes_from_raw(data.get("requested_changes")),
        constraints=data.get("constraints") if isinstance(data.get("constraints"), dict) else {},
        confirmation=confirmation,
        risk_intent=risk_intent,
        ambiguities=[str(item) for item in data.get("ambiguities") or [] if str(item)],
        confidence=confidence,
        reason=str(data.get("reason") or ""),
        source=source,
    )


def _requested_changes_from_raw(raw: Any) -> list[V3RequestedChange]:
    if not isinstance(raw, list):
        return []
    changes: list[V3RequestedChange] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or item.get("path") or item.get("parameter") or "").strip()
        if not field:
            continue
        changes.append(
            V3RequestedChange(
                field=field,
                value=item.get("value"),
                unit=str(item.get("unit") or ""),
                evidence=str(item.get("evidence") or "llm_turn_understanding"),
            )
        )
    return changes


def _llm_uncertain_understanding(
    turn: V3TurnInput,
    state: V3AgentState | None,
    reason: str,
    parsed: dict[str, Any] | None = None,
) -> V3TurnUnderstanding:
    ambiguities = []
    if isinstance(parsed, dict):
        ambiguities = [str(item) for item in parsed.get("ambiguities") or [] if str(item)]
    return V3TurnUnderstanding(
        dialogue_act="ask",
        user_goal=turn.user_text,
        referenced_state=_referenced_state(state),
        confirmation="not_applicable",
        risk_intent="read_only",
        ambiguities=ambiguities or ["Please clarify the intended Geant4 action or parameter change."],
        confidence=_float_between(parsed.get("confidence"), default=0.0) if isinstance(parsed, dict) else 0.0,
        reason=reason,
        source="llm_uncertain",
    )


def _allowed_value(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _float_between(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


__all__ = [
    "LLMTurnUnderstandingProvider",
    "V3RequestedChange",
    "V3TurnUnderstanding",
    "V3_TURN_UNDERSTANDING_SCHEMA_VERSION",
    "build_v3_turn_understanding",
    "normalize_confirmation_event",
]

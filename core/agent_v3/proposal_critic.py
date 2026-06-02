from __future__ import annotations

import json
from typing import Any

from .contracts import V3ActionKind, V3ActionProposal, V3AgentState, V3Observation, V3ObservationStatus
from .tool_registry import ToolRegistry
from .tools import (
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_LLM_DESIGN_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
)


_INTERNAL_ONLY_ARGUMENTS = {
    "run_confirmed",
    "confirmation_event",
    "pending_action",
    "state_patch",
    "turn_understanding",
}


def review_v3_proposal(
    proposal: V3ActionProposal,
    state: V3AgentState,
    tools: ToolRegistry | None = None,
) -> V3Observation | None:
    tool_name = proposal.tool_call.tool_name if proposal.tool_call else ""
    if proposal.tool_call is not None and tools is not None:
        if not tools.has_tool(tool_name):
            return _blocked(
                "unknown_tool",
                f"Tool `{tool_name}` is not registered in the v3 tool registry.",
                {"tool_name": tool_name},
            )
        internal_keys = sorted(key for key in proposal.tool_call.arguments if key in _INTERNAL_ONLY_ARGUMENTS)
        if internal_keys:
            return _blocked(
                "internal_argument_not_allowed",
                f"Tool arguments include internal-only fields: {', '.join(internal_keys)}.",
                {"fields": internal_keys, "tool_name": tool_name},
            )

    wants_runtime = proposal.kind == V3ActionKind.RUN_SIMULATION or tool_name == GEANT4_RUNTIME_TOOL
    if wants_runtime:
        if not _has_ok_observation(state, GEANT4_PAYLOAD_BUILDER_TOOL):
            return _blocked("runtime_without_payload", "Runtime execution needs a drafted payload before Geant4 can run.")
        if not _has_ok_observation(state, GEANT4_RUNTIME_PREFLIGHT_TOOL):
            return _blocked("runtime_without_preflight", "Runtime execution needs a preflight observation before confirmation.")

    if proposal.tool_call is not None and tools is not None:
        schema_errors = tools.validate_call(proposal.tool_call)
        if schema_errors:
            return _blocked(
                "tool_schema_invalid",
                _schema_error_message(tool_name, schema_errors),
                {"tool_name": tool_name, "schema_errors": schema_errors, "repair_suggestions": _repair_suggestions(schema_errors)},
            )
        grounding_errors = _grounding_errors(proposal, state)
        if grounding_errors:
            return _blocked(
                "context_fact_not_grounded",
                _grounding_error_message(tool_name, grounding_errors),
                {
                    "tool_name": tool_name,
                    "grounding_errors": grounding_errors,
                    "repair_suggestions": _grounding_repair_suggestions(grounding_errors),
                },
            )
        effective_risk = tools.effective_risk(proposal.tool_call)
        if effective_risk != proposal.tool_call.risk_level or effective_risk != proposal.risk_level:
            return V3Observation(
                source="proposal_critic",
                status=V3ObservationStatus.OK,
                data={
                    "reason": "tool_contract_risk_mismatch",
                    "tool_name": tool_name,
                    "proposal_tool_risk": proposal.tool_call.risk_level.label,
                    "proposal_risk": proposal.risk_level.label,
                    "registered_risk": effective_risk.label,
                    "action": "controller_will_gate_using_registered_risk",
                },
                message=(
                    f"Proposal risk for `{tool_name}` differs from the registered tool risk. "
                    f"The controller will use `{effective_risk.label}`."
                ),
            )
    return None


def _has_ok_observation(state: V3AgentState, source: str) -> bool:
    return any(
        observation.source == source and observation.status == V3ObservationStatus.OK
        for observation in state.observations
    )


def _grounding_errors(proposal: V3ActionProposal, state: V3AgentState) -> list[str]:
    if proposal.tool_call is None:
        return []
    tool_name = proposal.tool_call.tool_name
    args = proposal.tool_call.arguments
    errors: list[str] = []
    if tool_name == GEANT4_PAYLOAD_BUILDER_TOOL and "design" in args:
        latest_design = _latest_design(state)
        if latest_design is None:
            errors.append("missing_context_design")
        elif not _json_equal(args.get("design"), latest_design):
            errors.append("ungrounded_design_argument")
    if tool_name in {GEANT4_RUNTIME_PREFLIGHT_TOOL, GEANT4_RUNTIME_TOOL}:
        if "payload_builder_observation" in args:
            latest_payload = _latest_payload(state)
            if latest_payload is None:
                errors.append("missing_context_payload")
            elif not _json_equal(args.get("payload_builder_observation"), latest_payload):
                errors.append("ungrounded_payload_argument")
        if "recommended_config" in args:
            latest_config = _latest_payload_field(state, "recommended_config")
            if latest_config is None:
                errors.append("missing_context_recommended_config")
            elif not _json_equal(args.get("recommended_config"), latest_config):
                errors.append("ungrounded_recommended_config")
    return errors


def _latest_design(state: V3AgentState) -> dict[str, Any] | None:
    observation = _latest_observation(state, {GEANT4_LLM_DESIGN_TOOL, GEANT4_DESIGN_TEMPLATE_TOOL})
    if observation is None:
        return None
    design = observation.data.get("design") if isinstance(observation.data, dict) else None
    return design if isinstance(design, dict) else None


def _latest_payload(state: V3AgentState) -> dict[str, Any] | None:
    observation = _latest_observation(state, {GEANT4_PAYLOAD_BUILDER_TOOL})
    if observation is None:
        return None
    return observation.data if isinstance(observation.data, dict) else None


def _latest_payload_field(state: V3AgentState, field: str) -> Any | None:
    payload = _latest_payload(state)
    if not isinstance(payload, dict):
        return None
    value = payload.get(field)
    return value if value is not None else None


def _latest_observation(state: V3AgentState, sources: set[str]) -> V3Observation | None:
    for observation in reversed(state.observations):
        if observation.source in sources and observation.status == V3ObservationStatus.OK:
            return observation
    return None


def _json_equal(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return repr(value)


def _blocked(reason: str, message: str, extra: dict | None = None) -> V3Observation:
    data = {"reason": reason}
    if extra:
        data.update(extra)
    return V3Observation(
        source="proposal_critic",
        status=V3ObservationStatus.BLOCKED,
        data=data,
        message=message,
    )


def _schema_error_message(tool_name: str, errors: list[str]) -> str:
    suggestion = "; ".join(_repair_suggestions(errors))
    return f"Tool arguments for `{tool_name}` do not match the registered schema: {errors[0]}. {suggestion}"


def _repair_suggestions(errors: list[str]) -> list[str]:
    suggestions: list[str] = []
    for error in errors:
        if error.startswith("missing_required:"):
            field = error.split(":", 1)[1]
            suggestions.append(f"Provide required field `{field}` from current context or rebuild the prerequisite draft.")
        elif error.startswith("unexpected_property:"):
            field = error.split(":", 1)[1]
            suggestions.append(f"Remove unsupported field `{field}` from tool arguments.")
        elif error.startswith("type_mismatch:"):
            parts = error.split(":")
            field = parts[1] if len(parts) > 1 else "value"
            expected = parts[2].replace("expected_", "") if len(parts) > 2 else "the expected type"
            suggestions.append(f"Normalize `{field}` to {expected} before calling the tool.")
        elif error.startswith("minimum:"):
            parts = error.split(":")
            field = parts[1] if len(parts) > 1 else "value"
            minimum = parts[2] if len(parts) > 2 else "the minimum"
            suggestions.append(f"Set `{field}` to at least {minimum}.")
    return suggestions or ["Rebuild the tool arguments from V3ContextPack and try again."]


def _grounding_error_message(tool_name: str, errors: list[str]) -> str:
    suggestion = "; ".join(_grounding_repair_suggestions(errors))
    return f"Tool arguments for `{tool_name}` reference facts that are not grounded in the current v3 context: {errors[0]}. {suggestion}"


def _grounding_repair_suggestions(errors: list[str]) -> list[str]:
    suggestions: list[str] = []
    for error in errors:
        if error in {"missing_context_design", "ungrounded_design_argument"}:
            suggestions.append("Use the latest design observation from the current session, or draft a new design first.")
        elif error in {"missing_context_payload", "ungrounded_payload_argument"}:
            suggestions.append("Use the latest payload observation from the current session, or rebuild the payload first.")
        elif error in {"missing_context_recommended_config", "ungrounded_recommended_config"}:
            suggestions.append("Use the recommended_config from the latest payload observation.")
    return suggestions or ["Rebuild the proposal from V3ContextPack before calling the tool."]


__all__ = ["review_v3_proposal"]

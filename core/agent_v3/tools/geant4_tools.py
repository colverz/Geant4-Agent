from __future__ import annotations

import re
from typing import Any

from .._v2_bridge import (
    bridge_design_candidate,
    bridge_design_reference_pack,
    bridge_llm_design_candidate,
    bridge_runtime_capabilities,
)
from core.simulation import build_simulation_spec
from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import InMemoryGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.runtime_payload import runtime_capabilities_payload
from mcp.geant4.runtime_payload import build_runtime_payload
from mcp.geant4.server import Geant4McpServer

from ._config_builder import (
    _apply_config_overrides,
    _augment_goal_for_design,
    _coerce_int,
    _events_from_config,
    _simulation_spec_summary,
    build_recommended_config_from_design,
)
from ..contracts import V3Observation, V3ObservationStatus, V3ToolCall, V3ToolRiskLevel
from ..tool_registry import ToolRegistry, ToolSpec


GEANT4_CAPABILITY_TOOL = "geant4_capability_tool"
GEANT4_LLM_DESIGN_TOOL = "geant4_llm_design_tool"
GEANT4_DESIGN_TEMPLATE_TOOL = "geant4_design_template_tool"
GEANT4_PAYLOAD_BUILDER_TOOL = "geant4_payload_builder_tool"
GEANT4_RUNTIME_PREFLIGHT_TOOL = "geant4_runtime_preflight_tool"
GEANT4_RUNTIME_TOOL = "geant4_runtime_tool"


def geant4_capability_handler(call: V3ToolCall) -> V3Observation:
    runtime_capabilities = runtime_capabilities_payload()
    design_capabilities = bridge_runtime_capabilities()
    data = {
        "runtime_capabilities": runtime_capabilities,
        "design_capabilities": design_capabilities,
        "supported_scenarios": [
            "shielding",
            "detector_response",
            "medical_dose",
            "ndt_contrast",
        ],
        "tool_arguments": dict(call.arguments),
    }
    return V3Observation(
        source=GEANT4_CAPABILITY_TOOL,
        status=V3ObservationStatus.OK,
        data=data,
        message="Geant4 capability information is available for design planning.",
    )


def geant4_design_template_handler(call: V3ToolCall) -> V3Observation:
    goal = str(call.arguments.get("goal") or "").strip()
    if not goal:
        return V3Observation(
            source=GEANT4_DESIGN_TEMPLATE_TOOL,
            status=V3ObservationStatus.FAILED,
            message="A user goal is required to draft a Geant4 design.",
        )
    augmented_goal = _augment_goal_for_design(goal)
    runtime_capabilities = call.arguments.get("runtime_capabilities")
    capabilities = runtime_capabilities if isinstance(runtime_capabilities, dict) else None
    candidate = bridge_design_candidate(
        augmented_goal,
        runtime_capabilities=capabilities,
    )
    reference_pack = bridge_design_reference_pack(
        augmented_goal,
        runtime_capabilities=capabilities,
    )
    data = {
        "schema_version": "geant4_agent_v3_design_template_observation.v1",
        "goal": goal,
        "augmented_goal": augmented_goal,
        "design": candidate.to_dict(),
        "reference_pack": {
            "schema_version": reference_pack.get("schema_version"),
            "query_hints": reference_pack.get("query_hints"),
            "runtime_capabilities": reference_pack.get("runtime_capabilities"),
        },
        "artifact": {
            "type": "SimulationDesign",
            "artifact_id": str(call.arguments.get("artifact_id") or "geant4_design_draft"),
            "status": "draft",
        },
    }
    next_action = candidate.next_action
    return V3Observation(
        source=GEANT4_DESIGN_TEMPLATE_TOOL,
        status=V3ObservationStatus.OK,
        data=data,
        message=f"Drafted a Geant4 simulation design; recommended next action: {next_action}.",
    )


def geant4_llm_design_handler(call: V3ToolCall) -> V3Observation:
    goal = str(call.arguments.get("goal") or "").strip()
    config_path = str(call.arguments.get("llm_config_path") or "").strip()
    if not goal:
        return V3Observation(
            source=GEANT4_LLM_DESIGN_TOOL,
            status=V3ObservationStatus.FAILED,
            message="A user goal is required to draft a Geant4 design.",
        )
    if not config_path:
        return V3Observation(
            source=GEANT4_LLM_DESIGN_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            not_evaluable_reason="llm_config_path_required",
            message="LLM design was requested, but no LLM config path was provided.",
        )
    runtime_capabilities = call.arguments.get("runtime_capabilities")
    capabilities = runtime_capabilities if isinstance(runtime_capabilities, dict) else None
    lang = str(call.arguments.get("lang") or "zh-CN")
    llm_result = bridge_llm_design_candidate(
        goal,
        config_path=config_path,
        lang=lang,
        runtime_capabilities=capabilities,
    )
    if llm_result.get("ok") and isinstance(llm_result.get("candidate"), dict):
        candidate = _ground_design_candidate_to_goal(llm_result["candidate"], goal)
        reference_pack = llm_result.get("reference_pack") if isinstance(llm_result.get("reference_pack"), dict) else {}
        data = {
            "schema_version": "geant4_agent_v3_llm_design_observation.v1",
            "goal": goal,
            "design": candidate,
            "llm": {
                "used": True,
                "ok": True,
                "prompt_profile_id": llm_result.get("prompt_profile_id", ""),
                "prompt_validation": llm_result.get("prompt_validation", {}),
                "raw_response": llm_result.get("raw_response", ""),
            },
            "reference_pack": _reference_pack_summary(reference_pack),
            "artifact": {
                "type": "SimulationDesign",
                "artifact_id": str(call.arguments.get("artifact_id") or "geant4_design_draft"),
                "status": "draft",
            },
        }
        return V3Observation(
            source=GEANT4_LLM_DESIGN_TOOL,
            status=V3ObservationStatus.OK,
            data=data,
            message="Drafted a Geant4 simulation design with the LLM-assisted v3 design tool.",
        )

    fallback = bridge_design_candidate(
        _augment_goal_for_design(goal),
        runtime_capabilities=capabilities,
    )
    reference_pack = bridge_design_reference_pack(goal, runtime_capabilities=capabilities)
    data = {
        "schema_version": "geant4_agent_v3_llm_design_observation.v1",
        "goal": goal,
        "design": _ground_design_candidate_to_goal(fallback.to_dict(), goal),
            "llm": {
                "used": True,
                "ok": False,
                "fallback_reason": str(llm_result.get("fallback_reason") or "llm_design_failed"),
                "prompt_profile_id": llm_result.get("prompt_profile_id", ""),
                "prompt_validation": llm_result.get("prompt_validation", {}),
                "raw_response": llm_result.get("raw_response", ""),
            },
        "reference_pack": _reference_pack_summary(reference_pack),
        "artifact": {
            "type": "SimulationDesign",
            "artifact_id": str(call.arguments.get("artifact_id") or "geant4_design_draft"),
            "status": "draft",
        },
    }
    return V3Observation(
        source=GEANT4_LLM_DESIGN_TOOL,
        status=V3ObservationStatus.OK,
        data=data,
        message="LLM design was unavailable or rejected; v3 used deterministic design fallback.",
    )


_OBSERVATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["source", "status", "data"],
    "properties": {
        "source": {"type": "string"},
        "status": {"type": "string", "enum": ["ok", "failed", "blocked", "not_evaluable"]},
        "data": {"type": "object"},
        "message": {"type": "string"},
        "not_evaluable_reason": {"type": "string"},
    },
}

_CAPABILITY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
    },
    "additionalProperties": True,
}

_DESIGN_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["goal"],
    "properties": {
        "goal": {"type": "string"},
        "artifact_id": {"type": "string"},
        "runtime_capabilities": {"type": "object"},
    },
    "additionalProperties": False,
}

_LLM_DESIGN_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["goal", "llm_config_path"],
    "properties": {
        "goal": {"type": "string"},
        "artifact_id": {"type": "string"},
        "runtime_capabilities": {"type": "object"},
        "llm_config_path": {"type": "string"},
        "lang": {"type": "string"},
    },
    "additionalProperties": False,
}

_PAYLOAD_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["design"],
    "properties": {
        "design": {"type": "object"},
        "events": {"type": "integer", "minimum": 1},
        "accept_defaults": {"type": "boolean"},
        "config_overrides": {"type": "object"},
        "artifact_id": {"type": "string"},
    },
    "additionalProperties": False,
}

_RUNTIME_PREFLIGHT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["payload_builder_observation"],
    "properties": {
        "payload_builder_observation": {"type": "object"},
        "recommended_config": {"type": "object"},
        "events": {"type": "integer", "minimum": 1},
        "allow_in_memory": {"type": "boolean"},
        "env": {"type": "object"},
    },
    "additionalProperties": False,
}

_RUNTIME_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["payload_builder_observation"],
    "properties": {
        "payload_builder_observation": {"type": "object"},
        "recommended_config": {"type": "object"},
        "events": {"type": "integer", "minimum": 1},
        "allow_in_memory": {"type": "boolean"},
        "env": {"type": "object"},
    },
    "additionalProperties": False,
}


def build_default_geant4_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=GEANT4_CAPABILITY_TOOL,
            risk_level=V3ToolRiskLevel.READ_ONLY,
            handler=geant4_capability_handler,
            description="Read Geant4 runtime and design capabilities.",
            input_schema=_CAPABILITY_INPUT_SCHEMA,
            output_schema=_OBSERVATION_OUTPUT_SCHEMA,
            idempotency_hint="Read-only; safe to repeat.",
        )
    )
    registry.register(
        ToolSpec(
            name=GEANT4_LLM_DESIGN_TOOL,
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            handler=geant4_llm_design_handler,
            description="Draft a Geant4 simulation design with an LLM inside the v3 draft-only boundary.",
            input_schema=_LLM_DESIGN_INPUT_SCHEMA,
            output_schema=_OBSERVATION_OUTPUT_SCHEMA,
            idempotency_hint="Draft-only; repeat may vary with LLM output.",
        )
    )
    registry.register(
        ToolSpec(
            name=GEANT4_DESIGN_TEMPLATE_TOOL,
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            handler=geant4_design_template_handler,
            description="Draft a Geant4 simulation design from an open-ended physics goal.",
            input_schema=_DESIGN_INPUT_SCHEMA,
            output_schema=_OBSERVATION_OUTPUT_SCHEMA,
            idempotency_hint="Draft-only deterministic helper; safe to repeat.",
        )
    )
    registry.register(
        ToolSpec(
            name=GEANT4_PAYLOAD_BUILDER_TOOL,
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            handler=geant4_payload_builder_handler,
            description="Build a draft SimulationSpec and runtime payload from a Geant4 simulation design.",
            input_schema=_PAYLOAD_INPUT_SCHEMA,
            output_schema=_OBSERVATION_OUTPUT_SCHEMA,
            idempotency_hint="Draft-only; safe to repeat for the same design and overrides.",
        )
    )
    registry.register(
        ToolSpec(
            name=GEANT4_RUNTIME_PREFLIGHT_TOOL,
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            handler=geant4_runtime_preflight_handler,
            description="Validate the draft runtime payload and report whether a real local Geant4 runtime is available.",
            input_schema=_RUNTIME_PREFLIGHT_INPUT_SCHEMA,
            output_schema=_OBSERVATION_OUTPUT_SCHEMA,
            idempotency_hint="Validation-only; safe to repeat.",
        )
    )
    registry.register(
        ToolSpec(
            name=GEANT4_RUNTIME_TOOL,
            risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
            handler=geant4_runtime_handler,
            description="Execute a Geant4 run through the MCP runtime adapter after confirmation.",
            input_schema=_RUNTIME_INPUT_SCHEMA,
            output_schema=_OBSERVATION_OUTPUT_SCHEMA,
            confirmation_required=True,
            idempotency_hint="Runtime execution; use a run id or pending action confirmation to avoid accidental repeats.",
        )
    )
    return registry


def _reference_pack_summary(reference_pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": reference_pack.get("schema_version"),
        "query_hints": reference_pack.get("query_hints"),
        "runtime_capabilities": reference_pack.get("runtime_capabilities"),
    }


def _ground_design_candidate_to_goal(candidate: dict[str, Any], goal: str) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    grounded = dict(candidate)
    setup = grounded.get("recommended_setup") if isinstance(grounded.get("recommended_setup"), dict) else {}
    setup = dict(setup)
    corrections: list[str] = []

    material = _explicit_material_from_goal(goal)
    if material:
        if setup.get("material") != material:
            corrections.append(f"material:{setup.get('material')}->{material}")
        setup["material"] = material
        setup["target_material"] = material
        _ground_geometry_material(setup.get("geometry"), material)
        refs = grounded.get("knowledge_references") if isinstance(grounded.get("knowledge_references"), list) else []
        material_ref = f"materials:{material}"
        grounded["knowledge_references"] = list(dict.fromkeys([*refs, material_ref]))

    energy = _explicit_energy_mev_from_goal(goal)
    if energy is not None:
        if setup.get("source_energy_mev") != energy:
            corrections.append(f"source_energy_mev:{setup.get('source_energy_mev')}->{energy:g}")
        setup["source_energy_mev"] = energy

    particle = _explicit_particle_from_goal(goal)
    if particle:
        if setup.get("source_particle") != particle:
            corrections.append(f"source_particle:{setup.get('source_particle')}->{particle}")
        setup["source_particle"] = particle

    grounded["recommended_setup"] = setup
    if corrections:
        grounded["grounding_corrections"] = corrections
    return grounded


def _ground_geometry_material(geometry: Any, material: str) -> None:
    if not isinstance(geometry, dict):
        return
    volumes = geometry.get("volumes")
    if not isinstance(volumes, list):
        return
    for volume in volumes:
        if not isinstance(volume, dict):
            continue
        role = str(volume.get("role") or "").lower()
        if role in {"target", "phantom", "root"} or not role:
            volume["material"] = material
            return


def _explicit_material_from_goal(goal: str) -> str:
    text = str(goal or "")
    lowered = text.lower()
    match = re.search(r"\bG4_[A-Za-z0-9_-]+\b", text)
    if match:
        return match.group(0)
    word_map = {
        "water": "G4_WATER",
        "phantom": "G4_WATER",
        "水": "G4_WATER",
        "水等效": "G4_WATER",
        "lead": "G4_Pb",
        "铅": "G4_Pb",
        "copper": "G4_Cu",
        "铜": "G4_Cu",
        "aluminum": "G4_Al",
        "aluminium": "G4_Al",
        "铝": "G4_Al",
        "silicon": "G4_Si",
        "硅": "G4_Si",
        "polyethylene": "G4_POLYETHYLENE",
        "聚乙烯": "G4_POLYETHYLENE",
    }
    for token, material in word_map.items():
        if token in lowered or token in text:
            return material
    return ""


def _explicit_energy_mev_from_goal(goal: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(keV|MeV|GeV)\b", str(goal or ""), flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "kev":
        return value / 1000.0
    if unit == "gev":
        return value * 1000.0
    return value


def _explicit_particle_from_goal(goal: str) -> str:
    text = str(goal or "").lower()
    particles = {
        "proton": "proton",
        "质子": "proton",
        "gamma": "gamma",
        "photon": "gamma",
        "光子": "gamma",
        "neutron": "neutron",
        "中子": "neutron",
        "electron": "electron",
        "电子": "electron",
    }
    for token, particle in particles.items():
        if token in text or token in str(goal or ""):
            return particle
    return ""


def geant4_payload_builder_handler(call: V3ToolCall) -> V3Observation:
    design = call.arguments.get("design")
    if not isinstance(design, dict):
        return V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.FAILED,
            message="A SimulationDesign dictionary is required to build a runtime payload.",
        )
    events = _coerce_int(call.arguments.get("events"), 1000)
    accept_defaults = bool(call.arguments.get("accept_defaults"))
    config, assumptions = build_recommended_config_from_design(design, events=events, accept_defaults=accept_defaults)
    if not config:
        return V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            data={"design_next_action": design.get("next_action")},
            not_evaluable_reason="design_not_ready_for_config",
            message="The design is not ready for SimulationSpec or runtime payload generation.",
        )
    applied_overrides = _apply_config_overrides(config, call.arguments.get("config_overrides"))
    if applied_overrides:
        assumptions.extend(f"v3 language override applied: {key}={value}" for key, value in applied_overrides.items())
    spec = build_simulation_spec(config, events=events)
    payload = build_runtime_payload(spec)
    data = {
        "schema_version": "geant4_agent_v3_payload_builder_observation.v1",
        "status": "draft",
        "recommended_config": config,
        "recommended_config_assumptions": assumptions,
        "applied_overrides": applied_overrides,
        "simulation_spec": _simulation_spec_summary(spec),
        "runtime_payload": payload,
        "artifact": {
            "type": "RuntimePayload",
            "artifact_id": str(call.arguments.get("artifact_id") or "geant4_runtime_payload_draft"),
            "status": "draft",
        },
    }
    return V3Observation(
        source=GEANT4_PAYLOAD_BUILDER_TOOL,
        status=V3ObservationStatus.OK,
        data=data,
        message="Built a draft SimulationSpec and Geant4 runtime payload. Nothing has been executed.",
    )


def geant4_runtime_preflight_handler(call: V3ToolCall) -> V3Observation:
    config = _config_from_call(call)
    events = _coerce_int(call.arguments.get("events"), _events_from_config(config, 1000))
    allow_in_memory = bool(call.arguments.get("allow_in_memory"))
    if not config:
        return V3Observation(
            source=GEANT4_RUNTIME_PREFLIGHT_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            not_evaluable_reason="missing_recommended_config",
            message="No recommended Geant4 config is available for runtime preflight.",
        )
    adapter = build_geant4_adapter_from_env(_env_from_call(call))
    server = Geant4McpServer(adapter=adapter)
    runtime_state = server.call_tool(ToolCallRequest(tool_name="get_runtime_state", arguments={}))
    validate_obs = server.call_tool(ToolCallRequest(tool_name="validate_config", arguments={"config": config, "events": events}))
    adapter_kind = _adapter_kind(adapter)
    data = {
        "schema_version": "geant4_agent_v3_runtime_preflight_observation.v1",
        "adapter": adapter_kind,
        "allow_in_memory": allow_in_memory,
        "events": events,
        "runtime_state": _execution_observation_payload(runtime_state),
        "validation": _execution_observation_payload(validate_obs),
        "config_ok": validate_obs.status == RuntimeActionStatus.COMPLETED and bool((validate_obs.payload or {}).get("ok")),
    }
    if not data["config_ok"]:
        return V3Observation(
            source=GEANT4_RUNTIME_PREFLIGHT_TOOL,
            status=V3ObservationStatus.BLOCKED,
            data=data,
            message="Runtime preflight failed because the config is incomplete or invalid.",
        )
    if adapter_kind != "local_process" and not allow_in_memory:
        return V3Observation(
            source=GEANT4_RUNTIME_PREFLIGHT_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            data=data,
            not_evaluable_reason="local_process_runtime_required",
            message="Runtime preflight passed structurally, but no real local-process Geant4 runtime is configured.",
        )
    return V3Observation(
        source=GEANT4_RUNTIME_PREFLIGHT_TOOL,
        status=V3ObservationStatus.OK,
        data=data,
        message="Runtime preflight passed.",
    )


def geant4_runtime_handler(call: V3ToolCall) -> V3Observation:
    config = _config_from_call(call)
    events = _coerce_int(call.arguments.get("events"), _events_from_config(config, 1000))
    allow_in_memory = bool(call.arguments.get("allow_in_memory"))
    if not config:
        return V3Observation(
            source=GEANT4_RUNTIME_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            not_evaluable_reason="missing_recommended_config",
            message="No recommended Geant4 config is available for execution.",
        )
    adapter = build_geant4_adapter_from_env(_env_from_call(call))
    adapter_kind = _adapter_kind(adapter)
    if adapter_kind != "local_process" and not allow_in_memory:
        return V3Observation(
            source=GEANT4_RUNTIME_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            data={"adapter": adapter_kind, "events": events},
            not_evaluable_reason="local_process_runtime_required",
            message="A real local-process Geant4 runtime is required before v3 can report simulation results.",
        )
    server = Geant4McpServer(adapter=adapter)
    validate_obs = server.call_tool(ToolCallRequest(tool_name="validate_config", arguments={"config": config, "events": events}))
    if validate_obs.status != RuntimeActionStatus.COMPLETED or not bool((validate_obs.payload or {}).get("ok")):
        return V3Observation(
            source=GEANT4_RUNTIME_TOOL,
            status=V3ObservationStatus.BLOCKED,
            data={"validation": _execution_observation_payload(validate_obs), "adapter": adapter_kind},
            message="Runtime execution was blocked because validation failed.",
        )
    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": config}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed("apply_config_patch_failed", apply_obs, adapter_kind)
    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed("initialize_run_failed", init_obs, adapter_kind)
    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": events}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed("run_beam_failed", run_obs, adapter_kind)
    summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
    if summary_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed("summarize_last_result_failed", summary_obs, adapter_kind)
    data = {
        "schema_version": "geant4_agent_v3_runtime_observation.v1",
        "adapter": adapter_kind,
        "events": events,
        "runtime_payload": _runtime_payload_from_call(call),
        "validation": _execution_observation_payload(validate_obs),
        "apply": _execution_observation_payload(apply_obs),
        "initialize": _execution_observation_payload(init_obs),
        "run": _execution_observation_payload(run_obs),
        "summary": _execution_observation_payload(summary_obs),
        "result_summary": (summary_obs.payload or {}).get("result_summary") if isinstance(summary_obs.payload, dict) else None,
    }
    artifact_paths = _artifact_paths(summary_obs.payload if isinstance(summary_obs.payload, dict) else {})
    return V3Observation(
        source=GEANT4_RUNTIME_TOOL,
        status=V3ObservationStatus.OK,
        data=data,
        artifact_paths=artifact_paths,
        message="Geant4 runtime execution completed and produced a runtime observation.",
    )


def _config_from_call(call: V3ToolCall) -> dict[str, Any]:
    config = call.arguments.get("recommended_config")
    if isinstance(config, dict) and config:
        return config
    payload = call.arguments.get("payload_builder_observation")
    if isinstance(payload, dict):
        nested = payload.get("recommended_config")
        if isinstance(nested, dict):
            return nested
    return {}


def _runtime_payload_from_call(call: V3ToolCall) -> dict[str, Any]:
    payload = call.arguments.get("payload_builder_observation")
    if isinstance(payload, dict):
        nested = payload.get("runtime_payload")
        if isinstance(nested, dict):
            return nested
    return {}


def _env_from_call(call: V3ToolCall) -> dict[str, str] | None:
    env = call.arguments.get("env")
    if not isinstance(env, dict) or not env:
        return None  # Fall back to os.environ
    return {str(key): str(value) for key, value in env.items()}


def _adapter_kind(adapter: Any) -> str:
    try:
        snapshot = adapter.snapshot()
    except Exception:
        return adapter.__class__.__name__
    metadata = getattr(snapshot, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("adapter"):
        return str(metadata["adapter"])
    if isinstance(adapter, InMemoryGeant4Adapter):
        return "in_memory"
    return adapter.__class__.__name__


def _execution_observation_payload(obs: Any) -> dict[str, Any]:
    status = getattr(getattr(obs, "status", None), "value", str(getattr(obs, "status", "")))
    phase = getattr(getattr(obs, "runtime_phase", None), "value", str(getattr(obs, "runtime_phase", "")))
    return {
        "status": status,
        "message": str(getattr(obs, "message", "") or ""),
        "payload": getattr(obs, "payload", {}) or {},
        "errors": list(getattr(obs, "errors", []) or []),
        "warnings": list(getattr(obs, "warnings", []) or []),
        "runtime_phase": phase,
    }


def _runtime_failed(reason: str, obs: Any, adapter_kind: str) -> V3Observation:
    return V3Observation(
        source=GEANT4_RUNTIME_TOOL,
        status=V3ObservationStatus.FAILED,
        data={"adapter": adapter_kind, "observation": _execution_observation_payload(obs)},
        message=f"Geant4 runtime execution failed: {reason}",
    )


def _artifact_paths(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("artifact_dir", "run_summary_path"):
        value = payload.get(key)
        if value:
            paths.append(str(value))
    return paths


__all__ = [
    "GEANT4_CAPABILITY_TOOL",
    "GEANT4_DESIGN_TEMPLATE_TOOL",
    "GEANT4_PAYLOAD_BUILDER_TOOL",
    "GEANT4_RUNTIME_PREFLIGHT_TOOL",
    "GEANT4_RUNTIME_TOOL",
    "build_default_geant4_tool_registry",
    "build_recommended_config_from_design",
    "geant4_capability_handler",
    "geant4_design_template_handler",
    "geant4_payload_builder_handler",
    "geant4_runtime_handler",
    "geant4_runtime_preflight_handler",
]

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .intent_router import IntentDecision
from .turn_trace import stable_hash


KnowledgeSourceType = Literal[
    "capability",
    "domain_explanation",
    "implementation_contract",
    "runtime_result",
    "deprecated",
    "unsupported",
]


SUPPORTED_GEOMETRY = [
    "single_box",
    "single_tubs",
    "multi_layer_stack",
    "step_wedge",
    "embedded_void",
    "embedded_inclusion",
    "slab_pair",
    "water_phantom",
]
UNSUPPORTED_GEOMETRY = [
    "arbitrary_cad_import",
    "ct_scanner_from_free_text",
    "moving_geometry",
    "free_form_boolean_solids",
]
SUPPORTED_PARTICLES = ["gamma", "electron", "proton", "neutron"]
SUPPORTED_SOURCE_TYPES = ["point", "beam", "isotropic"]
SUPPORTED_SCORING = [
    "target_edep",
    "detector_crossings",
    "plane_crossings",
    "region_contrast",
    "depth_bins",
    "transmission_factor",
]
SUPPORTED_RUNTIME_ACTIONS = ["validate_config", "apply_config_patch", "initialize_run", "run_beam", "summarize_last_result"]
ALLOWED_CONFIG_PATH_PREFIXES = [
    "geometry.",
    "materials.",
    "source.",
    "physics.",
    "physics_list.",
    "output.",
    "simulation.detector.",
    "simulation.run.",
    "scoring.",
]


@dataclass(frozen=True)
class KnowledgeSnippet:
    source_id: str
    source_type: KnowledgeSourceType
    title: str
    content: str
    version: str | None = None
    freshness: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextPack:
    user_turn: str
    intent: str
    current_session_summary: dict[str, Any]
    stable_slots: dict[str, Any]
    staged_patch_summary: dict[str, Any] | None
    allowed_config_paths: list[str]
    supported_capabilities: dict[str, Any]
    unsupported_capabilities: dict[str, Any]
    retrieved_knowledge: list[KnowledgeSnippet] = field(default_factory=list)
    runtime_summary: dict[str, Any] | None = None
    context_pack_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["retrieved_knowledge"] = [snippet.to_dict() for snippet in self.retrieved_knowledge]
        return data


def capability_kb() -> dict[str, Any]:
    return {
        "supported_geometry": list(SUPPORTED_GEOMETRY),
        "supported_particles": list(SUPPORTED_PARTICLES),
        "supported_source_types": list(SUPPORTED_SOURCE_TYPES),
        "supported_scoring": list(SUPPORTED_SCORING),
        "supported_runtime_actions": list(SUPPORTED_RUNTIME_ACTIONS),
    }


def unsupported_kb() -> dict[str, Any]:
    return {
        "unsupported_geometry": list(UNSUPPORTED_GEOMETRY),
        "unsupported_runtime_actions": ["implicit_chat_run", "implicit_chat_viewer_launch"],
        "unsupported_scoring": ["let", "clinical_dose_from_free_text"],
    }


def _session_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry_structure": (config.get("geometry") or {}).get("structure"),
        "materials": list((config.get("materials") or {}).get("selected_materials") or []),
        "source_type": (config.get("source") or {}).get("type"),
        "particle": (config.get("source") or {}).get("particle"),
        "physics_list": (config.get("physics") or {}).get("physics_list")
        or (config.get("physics_list") or {}).get("name"),
        "output_format": (config.get("output") or {}).get("format"),
    }


def _stable_slots(config: dict[str, Any]) -> dict[str, Any]:
    summary = _session_summary(config)
    return {key: value for key, value in summary.items() if value not in (None, "", [])}


def _retrieve_knowledge(intent: str) -> list[KnowledgeSnippet]:
    if intent == "config_mutation":
        return [
            KnowledgeSnippet(
                source_id="capability.runtime_bridge",
                source_type="capability",
                title="Supported runtime bridge capabilities",
                content="Current runtime bridge supports box/tubs/sphere/water phantom basics, common particles, point/beam sources, and target/detector/plane scoring.",
                version="v1",
            ),
            KnowledgeSnippet(
                source_id="unsupported.geometry.free_text_ct",
                source_type="unsupported",
                title="Unsupported free-text CT scanner geometry",
                content="A complete CT scanner cannot be generated from free text unless explicit supported components are implemented.",
                version="v1",
            ),
        ]
    if intent == "read_summary":
        return [
            KnowledgeSnippet(
                source_id="implementation.runtime_smoke_report",
                source_type="implementation_contract",
                title="RuntimeSmokeReport contract",
                content="Result answers must use RuntimeSmokeReport fields and say unavailable when a metric is absent.",
                version="v1",
            )
        ]
    if intent == "read_config":
        return [
            KnowledgeSnippet(
                source_id="implementation.config_summary",
                source_type="implementation_contract",
                title="Config summary contract",
                content="Config summary is read-only and reports current config identity and missing fields.",
                version="v1",
            )
        ]
    return []


def build_context_pack(
    *,
    user_turn: str,
    intent_decision: IntentDecision,
    config: dict[str, Any] | None = None,
    staged_patch_summary: dict[str, Any] | None = None,
    runtime_summary: dict[str, Any] | None = None,
) -> ContextPack:
    cfg = config or {}
    base = {
        "user_turn": str(user_turn or ""),
        "intent": intent_decision.intent,
        "current_session_summary": _session_summary(cfg),
        "stable_slots": _stable_slots(cfg),
        "staged_patch_summary": staged_patch_summary,
        "allowed_config_paths": list(ALLOWED_CONFIG_PATH_PREFIXES),
        "supported_capabilities": capability_kb(),
        "unsupported_capabilities": unsupported_kb(),
        "retrieved_knowledge": [snippet.to_dict() for snippet in _retrieve_knowledge(intent_decision.intent)],
        "runtime_summary": runtime_summary,
    }
    return ContextPack(
        user_turn=base["user_turn"],
        intent=base["intent"],
        current_session_summary=base["current_session_summary"],
        stable_slots=base["stable_slots"],
        staged_patch_summary=staged_patch_summary,
        allowed_config_paths=base["allowed_config_paths"],
        supported_capabilities=base["supported_capabilities"],
        unsupported_capabilities=base["unsupported_capabilities"],
        retrieved_knowledge=_retrieve_knowledge(intent_decision.intent),
        runtime_summary=runtime_summary,
        context_pack_hash=stable_hash(base),
    )


__all__ = [
    "ContextPack",
    "KnowledgeSnippet",
    "build_context_pack",
    "capability_kb",
    "unsupported_kb",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agent.candidate_patch import CandidatePatchEnvelope, PatchEvidence, PatchOperation
from core.agent.turn_trace import stable_hash
from core.orchestrator.types import Intent


DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION = "geant4_agent_design_acceptance_patch.v1"
DESIGN_ACCEPTANCE_ALLOWED_TOP_LEVEL = (
    "geometry",
    "materials",
    "source",
    "physics",
    "output",
    "simulation",
    "scoring",
    "detector",
)


@dataclass(frozen=True)
class DesignAcceptancePatch:
    schema_version: str = DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION
    source: str = "accepted_simulation_design"
    operations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_reasons: list[str] = field(default_factory=list)
    patch_hash: str = ""
    candidate_patch: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "operations": [dict(item) for item in self.operations],
            "evidence": [dict(item) for item in self.evidence],
            "requires_confirmation": self.requires_confirmation,
            "confirmation_reasons": list(self.confirmation_reasons),
            "patch_hash": self.patch_hash,
            "candidate_patch": dict(self.candidate_patch),
        }


def _existing_value(config: dict[str, Any], path: str) -> Any:
    if not isinstance(config, dict):
        return None
    return config.get(path)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _evidence_from_design(design_advice: dict[str, Any] | None, *, source: str) -> list[PatchEvidence]:
    advice = design_advice if isinstance(design_advice, dict) else {}
    text = str(advice.get("user_visible_summary") or advice.get("goal") or source or "accepted design")
    evidence = [PatchEvidence(text=text, source="design_advice", role="accepted_design")]
    primary = advice.get("primary_option") if isinstance(advice.get("primary_option"), dict) else {}
    if primary.get("title"):
        evidence.append(PatchEvidence(text=str(primary["title"]), source="design_advice", role="primary_option"))
    return evidence


def build_design_acceptance_patch(
    recommended_config: dict[str, Any],
    *,
    base_config: dict[str, Any] | None = None,
    design_advice: dict[str, Any] | None = None,
    source: str = "accepted_simulation_design",
    confidence: float = 1.0,
) -> dict[str, Any]:
    base = base_config if isinstance(base_config, dict) else {}
    evidence = _evidence_from_design(design_advice, source=source)
    operations: list[PatchOperation] = []
    confirmation_reasons: list[str] = []
    for key in DESIGN_ACCEPTANCE_ALLOWED_TOP_LEVEL:
        if key not in recommended_config:
            continue
        value = recommended_config.get(key)
        if value is None:
            continue
        old = _existing_value(base, key)
        overwrites_existing = _is_present(old) and old != value
        reasons = ["accepted_design_overwrites_existing_config"] if overwrites_existing else []
        if reasons:
            confirmation_reasons.extend(reasons)
        operations.append(
            PatchOperation(
                path=key,
                op="set",
                value=value,
                confidence=confidence,
                evidence=evidence,
                requires_confirmation=bool(reasons),
                confirmation_reasons=reasons,
            )
        )
    physics_list = recommended_config.get("physics_list")
    if isinstance(physics_list, dict) and physics_list.get("name"):
        operations.append(
            PatchOperation(
                path="physics.physics_list",
                op="set",
                value=str(physics_list["name"]),
                confidence=confidence,
                evidence=evidence,
            )
        )
    elif isinstance(physics_list, str) and physics_list:
        operations.append(
            PatchOperation(
                path="physics.physics_list",
                op="set",
                value=physics_list,
                confidence=confidence,
                evidence=evidence,
            )
        )
    envelope = CandidatePatchEnvelope(
        source=source,
        intent=Intent.SET,
        operations=operations,
        patch_hash="",
    )
    envelope = CandidatePatchEnvelope(
        source=envelope.source,
        intent=envelope.intent,
        operations=envelope.operations,
        guarded_actions=envelope.guarded_actions,
        ambiguities=envelope.ambiguities,
        unsupported_requests=envelope.unsupported_requests,
        patch_hash=stable_hash(envelope.to_dict()),
    )
    patch = DesignAcceptancePatch(
        source=source,
        operations=[
            {
                "path": op.path,
                "op": op.op,
                "confidence": op.confidence,
                "requires_confirmation": op.requires_confirmation,
                "confirmation_reasons": list(op.confirmation_reasons),
            }
            for op in operations
        ],
        evidence=[{"text": item.text, "source": item.source, "role": item.role} for item in evidence],
        requires_confirmation=bool(confirmation_reasons),
        confirmation_reasons=list(dict.fromkeys(confirmation_reasons)),
        patch_hash=envelope.patch_hash,
        candidate_patch=envelope.to_dict(),
    )
    return patch.to_dict()


__all__ = [
    "DESIGN_ACCEPTANCE_ALLOWED_TOP_LEVEL",
    "DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION",
    "DesignAcceptancePatch",
    "build_design_acceptance_patch",
]

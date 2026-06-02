from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import V3AgentState, V3Observation


V3_PATCH_SCHEMA_VERSION = "geant4_agent_v3_patch.v1"
V3_PATCH_APPLY_SCHEMA_VERSION = "geant4_agent_v3_patch_apply.v1"


_ALLOWED_FIELDS = {
    "source_energy_mev",
    "run_events",
    "target_material",
    "target_thickness_mm",
    "geometry_dimensions_mm",
}

_STALE_ARTIFACT_IDS_BY_SOURCE = {
    "geant4_payload_builder_tool": {
        "geant4_runtime_payload_draft",
        "draft_geant4_runtime_payload",
    },
    "geant4_runtime_preflight_tool": {
        "preflight_geant4_runtime",
    },
}

_PATCH_METADATA_KEYS_TO_CLEAR = (
    "pending_action",
    "suggested_next_actions",
    "suggestions",
    "llm_sweep_suggestion",
)

_FIELD_ALIASES = {
    "energy": "source_energy_mev",
    "energy_mev": "source_energy_mev",
    "source.energy": "source_energy_mev",
    "source.energy_mev": "source_energy_mev",
    "source_energy": "source_energy_mev",
    "source_energy_mev": "source_energy_mev",
    "events": "run_events",
    "run.events": "run_events",
    "run_events": "run_events",
    "material": "target_material",
    "target.material": "target_material",
    "target_material": "target_material",
    "thickness": "target_thickness_mm",
    "thickness_mm": "target_thickness_mm",
    "target.thickness_mm": "target_thickness_mm",
    "target_thickness_mm": "target_thickness_mm",
    "geometry.dimensions_mm": "geometry_dimensions_mm",
    "geometry_dimensions_mm": "geometry_dimensions_mm",
}


@dataclass(slots=True)
class V3StatePatch:
    path: str
    op: str
    value: Any
    evidence: str = ""
    source: str = "turn_understanding"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_PATCH_SCHEMA_VERSION,
            "path": self.path,
            "op": self.op,
            "value": self.value,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass(slots=True)
class V3PatchValidationResult:
    patches: list[V3StatePatch]
    config_overrides: dict[str, Any]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_PATCH_SCHEMA_VERSION,
            "ok": self.ok,
            "patches": [patch.to_dict() for patch in self.patches],
            "config_overrides": dict(self.config_overrides),
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class V3PatchApplyResult:
    ok: bool
    applied: bool
    stale_sources: list[str]
    removed_observations: list[dict[str, Any]]
    removed_artifacts: list[str]
    cleared_metadata: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_PATCH_APPLY_SCHEMA_VERSION,
            "ok": self.ok,
            "applied": self.applied,
            "stale_sources": list(self.stale_sources),
            "removed_observations": list(self.removed_observations),
            "removed_artifacts": list(self.removed_artifacts),
            "cleared_metadata": list(self.cleared_metadata),
            "errors": list(self.errors),
        }


def apply_patches_to_state(state: V3AgentState, patch_result: V3PatchValidationResult) -> V3PatchApplyResult:
    if not patch_result.ok or not patch_result.patches:
        return V3PatchApplyResult(
            ok=patch_result.ok,
            applied=False,
            stale_sources=[],
            removed_observations=[],
            removed_artifacts=[],
            cleared_metadata=[],
            errors=list(patch_result.errors),
        )

    stale_sources = stale_sources_for_patches(patch_result.patches)
    removed_observations: list[dict[str, Any]] = []
    kept_observations: list[V3Observation] = []
    for observation in state.observations:
        if observation.source in stale_sources:
            removed_observations.append(_observation_summary(observation))
            continue
        kept_observations.append(observation)
    state.observations = kept_observations

    removed_artifacts: list[str] = []
    for artifact_id in sorted(_stale_artifact_ids(stale_sources)):
        if artifact_id in state.artifacts:
            state.artifacts.pop(artifact_id, None)
            removed_artifacts.append(artifact_id)

    cleared_metadata: list[str] = []
    for key in _PATCH_METADATA_KEYS_TO_CLEAR:
        if key in state.metadata:
            state.metadata.pop(key, None)
            cleared_metadata.append(key)

    applied = bool(removed_observations or removed_artifacts or cleared_metadata or patch_result.config_overrides)
    result = V3PatchApplyResult(
        ok=True,
        applied=applied,
        stale_sources=sorted(stale_sources),
        removed_observations=removed_observations,
        removed_artifacts=removed_artifacts,
        cleared_metadata=cleared_metadata,
        errors=[],
    )
    state.metadata["last_state_patch_apply"] = result.to_dict()
    return result


def build_patches_from_config_overrides(overrides: Any, *, evidence: str = "config_overrides") -> V3PatchValidationResult:
    if not isinstance(overrides, dict) or not overrides:
        return V3PatchValidationResult(patches=[], config_overrides={}, errors=[])
    patches: list[V3StatePatch] = []
    normalized: dict[str, Any] = {}
    errors: list[str] = []
    for key, raw_value in overrides.items():
        field = canonical_patch_field(str(key))
        if field not in _ALLOWED_FIELDS:
            errors.append(f"unsupported_patch_field:{field}")
            continue
        ok, value_or_error = _normalize_value(field, raw_value)
        if not ok:
            errors.append(f"invalid_patch_value:{field}:{value_or_error}")
            continue
        normalized[field] = value_or_error
        patches.append(
            V3StatePatch(
                path=field,
                op="replace",
                value=value_or_error,
                evidence=evidence,
            )
        )
    if errors:
        return V3PatchValidationResult(patches=[], config_overrides={}, errors=errors)
    return V3PatchValidationResult(patches=patches, config_overrides=normalized, errors=errors)


def build_patches_from_requested_changes(changes: Any, *, evidence: str = "requested_changes") -> V3PatchValidationResult:
    if not isinstance(changes, list) or not changes:
        return V3PatchValidationResult(patches=[], config_overrides={}, errors=[])
    overrides: dict[str, Any] = {}
    errors: list[str] = []
    for item in changes:
        if not isinstance(item, dict):
            errors.append("invalid_requested_change:not_object")
            continue
        field = canonical_patch_field(str(item.get("field") or item.get("path") or item.get("parameter") or ""))
        if not field:
            errors.append("invalid_requested_change:missing_field")
            continue
        if "value" not in item:
            errors.append(f"invalid_requested_change:{field}:missing_value")
            continue
        overrides[field] = item.get("value")
    result = build_patches_from_config_overrides(overrides, evidence=evidence)
    return V3PatchValidationResult(
        patches=result.patches,
        config_overrides=result.config_overrides,
        errors=[*errors, *result.errors],
    )


def config_overrides_from_llm_parameters(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in params.items():
        field = canonical_patch_field(str(key))
        if field in _ALLOWED_FIELDS:
            out[field] = value
    return out


def canonical_patch_field(field: str) -> str:
    normalized = str(field or "").strip()
    return _FIELD_ALIASES.get(normalized, normalized)


def stale_sources_for_patches(patches: list[V3StatePatch]) -> set[str]:
    if not patches:
        return set()
    return {"geant4_payload_builder_tool", "geant4_runtime_preflight_tool", "geant4_runtime_tool", "commit_gate"}


def _stale_artifact_ids(stale_sources: set[str]) -> set[str]:
    artifact_ids: set[str] = set()
    for source in stale_sources:
        artifact_ids.update(_STALE_ARTIFACT_IDS_BY_SOURCE.get(source, set()))
    return artifact_ids


def _observation_summary(observation: V3Observation) -> dict[str, Any]:
    return {
        "source": observation.source,
        "status": observation.status.value,
        "message": observation.message,
    }


def _normalize_value(field: str, value: Any) -> tuple[bool, Any]:
    if field == "source_energy_mev":
        return _positive_float(value)
    if field == "target_thickness_mm":
        return _positive_float(value)
    if field == "run_events":
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return False, "not_integer"
        if parsed <= 0:
            return False, "must_be_positive"
        return True, parsed
    if field == "target_material":
        material = str(value or "").strip()
        if not material:
            return False, "empty"
        if not material.startswith("G4_") or any(ch.isspace() for ch in material):
            return False, "must_be_geant4_material_id"
        return True, material
    if field == "geometry_dimensions_mm":
        if not isinstance(value, list) or len(value) != 3:
            return False, "must_be_three_item_list"
        dims: list[float] = []
        for item in value:
            ok, parsed = _positive_float(item)
            if not ok:
                return False, parsed
            dims.append(parsed)
        return True, dims
    return False, "unsupported"


def _positive_float(value: Any) -> tuple[bool, Any]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False, "not_number"
    if parsed <= 0:
        return False, "must_be_positive"
    return True, parsed


__all__ = [
    "V3_PATCH_APPLY_SCHEMA_VERSION",
    "V3_PATCH_SCHEMA_VERSION",
    "V3PatchApplyResult",
    "V3PatchValidationResult",
    "V3StatePatch",
    "apply_patches_to_state",
    "build_patches_from_config_overrides",
    "build_patches_from_requested_changes",
    "canonical_patch_field",
    "config_overrides_from_llm_parameters",
    "stale_sources_for_patches",
]

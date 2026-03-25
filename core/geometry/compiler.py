from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.contracts.semantic import SemanticFrame
from core.contracts.slots import SlotFrame

from core.geometry.catalog import GeometryCatalogEntry, get_geometry_catalog_entry, resolve_geometry_structure
from core.geometry.spec import GeometryEvidence, GeometryFieldResolution, GeometryIntent, GeometrySpec


@dataclass(frozen=True)
class GeometryCompileResult:
    intent: GeometryIntent
    spec: GeometrySpec | None
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.spec is not None and not self.errors


def _append_param_if_present(
    *,
    params: dict[str, Any],
    evidence: list[GeometryEvidence],
    field_resolutions: dict[str, GeometryFieldResolution],
    param_def: GeometryCatalogEntry | Any,
    field_name: str,
    value: Any,
    source: str,
    confidence: float,
    detail: str = "",
    status: str = "derived",
) -> None:
    normalized = param_def.normalize_value(value)
    if normalized is None:
        return
    params[field_name] = normalized
    evidence.append(
        GeometryEvidence(
            source=source,
            field=field_name,
            value=normalized,
            confidence=confidence,
            detail=detail,
        )
    )
    field_resolutions[field_name] = GeometryFieldResolution(
        field=field_name,
        value=normalized,
        status=status,
        evidence_sources=(source,),
        note=detail,
    )


def _collect_slot_intent_params(entry: GeometryCatalogEntry, frame: SlotFrame) -> tuple[dict[str, Any], list[GeometryEvidence]]:
    params: dict[str, Any] = {}
    evidence: list[GeometryEvidence] = []
    field_resolutions: dict[str, GeometryFieldResolution] = {}
    confidence = float(frame.confidence or 0.8)
    for param in entry.params:
        for slot_field in param.slot_fields:
            value = getattr(frame.geometry, slot_field, None)
            if value is None:
                continue
            _append_param_if_present(
                params=params,
                evidence=evidence,
                field_resolutions=field_resolutions,
                param_def=param,
                field_name=param.name,
                value=value,
                source="slot_frame",
                confidence=confidence,
                detail=slot_field,
                status="user_explicit",
            )
            break
    return params, evidence, field_resolutions


def build_geometry_intent_from_slot_frame(frame: SlotFrame) -> GeometryIntent:
    structure = resolve_geometry_structure(frame.geometry.kind)
    entry = get_geometry_catalog_entry(structure)
    params, evidence, field_resolutions = (
        _collect_slot_intent_params(entry, frame) if entry is not None else ({}, [], {})
    )
    intent = GeometryIntent(
        structure=structure,
        kind=frame.geometry.kind,
        params=params,
        evidence=evidence,
        field_resolutions=field_resolutions,
    )
    if entry is None:
        if frame.geometry.kind:
            intent.ambiguities.append(f"unsupported_geometry_kind:{frame.geometry.kind}")
        return intent
    for required_field in entry.required_slot_fields:
        if required_field == "kind":
            if not frame.geometry.kind:
                intent.missing_fields.append(required_field)
        elif required_field not in params:
            intent.missing_fields.append(required_field)
    return intent


def build_geometry_intent_from_semantic_frame(frame: SemanticFrame) -> GeometryIntent:
    structure = resolve_geometry_structure(frame.geometry.structure)
    entry = get_geometry_catalog_entry(structure)
    params: dict[str, Any] = {}
    evidence: list[GeometryEvidence] = []
    field_resolutions: dict[str, GeometryFieldResolution] = {}
    if entry is not None:
        for param in entry.params:
            if param.value_kind == "triplet" and param.arity == 3 and len(param.config_param_keys) == 3:
                values = [frame.geometry.params.get(key) for key in param.config_param_keys]
                if all(value is not None for value in values):
                    _append_param_if_present(
                        params=params,
                        evidence=evidence,
                        field_resolutions=field_resolutions,
                        param_def=param,
                        field_name=param.name,
                        value=values,
                        source="semantic_frame",
                        confidence=1.0,
                        detail=",".join(param.config_param_keys),
                        status="carried_forward",
                    )
                    continue
            for key in param.config_param_keys:
                value = frame.geometry.params.get(key)
                if value is None:
                    continue
                _append_param_if_present(
                    params=params,
                    evidence=evidence,
                    field_resolutions=field_resolutions,
                    param_def=param,
                    field_name=param.name,
                    value=value,
                    source="semantic_frame",
                    confidence=1.0,
                    detail=key,
                    status="carried_forward",
                )
                break

    intent = GeometryIntent(
        structure=structure,
        kind=frame.geometry.structure,
        params=params,
        evidence=evidence,
        field_resolutions=field_resolutions,
    )
    if entry is None:
        if frame.geometry.structure:
            intent.ambiguities.append(f"unsupported_geometry_structure:{frame.geometry.structure}")
        return intent
    for required_field in entry.required_slot_fields:
        if required_field == "kind":
            continue
        if required_field not in params:
            intent.missing_fields.append(required_field)
    return intent


def build_geometry_intent_from_config(config: dict[str, Any]) -> GeometryIntent:
    geometry = config.get("geometry", {}) if isinstance(config.get("geometry"), dict) else {}
    structure = resolve_geometry_structure(geometry.get("structure"))
    entry = get_geometry_catalog_entry(structure)
    params_blob = geometry.get("params", {}) if isinstance(geometry.get("params"), dict) else {}
    params: dict[str, Any] = {}
    evidence: list[GeometryEvidence] = []
    field_resolutions: dict[str, GeometryFieldResolution] = {}
    if entry is not None:
        for param in entry.params:
            direct_value = geometry.get(param.name)
            if direct_value is not None:
                _append_param_if_present(
                    params=params,
                    evidence=evidence,
                    field_resolutions=field_resolutions,
                    param_def=param,
                    field_name=param.name,
                    value=direct_value,
                    source="config",
                    confidence=1.0,
                    detail=param.name,
                    status="carried_forward",
                )
                continue
            if param.value_kind == "triplet" and param.arity == 3 and len(param.config_param_keys) == 3:
                values = [params_blob.get(key) for key in param.config_param_keys]
                if all(value is not None for value in values):
                    _append_param_if_present(
                        params=params,
                        evidence=evidence,
                        field_resolutions=field_resolutions,
                        param_def=param,
                        field_name=param.name,
                        value=values,
                        source="config",
                        confidence=1.0,
                        detail=",".join(param.config_param_keys),
                        status="carried_forward",
                    )
                    continue
            for key in param.config_param_keys:
                value = params_blob.get(key)
                if value is None:
                    continue
                _append_param_if_present(
                    params=params,
                    evidence=evidence,
                    field_resolutions=field_resolutions,
                    param_def=param,
                    field_name=param.name,
                    value=value,
                    source="config",
                    confidence=1.0,
                    detail=key,
                    status="carried_forward",
                )
                break

    intent = GeometryIntent(
        structure=structure,
        kind=geometry.get("structure"),
        params=params,
        evidence=evidence,
        field_resolutions=field_resolutions,
    )
    if entry is None:
        if geometry.get("structure"):
            intent.ambiguities.append(f"unsupported_geometry_structure:{geometry.get('structure')}")
        return intent
    for required_field in entry.required_slot_fields:
        if required_field == "kind":
            continue
        if required_field not in params:
            intent.missing_fields.append(required_field)
    return intent


def compile_geometry_intent(intent: GeometryIntent) -> GeometryCompileResult:
    if not intent.structure:
        return GeometryCompileResult(
            intent=intent,
            spec=None,
            errors=("missing_geometry_structure",),
        )
    entry = get_geometry_catalog_entry(intent.structure)
    if entry is None:
        return GeometryCompileResult(
            intent=intent,
            spec=None,
            errors=(f"unsupported_geometry_structure:{intent.structure}",),
        )

    missing_fields = list(intent.missing_fields)
    for param in entry.params:
        if param.required and param.name not in intent.params and param.name not in missing_fields:
            missing_fields.append(param.name)
    if missing_fields:
        return GeometryCompileResult(
            intent=intent,
            spec=None,
            missing_fields=tuple(missing_fields),
        )

    spec = GeometrySpec(
        structure=entry.structure,
        params=dict(intent.params),
        allowed_paths=entry.allowed_paths,
        required_paths=entry.required_paths,
        confidence=max((item.confidence for item in intent.evidence), default=1.0),
        finalization_status="ready" if not intent.ambiguities else "ambiguous",
        field_resolutions=dict(intent.field_resolutions),
    )
    return GeometryCompileResult(intent=intent, spec=spec)


def compile_geometry_spec_from_slot_frame(frame: SlotFrame) -> GeometryCompileResult:
    intent = build_geometry_intent_from_slot_frame(frame)
    return compile_geometry_intent(intent)


def compile_geometry_spec_from_semantic_frame(frame: SemanticFrame) -> GeometryCompileResult:
    intent = build_geometry_intent_from_semantic_frame(frame)
    return compile_geometry_intent(intent)


def compile_geometry_spec_from_config(config: dict[str, Any]) -> GeometryCompileResult:
    intent = build_geometry_intent_from_config(config)
    return compile_geometry_intent(intent)

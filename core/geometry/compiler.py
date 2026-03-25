from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.contracts.semantic import SemanticFrame
from core.contracts.slots import SlotFrame

from core.geometry.catalog import get_geometry_catalog_entry, resolve_geometry_structure
from core.geometry.spec import GeometryEvidence, GeometryIntent, GeometrySpec


@dataclass(frozen=True)
class GeometryCompileResult:
    intent: GeometryIntent
    spec: GeometrySpec | None
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.spec is not None and not self.errors


def _float_triplet(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    return [float(values[0]), float(values[1]), float(values[2])]


def _collect_intent_params(structure: str, frame: SlotFrame) -> tuple[dict[str, Any], list[GeometryEvidence]]:
    params: dict[str, Any] = {}
    evidence: list[GeometryEvidence] = []

    def add(field: str, value: Any) -> None:
        params[field] = value
        evidence.append(
            GeometryEvidence(
                source="slot_frame",
                field=field,
                value=value,
                confidence=float(frame.confidence or 0.8),
            )
        )

    if structure == "single_box":
        triplet = _float_triplet(frame.geometry.size_triplet_mm)
        if triplet:
            add("size_triplet_mm", triplet)
    elif structure == "single_tubs":
        if frame.geometry.radius_mm is not None:
            add("radius_mm", float(frame.geometry.radius_mm))
        if frame.geometry.half_length_mm is not None:
            add("half_length_mm", float(frame.geometry.half_length_mm))

    return params, evidence


def build_geometry_intent_from_slot_frame(frame: SlotFrame) -> GeometryIntent:
    structure = resolve_geometry_structure(frame.geometry.kind)
    params, evidence = _collect_intent_params(structure or "", frame)
    intent = GeometryIntent(
        structure=structure,
        kind=frame.geometry.kind,
        params=params,
        evidence=evidence,
    )
    entry = get_geometry_catalog_entry(structure)
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
    params: dict[str, Any] = {}
    evidence: list[GeometryEvidence] = []

    def add(field: str, value: Any) -> None:
        params[field] = value
        evidence.append(
            GeometryEvidence(
                source="semantic_frame",
                field=field,
                value=value,
                confidence=1.0,
            )
        )

    if structure == "single_box":
        triplet = [
            frame.geometry.params.get("module_x"),
            frame.geometry.params.get("module_y"),
            frame.geometry.params.get("module_z"),
        ]
        if all(value is not None for value in triplet):
            add("size_triplet_mm", [float(triplet[0]), float(triplet[1]), float(triplet[2])])
    elif structure == "single_tubs":
        radius = frame.geometry.params.get("child_rmax")
        half_length = frame.geometry.params.get("child_hz")
        if radius is not None:
            add("radius_mm", float(radius))
        if half_length is not None:
            add("half_length_mm", float(half_length))

    intent = GeometryIntent(
        structure=structure,
        kind=frame.geometry.structure,
        params=params,
        evidence=evidence,
    )
    entry = get_geometry_catalog_entry(structure)
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
    params_blob = geometry.get("params", {}) if isinstance(geometry.get("params"), dict) else {}
    params: dict[str, Any] = {}
    evidence: list[GeometryEvidence] = []

    def add(field: str, value: Any) -> None:
        params[field] = value
        evidence.append(
            GeometryEvidence(
                source="config",
                field=field,
                value=value,
                confidence=1.0,
            )
        )

    if structure == "single_box":
        size_triplet = geometry.get("size_triplet_mm")
        triplet = _float_triplet(size_triplet)
        if triplet:
            add("size_triplet_mm", triplet)
        else:
            if all(key in params_blob for key in ("module_x", "module_y", "module_z")):
                add(
                    "size_triplet_mm",
                    [
                        float(params_blob["module_x"]),
                        float(params_blob["module_y"]),
                        float(params_blob["module_z"]),
                    ],
                )
    elif structure == "single_tubs":
        if geometry.get("radius_mm") is not None:
            add("radius_mm", float(geometry["radius_mm"]))
        elif params_blob.get("child_rmax") is not None:
            add("radius_mm", float(params_blob["child_rmax"]))
        if geometry.get("half_length_mm") is not None:
            add("half_length_mm", float(geometry["half_length_mm"]))
        elif params_blob.get("child_hz") is not None:
            add("half_length_mm", float(params_blob["child_hz"]))

    intent = GeometryIntent(
        structure=structure,
        kind=geometry.get("structure"),
        params=params,
        evidence=evidence,
    )
    entry = get_geometry_catalog_entry(structure)
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

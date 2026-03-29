from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.geometry.catalog import get_geometry_catalog_entry, resolve_geometry_structure
from core.geometry.spec import GeometryEvidence, GeometryFieldResolution, GeometryIntent
from core.interpreter.merged import MergedField, MergedGeometry


@dataclass(frozen=True)
class GeometryResolutionDraft:
    structure: str | None = None
    material: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    ambiguities: tuple[str, ...] = field(default_factory=tuple)
    open_questions: tuple[str, ...] = field(default_factory=tuple)
    trust_report: dict[str, Any] = field(default_factory=dict)


def _normalize_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _field_value(field: MergedField | None) -> Any:
    if not isinstance(field, MergedField):
        return None
    return field.value


def _field_source(field: MergedField | None) -> str:
    if not isinstance(field, MergedField):
        return ""
    return str(field.chosen_from or "")


def _build_geometry_evidence(field: str, value: Any, source: str, note: str) -> GeometryEvidence:
    return GeometryEvidence(
        source=source or "resolver",
        field=field,
        value=value,
        confidence=1.0 if source in {"shared", "evidence"} else 0.8,
        detail=note,
    )


def _build_geometry_resolution(field: str, value: Any, source: str, note: str) -> GeometryFieldResolution:
    status = "derived" if source == "llm" else "carried_forward"
    if source == "shared":
        status = "confirmed"
    return GeometryFieldResolution(
        field=field,
        value=value,
        status=status,
        evidence_sources=((source,) if source else ()),
        note=note,
    )


def resolve_geometry_from_merged(merged_geometry: MergedGeometry) -> GeometryResolutionDraft:
    conflicts = []
    ambiguities = list(merged_geometry.ambiguities)
    open_questions: list[str] = []
    params: dict[str, Any] = {}
    trust_report = {
        "kind_source": _field_source(merged_geometry.kind),
        "material_source": _field_source(merged_geometry.material),
    }

    kind_value = _field_value(merged_geometry.kind)
    if bool(getattr(merged_geometry.kind, "conflict", False)):
        conflicts.append("geometry.kind")
    structure = resolve_geometry_structure(kind_value)

    dims = merged_geometry.dimensions or {}
    side_length = _normalize_scalar(_field_value(dims.get("side_length_mm")))
    size_triplet = _field_value(dims.get("size_triplet_mm"))
    radius = _normalize_scalar(_field_value(dims.get("radius_mm")))
    diameter = _normalize_scalar(_field_value(dims.get("diameter_mm")))
    half_length = _normalize_scalar(_field_value(dims.get("half_length_mm")))
    full_length = _normalize_scalar(_field_value(dims.get("full_length_mm")))
    thickness = _normalize_scalar(_field_value(dims.get("thickness_mm")))

    if isinstance(size_triplet, list) and len(size_triplet) == 3 and all(v is not None for v in size_triplet):
        params["size_triplet_mm"] = [float(size_triplet[0]), float(size_triplet[1]), float(size_triplet[2])]
    elif side_length is not None:
        params["size_triplet_mm"] = [side_length, side_length, side_length]

    if diameter is not None and radius is None:
        radius = diameter / 2.0
    if full_length is not None and half_length is None:
        half_length = full_length / 2.0

    if radius is not None:
        params["radius_mm"] = radius
    if half_length is not None:
        params["half_length_mm"] = half_length

    if structure == "single_box" and "radius_mm" in params and "half_length_mm" in params:
        structure = "single_tubs"

    if structure == "single_box" and "size_triplet_mm" not in params:
        if thickness is not None:
            ambiguities.append("box_thickness_without_face_dimensions")
        open_questions.append("geometry.size_triplet_mm")
    elif structure == "single_tubs":
        if "radius_mm" not in params:
            open_questions.append("geometry.radius_mm")
        if "half_length_mm" not in params:
            open_questions.append("geometry.half_length_mm")
    elif structure is None:
        if radius is not None and half_length is not None:
            structure = "single_tubs"
            params["radius_mm"] = radius
            params["half_length_mm"] = half_length
        elif "size_triplet_mm" in params:
            structure = "single_box"
        elif thickness is not None:
            ambiguities.append("geometry_kind_unresolved_from_thickness_only")
            open_questions.append("geometry.kind")
        else:
            open_questions.append("geometry.kind")

    material = _field_value(merged_geometry.material)
    if bool(getattr(merged_geometry.material, "conflict", False)):
        conflicts.append("geometry.material")

    if structure:
        entry = get_geometry_catalog_entry(structure)
        if entry is not None:
            if structure == "single_box" and "size_triplet_mm" not in params:
                open_questions.append("geometry.size_triplet_mm")
            if structure == "single_tubs":
                if "radius_mm" not in params:
                    open_questions.append("geometry.radius_mm")
                if "half_length_mm" not in params:
                    open_questions.append("geometry.half_length_mm")

    return GeometryResolutionDraft(
        structure=structure,
        material=str(material) if material else None,
        params=params,
        conflicts=tuple(dict.fromkeys(conflicts)),
        ambiguities=tuple(dict.fromkeys(item for item in ambiguities if item)),
        open_questions=tuple(dict.fromkeys(item for item in open_questions if item)),
        trust_report=trust_report,
    )


def build_geometry_intent_from_resolved_draft(draft: GeometryResolutionDraft) -> GeometryIntent:
    evidence: list[GeometryEvidence] = []
    field_resolutions: dict[str, GeometryFieldResolution] = {}

    if draft.structure:
        evidence.append(_build_geometry_evidence("kind", draft.structure, "resolver", "resolved structure"))
        field_resolutions["kind"] = _build_geometry_resolution("kind", draft.structure, "resolver", "resolved structure")

    if draft.material:
        evidence.append(_build_geometry_evidence("material", draft.material, "resolver", "resolved material"))
        field_resolutions["material"] = _build_geometry_resolution("material", draft.material, "resolver", "resolved material")

    for key, value in draft.params.items():
        evidence.append(_build_geometry_evidence(key, value, "resolver", key))
        field_resolutions[key] = _build_geometry_resolution(key, value, "resolver", key)

    return GeometryIntent(
        structure=draft.structure,
        kind=draft.structure,
        params=dict(draft.params),
        evidence=evidence,
        missing_fields=list(draft.open_questions),
        ambiguities=list(draft.ambiguities),
        field_resolutions=field_resolutions,
    )

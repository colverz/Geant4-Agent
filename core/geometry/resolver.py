from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.contracts.slots import SlotFrame
from core.geometry.catalog import get_geometry_catalog_entry, resolve_geometry_structure
from core.geometry.spec import GeometryEvidence, GeometryFieldResolution, GeometryIntent
from core.interpreter.merged import MergedField, MergedGeometry
from core.orchestrator.types import Intent


@dataclass(frozen=True)
class GeometryResolutionDraft:
    structure: str | None = None
    material: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    ambiguities: tuple[str, ...] = field(default_factory=tuple)
    open_questions: tuple[str, ...] = field(default_factory=tuple)
    trust_report: dict[str, Any] = field(default_factory=dict)
    bridge_allowed: bool = False


@dataclass(frozen=True)
class GeometryBridgeSeed:
    kind: str | None = None
    material: str | None = None
    size_triplet_mm: list[float] | None = None
    radius_mm: float | None = None
    half_length_mm: float | None = None


@dataclass(frozen=True)
class GeometrySignals:
    kind_value: str | None = None
    kind_source: str = ""
    kind_conflict: bool = False
    material_value: str | None = None
    material_source: str = ""
    material_conflict: bool = False
    size_triplet_mm: list[float] | None = None
    side_length_mm: float | None = None
    radius_mm: float | None = None
    diameter_mm: float | None = None
    half_length_mm: float | None = None
    full_length_mm: float | None = None
    thickness_mm: float | None = None
    ambiguities: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeometrySignalAssessment:
    structure_candidate: str | None = None
    normalized_params: dict[str, Any] = field(default_factory=dict)
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


def collect_geometry_signals(merged_geometry: MergedGeometry) -> GeometrySignals:
    dims = merged_geometry.dimensions or {}
    raw_triplet = _field_value(dims.get("size_triplet_mm"))
    size_triplet = None
    if isinstance(raw_triplet, list) and len(raw_triplet) == 3 and all(v is not None for v in raw_triplet):
        size_triplet = [float(raw_triplet[0]), float(raw_triplet[1]), float(raw_triplet[2])]

    material_value = _field_value(merged_geometry.material)
    material = str(material_value) if material_value else None

    return GeometrySignals(
        kind_value=_field_value(merged_geometry.kind),
        kind_source=_field_source(merged_geometry.kind),
        kind_conflict=bool(getattr(merged_geometry.kind, "conflict", False)),
        material_value=material,
        material_source=_field_source(merged_geometry.material),
        material_conflict=bool(getattr(merged_geometry.material, "conflict", False)),
        size_triplet_mm=size_triplet,
        side_length_mm=_normalize_scalar(_field_value(dims.get("side_length_mm"))),
        radius_mm=_normalize_scalar(_field_value(dims.get("radius_mm"))),
        diameter_mm=_normalize_scalar(_field_value(dims.get("diameter_mm"))),
        half_length_mm=_normalize_scalar(_field_value(dims.get("half_length_mm"))),
        full_length_mm=_normalize_scalar(_field_value(dims.get("full_length_mm"))),
        thickness_mm=_normalize_scalar(_field_value(dims.get("thickness_mm"))),
        ambiguities=tuple(dict.fromkeys(merged_geometry.ambiguities)),
    )


def assess_geometry_signals(signals: GeometrySignals) -> GeometrySignalAssessment:
    conflicts: list[str] = []
    ambiguities = list(signals.ambiguities)
    open_questions: list[str] = []
    params: dict[str, Any] = {}
    trust_report = {
        "kind_source": signals.kind_source,
        "material_source": signals.material_source,
    }

    if signals.kind_conflict:
        conflicts.append("geometry.kind")
    if signals.material_conflict:
        conflicts.append("geometry.material")

    structure = resolve_geometry_structure(signals.kind_value)

    if signals.size_triplet_mm is not None:
        params["size_triplet_mm"] = list(signals.size_triplet_mm)
    elif signals.side_length_mm is not None:
        params["size_triplet_mm"] = [
            signals.side_length_mm,
            signals.side_length_mm,
            signals.side_length_mm,
        ]

    radius = signals.radius_mm
    if radius is None and signals.diameter_mm is not None:
        radius = signals.diameter_mm / 2.0

    half_length = signals.half_length_mm
    if half_length is None and signals.full_length_mm is not None:
        half_length = signals.full_length_mm / 2.0

    if radius is not None:
        params["radius_mm"] = radius
    if half_length is not None:
        params["half_length_mm"] = half_length

    # Let strong cylinder dimensions override a conflicting box-like kind.
    if structure == "single_box" and "radius_mm" in params and "half_length_mm" in params:
        structure = "single_tubs"

    if structure == "single_box":
        if "size_triplet_mm" not in params:
            if signals.thickness_mm is not None:
                ambiguities.append("box_thickness_without_face_dimensions")
            open_questions.append("geometry.size_triplet_mm")
    elif structure == "single_tubs":
        if "radius_mm" not in params:
            open_questions.append("geometry.radius_mm")
        if "half_length_mm" not in params:
            open_questions.append("geometry.half_length_mm")
    elif structure is None:
        if "radius_mm" in params and "half_length_mm" in params:
            structure = "single_tubs"
        elif "size_triplet_mm" in params:
            structure = "single_box"
        elif signals.thickness_mm is not None:
            ambiguities.append("geometry_kind_unresolved_from_thickness_only")
            open_questions.append("geometry.kind")
        else:
            open_questions.append("geometry.kind")

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

    return GeometrySignalAssessment(
        structure_candidate=structure,
        normalized_params=params,
        conflicts=tuple(dict.fromkeys(conflicts)),
        ambiguities=tuple(dict.fromkeys(item for item in ambiguities if item)),
        open_questions=tuple(dict.fromkeys(item for item in open_questions if item)),
        trust_report=trust_report,
    )


def select_geometry_draft(
    signals: GeometrySignals,
    assessment: GeometrySignalAssessment,
) -> GeometryResolutionDraft:
    structure = assessment.structure_candidate
    bridge_allowed = structure in {"single_box", "single_tubs"} and "geometry.kind" not in assessment.conflicts
    return GeometryResolutionDraft(
        structure=structure,
        material=signals.material_value,
        params=dict(assessment.normalized_params),
        conflicts=tuple(assessment.conflicts),
        ambiguities=tuple(assessment.ambiguities),
        open_questions=tuple(assessment.open_questions),
        trust_report=dict(assessment.trust_report),
        bridge_allowed=bridge_allowed,
    )


def resolve_geometry_with_trace(
    merged_geometry: MergedGeometry,
) -> tuple[GeometrySignals, GeometrySignalAssessment, GeometryResolutionDraft]:
    signals = collect_geometry_signals(merged_geometry)
    assessment = assess_geometry_signals(signals)
    draft = select_geometry_draft(signals, assessment)
    return signals, assessment, draft


def resolve_geometry_from_merged(merged_geometry: MergedGeometry) -> GeometryResolutionDraft:
    _, _, draft = resolve_geometry_with_trace(merged_geometry)
    return draft


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


def geometry_resolution_to_payload(draft: GeometryResolutionDraft) -> dict[str, Any]:
    intent = build_geometry_intent_from_resolved_draft(draft)
    return {
        "draft": {
            "structure": draft.structure,
            "material": draft.material,
            "params": dict(draft.params),
            "conflicts": list(draft.conflicts),
            "ambiguities": list(draft.ambiguities),
            "open_questions": list(draft.open_questions),
            "trust_report": dict(draft.trust_report),
            "bridge_allowed": draft.bridge_allowed,
        },
        "intent": {
            "structure": intent.structure,
            "kind": intent.kind,
            "params": dict(intent.params),
            "missing_fields": list(intent.missing_fields),
            "ambiguities": list(intent.ambiguities),
        },
    }


def build_geometry_bridge_seed(
    *,
    draft: GeometryResolutionDraft | None,
    merged_geometry_payload: dict[str, Any] | None = None,
) -> GeometryBridgeSeed:
    kind: str | None = None
    material: str | None = None
    size_triplet: list[float] | None = None
    radius: float | None = None
    half_length: float | None = None

    if isinstance(draft, GeometryResolutionDraft) and draft.bridge_allowed:
        if draft.structure == "single_box":
            kind = "box"
        elif draft.structure == "single_tubs":
            kind = "cylinder"
        elif draft.structure:
            kind = draft.structure
        material = draft.material
        raw_triplet = draft.params.get("size_triplet_mm")
        if isinstance(raw_triplet, list) and len(raw_triplet) == 3 and all(v is not None for v in raw_triplet):
            size_triplet = [float(raw_triplet[0]), float(raw_triplet[1]), float(raw_triplet[2])]
        radius = _normalize_scalar(draft.params.get("radius_mm"))
        half_length = _normalize_scalar(draft.params.get("half_length_mm"))

    if isinstance(merged_geometry_payload, dict) and draft is None:
        if kind is None:
            field = merged_geometry_payload.get("kind")
            if isinstance(field, dict) and not field.get("conflict"):
                value = field.get("value")
                if isinstance(value, str) and value:
                    kind = value
        if material is None:
            field = merged_geometry_payload.get("material")
            if isinstance(field, dict) and not field.get("conflict"):
                value = field.get("value")
                if isinstance(value, str) and value:
                    material = value
        dimensions = merged_geometry_payload.get("dimensions")
        if isinstance(dimensions, dict):
            if size_triplet is None:
                field = dimensions.get("size_triplet_mm")
                if isinstance(field, dict) and not field.get("conflict"):
                    value = field.get("value")
                    if isinstance(value, list) and len(value) == 3 and all(v is not None for v in value):
                        size_triplet = [float(value[0]), float(value[1]), float(value[2])]
            if size_triplet is None:
                field = dimensions.get("side_length_mm")
                if isinstance(field, dict) and not field.get("conflict"):
                    value = _normalize_scalar(field.get("value"))
                    if value is not None and kind == "box":
                        size_triplet = [value, value, value]
            if radius is None:
                field = dimensions.get("radius_mm")
                if isinstance(field, dict) and not field.get("conflict"):
                    radius = _normalize_scalar(field.get("value"))
            if half_length is None:
                field = dimensions.get("half_length_mm")
                if isinstance(field, dict) and not field.get("conflict"):
                    half_length = _normalize_scalar(field.get("value"))

    return GeometryBridgeSeed(
        kind=kind,
        material=material,
        size_triplet_mm=size_triplet,
        radius_mm=radius,
        half_length_mm=half_length,
    )


def build_slot_frame_from_geometry_bridge_seed(
    seed: GeometryBridgeSeed,
    *,
    intent: Intent,
    confidence: float,
) -> SlotFrame:
    frame = SlotFrame(intent=intent, confidence=confidence)
    if isinstance(seed.kind, str) and seed.kind:
        frame.geometry.kind = seed.kind
    if isinstance(seed.material, str) and seed.material:
        frame.materials.primary = seed.material
    if isinstance(seed.size_triplet_mm, list) and len(seed.size_triplet_mm) == 3:
        frame.geometry.size_triplet_mm = [
            float(seed.size_triplet_mm[0]),
            float(seed.size_triplet_mm[1]),
            float(seed.size_triplet_mm[2]),
        ]
    if seed.radius_mm is not None:
        frame.geometry.radius_mm = float(seed.radius_mm)
    if seed.half_length_mm is not None:
        frame.geometry.half_length_mm = float(seed.half_length_mm)
    return frame

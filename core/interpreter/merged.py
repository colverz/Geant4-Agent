from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.interpreter.spec import GeometryCandidate, SourceCandidate, TurnSummary


@dataclass
class MergedField:
    value: Any = None
    chosen_from: str | None = None
    confidence: float = 0.0
    conflict: bool = False
    note: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MergedGeometry:
    kind: MergedField = field(default_factory=MergedField)
    material: MergedField = field(default_factory=MergedField)
    dimensions: dict[str, MergedField] = field(default_factory=dict)
    ambiguities: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.to_payload(),
            "material": self.material.to_payload(),
            "dimensions": {key: value.to_payload() for key, value in self.dimensions.items()},
            "ambiguities": list(self.ambiguities),
        }


@dataclass
class MergedSource:
    source_type: MergedField = field(default_factory=MergedField)
    particle: MergedField = field(default_factory=MergedField)
    energy_mev: MergedField = field(default_factory=MergedField)
    position: MergedField = field(default_factory=MergedField)
    direction: MergedField = field(default_factory=MergedField)
    ambiguities: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.to_payload(),
            "particle": self.particle.to_payload(),
            "energy_mev": self.energy_mev.to_payload(),
            "position": self.position.to_payload(),
            "direction": self.direction.to_payload(),
            "ambiguities": list(self.ambiguities),
        }


@dataclass
class MergedTurnInterpretation:
    merged_summary: TurnSummary = field(default_factory=TurnSummary)
    merged_geometry: MergedGeometry = field(default_factory=MergedGeometry)
    merged_source: MergedSource = field(default_factory=MergedSource)
    conflicts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    trust_report: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "merged_summary": self.merged_summary.to_payload(),
            "merged_geometry": self.merged_geometry.to_payload(),
            "merged_source": self.merged_source.to_payload(),
            "conflicts": list(self.conflicts),
            "open_questions": list(self.open_questions),
            "trust_report": dict(self.trust_report),
        }


def _pick_scalar(
    llm_value: Any,
    evidence_value: Any,
    *,
    llm_confidence: float = 0.0,
    evidence_confidence: float = 1.0,
    note: str = "",
) -> tuple[MergedField, bool]:
    if evidence_value is not None and llm_value is None:
        return (
            MergedField(
                value=evidence_value,
                chosen_from="evidence",
                confidence=evidence_confidence,
                conflict=False,
                note=note,
            ),
            False,
        )
    if llm_value is not None and evidence_value is None:
        return (
            MergedField(
                value=llm_value,
                chosen_from="llm",
                confidence=llm_confidence,
                conflict=False,
                note=note,
            ),
            False,
        )
    if llm_value is not None and evidence_value is not None:
        if llm_value == evidence_value:
            return (
                MergedField(
                    value=evidence_value,
                    chosen_from="shared",
                    confidence=max(llm_confidence, evidence_confidence),
                    conflict=False,
                    note=note,
                ),
                False,
            )
        return (
            MergedField(
                value=evidence_value,
                chosen_from="evidence",
                confidence=evidence_confidence,
                conflict=True,
                note=note or "llm/evidence conflict",
            ),
            True,
        )
    return (MergedField(note=note), False)


def _pick_geometry_kind(
    geometry_candidate: GeometryCandidate,
    geometry_evidence: dict[str, Any],
) -> tuple[MergedField, bool]:
    llm_value = geometry_candidate.kind_candidate
    evidence_value = geometry_evidence.get("kind")
    evidence_dimensions = geometry_evidence.get("dimensions")
    evidence_has_dimensions = isinstance(evidence_dimensions, dict) and bool(evidence_dimensions)
    llm_has_dimensions = bool(geometry_candidate.dimension_hints)

    if llm_value is not None and evidence_value is not None and llm_value != evidence_value:
        # If the slot/evidence side only guessed a kind but carries no concrete dimensions,
        # prefer the richer LLM interpretation while keeping the conflict visible.
        if llm_has_dimensions and not evidence_has_dimensions:
            return (
                MergedField(
                    value=llm_value,
                    chosen_from="llm",
                    confidence=geometry_candidate.confidence,
                    conflict=True,
                    note="geometry.kind prefers llm because evidence lacks dimension support",
                ),
                True,
            )
    return _pick_scalar(
        llm_value,
        evidence_value,
        llm_confidence=geometry_candidate.confidence,
        note="geometry.kind",
    )


def _should_ask_source_direction(
    turn_summary: TurnSummary,
    source_candidate: SourceCandidate,
    merged_source: MergedSource,
) -> bool:
    if "source" not in turn_summary.explicit_domains:
        return False
    source_type = merged_source.source_type.value or source_candidate.source_type_candidate
    if source_type not in {"beam", "plane"}:
        return False
    if merged_source.direction.value is not None:
        return False
    if source_candidate.direction_mode:
        return False
    return True


def merge_candidates(
    turn_summary: TurnSummary,
    geometry_candidate: GeometryCandidate,
    source_candidate: SourceCandidate,
    *,
    geometry_evidence: dict[str, Any] | None = None,
    source_evidence: dict[str, Any] | None = None,
) -> MergedTurnInterpretation:
    geometry_evidence = geometry_evidence or {}
    source_evidence = source_evidence or {}
    conflicts: list[str] = []

    merged_geometry = MergedGeometry(ambiguities=list(geometry_candidate.ambiguities))
    merged_source = MergedSource(ambiguities=list(source_candidate.ambiguities))

    kind_field, kind_conflict = _pick_geometry_kind(geometry_candidate, geometry_evidence)
    if kind_conflict:
        conflicts.append("geometry.kind")
    merged_geometry.kind = kind_field

    material_field, material_conflict = _pick_scalar(
        geometry_candidate.material_candidate,
        geometry_evidence.get("material"),
        llm_confidence=geometry_candidate.confidence,
        note="geometry.material",
    )
    if material_conflict:
        conflicts.append("geometry.material")
    merged_geometry.material = material_field

    for key, llm_value in geometry_candidate.dimension_hints.items():
        field, has_conflict = _pick_scalar(
            llm_value,
            geometry_evidence.get("dimensions", {}).get(key) if isinstance(geometry_evidence.get("dimensions"), dict) else None,
            llm_confidence=geometry_candidate.confidence,
            note=f"geometry.{key}",
        )
        if has_conflict:
            conflicts.append(f"geometry.{key}")
        merged_geometry.dimensions[key] = field

    source_type_field, source_type_conflict = _pick_scalar(
        source_candidate.source_type_candidate,
        source_evidence.get("source_type"),
        llm_confidence=source_candidate.confidence,
        note="source.type",
    )
    if source_type_conflict:
        conflicts.append("source.type")
    merged_source.source_type = source_type_field

    particle_field, particle_conflict = _pick_scalar(
        source_candidate.particle_candidate,
        source_evidence.get("particle"),
        llm_confidence=source_candidate.confidence,
        note="source.particle",
    )
    if particle_conflict:
        conflicts.append("source.particle")
    merged_source.particle = particle_field

    energy_field, energy_conflict = _pick_scalar(
        source_candidate.energy_candidate_mev,
        source_evidence.get("energy_mev"),
        llm_confidence=source_candidate.confidence,
        note="source.energy_mev",
    )
    if energy_conflict:
        conflicts.append("source.energy_mev")
    merged_source.energy_mev = energy_field

    position_field, position_conflict = _pick_scalar(
        source_candidate.position_hint or None,
        source_evidence.get("position"),
        llm_confidence=source_candidate.confidence,
        note="source.position",
    )
    if position_conflict:
        conflicts.append("source.position")
    merged_source.position = position_field

    direction_field, direction_conflict = _pick_scalar(
        {
            "mode": source_candidate.direction_mode,
            "hint": source_candidate.direction_hint,
        }
        if source_candidate.direction_mode or source_candidate.direction_hint
        else None,
        source_evidence.get("direction"),
        llm_confidence=source_candidate.confidence,
        note="source.direction",
    )
    if direction_conflict:
        conflicts.append("source.direction")
    merged_source.direction = direction_field

    open_questions: list[str] = []
    if merged_geometry.kind.value is None and "geometry" in turn_summary.explicit_domains:
        open_questions.append("geometry.kind")
    if merged_source.source_type.value is None and "source" in turn_summary.explicit_domains:
        open_questions.append("source.type")
    if _should_ask_source_direction(turn_summary, source_candidate, merged_source):
        open_questions.append("source.direction")

    trust_report = {
        "geometry_from_llm": sum(1 for field in [merged_geometry.kind, merged_geometry.material, *merged_geometry.dimensions.values()] if field.chosen_from == "llm"),
        "geometry_from_evidence": sum(1 for field in [merged_geometry.kind, merged_geometry.material, *merged_geometry.dimensions.values()] if field.chosen_from == "evidence"),
        "source_from_llm": sum(1 for field in [merged_source.source_type, merged_source.particle, merged_source.energy_mev, merged_source.position, merged_source.direction] if field.chosen_from == "llm"),
        "source_from_evidence": sum(1 for field in [merged_source.source_type, merged_source.particle, merged_source.energy_mev, merged_source.position, merged_source.direction] if field.chosen_from == "evidence"),
    }

    return MergedTurnInterpretation(
        merged_summary=turn_summary,
        merged_geometry=merged_geometry,
        merged_source=merged_source,
        conflicts=conflicts,
        open_questions=open_questions,
        trust_report=trust_report,
    )

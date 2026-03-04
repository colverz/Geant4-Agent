from __future__ import annotations

from dataclasses import dataclass, field

from core.config.output_format_registry import accepted_output_formats
from core.slots.slot_frame import SlotFrame


_GEOMETRY_KINDS = {"box", "cylinder", "sphere"}
_SOURCE_KINDS = {"point", "beam", "plane", "isotropic"}
@dataclass
class SlotValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _valid_vec3(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    return all(isinstance(x, (int, float)) for x in value)


def validate_slot_frame(frame: SlotFrame) -> SlotValidationResult:
    errors: list[str] = []

    if frame.geometry.kind is not None and frame.geometry.kind not in _GEOMETRY_KINDS:
        errors.append(f"geometry.kind_invalid:{frame.geometry.kind}")
    if frame.geometry.size_triplet_mm is not None:
        if not _valid_vec3(frame.geometry.size_triplet_mm):
            errors.append("geometry.size_triplet_mm_invalid")
        elif any(float(x) <= 0 for x in frame.geometry.size_triplet_mm):
            errors.append("geometry.size_triplet_mm_nonpositive")
    if frame.geometry.radius_mm is not None and float(frame.geometry.radius_mm) <= 0:
        errors.append("geometry.radius_mm_nonpositive")
    if frame.geometry.half_length_mm is not None and float(frame.geometry.half_length_mm) <= 0:
        errors.append("geometry.half_length_mm_nonpositive")

    if frame.source.kind is not None and frame.source.kind not in _SOURCE_KINDS:
        errors.append(f"source.kind_invalid:{frame.source.kind}")
    if frame.source.energy_mev is not None and float(frame.source.energy_mev) <= 0:
        errors.append("source.energy_mev_nonpositive")
    if frame.source.position_mm is not None and not _valid_vec3(frame.source.position_mm):
        errors.append("source.position_mm_invalid")
    if frame.source.direction_vec is not None and not _valid_vec3(frame.source.direction_vec):
        errors.append("source.direction_vec_invalid")

    if frame.output.format is not None and frame.output.format not in accepted_output_formats():
        errors.append(f"output.format_invalid:{frame.output.format}")

    return SlotValidationResult(ok=len(errors) == 0, errors=errors)

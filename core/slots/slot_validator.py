from __future__ import annotations

from dataclasses import dataclass, field

from core.config.output_format_registry import accepted_output_formats
from core.geometry.family_catalog import SUPPORTED_GEOMETRY_KINDS
from core.slots.slot_frame import SlotFrame


_SOURCE_KINDS = {"point", "beam", "plane", "isotropic"}
_SOURCE_SPOT_PROFILES = {"uniform_disk", "gaussian"}
_SOURCE_DIVERGENCE_PROFILES = {"uniform_cone", "gaussian"}
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

    if frame.geometry.kind is not None and frame.geometry.kind not in SUPPORTED_GEOMETRY_KINDS:
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
    if frame.geometry.radius1_mm is not None and float(frame.geometry.radius1_mm) <= 0:
        errors.append("geometry.radius1_mm_nonpositive")
    if frame.geometry.radius2_mm is not None and float(frame.geometry.radius2_mm) <= 0:
        errors.append("geometry.radius2_mm_nonpositive")
    if frame.geometry.x1_mm is not None and float(frame.geometry.x1_mm) <= 0:
        errors.append("geometry.x1_mm_nonpositive")
    if frame.geometry.x2_mm is not None and float(frame.geometry.x2_mm) <= 0:
        errors.append("geometry.x2_mm_nonpositive")
    if frame.geometry.y1_mm is not None and float(frame.geometry.y1_mm) <= 0:
        errors.append("geometry.y1_mm_nonpositive")
    if frame.geometry.y2_mm is not None and float(frame.geometry.y2_mm) <= 0:
        errors.append("geometry.y2_mm_nonpositive")
    if frame.geometry.z_mm is not None and float(frame.geometry.z_mm) <= 0:
        errors.append("geometry.z_mm_nonpositive")
    if frame.geometry.z_planes_mm is not None:
        if not isinstance(frame.geometry.z_planes_mm, list) or len(frame.geometry.z_planes_mm) != 3:
            errors.append("geometry.z_planes_mm_invalid")
        elif not all(isinstance(x, (int, float)) for x in frame.geometry.z_planes_mm):
            errors.append("geometry.z_planes_mm_invalid")
    if frame.geometry.radii_mm is not None:
        if not isinstance(frame.geometry.radii_mm, list) or len(frame.geometry.radii_mm) != 3:
            errors.append("geometry.radii_mm_invalid")
        elif not all(isinstance(x, (int, float)) and float(x) > 0 for x in frame.geometry.radii_mm):
            errors.append("geometry.radii_mm_invalid")
    for attr in (
        "trap_x1_mm",
        "trap_x2_mm",
        "trap_x3_mm",
        "trap_x4_mm",
        "trap_y1_mm",
        "trap_y2_mm",
        "trap_z_mm",
        "para_x_mm",
        "para_y_mm",
        "para_z_mm",
        "torus_major_radius_mm",
        "torus_minor_radius_mm",
        "ellipsoid_ax_mm",
        "ellipsoid_by_mm",
        "ellipsoid_cz_mm",
        "elltube_ax_mm",
        "elltube_by_mm",
        "elltube_hz_mm",
    ):
        value = getattr(frame.geometry, attr)
        if value is not None and float(value) <= 0:
            errors.append(f"geometry.{attr}_nonpositive")
    if frame.geometry.polyhedra_sides is not None and int(frame.geometry.polyhedra_sides) < 3:
        errors.append("geometry.polyhedra_sides_invalid")
    for attr in ("para_alpha_deg", "para_theta_deg", "para_phi_deg"):
        value = getattr(frame.geometry, attr)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"geometry.{attr}_invalid")
    if frame.geometry.tilt_x_deg is not None and float(frame.geometry.tilt_x_deg) < 0:
        errors.append("geometry.tilt_x_deg_negative")
    if frame.geometry.tilt_y_deg is not None and float(frame.geometry.tilt_y_deg) < 0:
        errors.append("geometry.tilt_y_deg_negative")

    if frame.source.kind is not None and frame.source.kind not in _SOURCE_KINDS:
        errors.append(f"source.kind_invalid:{frame.source.kind}")
    if frame.source.energy_mev is not None and float(frame.source.energy_mev) <= 0:
        errors.append("source.energy_mev_nonpositive")
    if frame.source.position_mm is not None and not _valid_vec3(frame.source.position_mm):
        errors.append("source.position_mm_invalid")
    if frame.source.direction_vec is not None and not _valid_vec3(frame.source.direction_vec):
        errors.append("source.direction_vec_invalid")
    for attr in ("spot_radius_mm", "spot_sigma_mm", "divergence_half_angle_deg", "divergence_sigma_deg"):
        value = getattr(frame.source, attr)
        if value is not None and float(value) < 0:
            errors.append(f"source.{attr}_negative")
    if frame.source.spot_profile is not None and frame.source.spot_profile not in _SOURCE_SPOT_PROFILES:
        errors.append(f"source.spot_profile_invalid:{frame.source.spot_profile}")
    if frame.source.divergence_profile is not None and frame.source.divergence_profile not in _SOURCE_DIVERGENCE_PROFILES:
        errors.append(f"source.divergence_profile_invalid:{frame.source.divergence_profile}")
    if frame.detector.position_mm is not None and not _valid_vec3(frame.detector.position_mm):
        errors.append("detector.position_mm_invalid")
    if frame.detector.size_triplet_mm is not None:
        if not _valid_vec3(frame.detector.size_triplet_mm):
            errors.append("detector.size_triplet_mm_invalid")
        elif any(float(x) <= 0 for x in frame.detector.size_triplet_mm):
            errors.append("detector.size_triplet_mm_nonpositive")
    if frame.scoring.plane_z_mm is not None and not isinstance(frame.scoring.plane_z_mm, (int, float)):
        errors.append("scoring.plane_z_mm_invalid")

    if frame.output.format is not None and frame.output.format not in accepted_output_formats():
        errors.append(f"output.format_invalid:{frame.output.format}")

    return SlotValidationResult(ok=len(errors) == 0, errors=errors)

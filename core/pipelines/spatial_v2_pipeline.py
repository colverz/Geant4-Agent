from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from core.orchestrator.types import UpdateOp
from core.pipelines.geometry_v2_pipeline import build_v2_geometry_updates
from core.pipelines.source_v2_pipeline import build_v2_source_updates
from core.slots.slot_frame import SlotFrame


@dataclass(frozen=True)
class SpatialV2Result:
    geometry_updates: tuple[UpdateOp, ...]
    geometry_targets: tuple[str, ...]
    geometry_meta: dict[str, Any]
    source_updates: tuple[UpdateOp, ...]
    source_targets: tuple[str, ...]
    source_meta: dict[str, Any]
    warnings: tuple[str, ...]
    spatial_meta: dict[str, Any]

    @property
    def updates(self) -> tuple[UpdateOp, ...]:
        return self.geometry_updates + self.source_updates

    @property
    def target_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.geometry_targets, *self.source_targets]))


def _vector3(value: Any) -> list[float] | None:
    if isinstance(value, dict) and isinstance(value.get("value"), list) and len(value["value"]) >= 3:
        try:
            return [float(value["value"][0]), float(value["value"][1]), float(value["value"][2])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _scalar(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _updates_to_mapping(updates: tuple[UpdateOp, ...]) -> dict[str, Any]:
    return {update.path: update.value for update in updates}


def _analyze_box_source(mapping: dict[str, Any]) -> dict[str, Any]:
    x = _scalar(mapping.get("geometry.params.module_x"))
    y = _scalar(mapping.get("geometry.params.module_y"))
    z = _scalar(mapping.get("geometry.params.module_z"))
    position = _vector3(mapping.get("source.position"))
    if x is None or y is None or z is None or position is None:
        return {}
    hx, hy, hz = x * 0.5, y * 0.5, z * 0.5
    px, py, pz = position
    inside = abs(px) < hx and abs(py) < hy and abs(pz) < hz
    if inside:
        surface_distance = min(hx - abs(px), hy - abs(py), hz - abs(pz))
    else:
        dx = max(abs(px) - hx, 0.0)
        dy = max(abs(py) - hy, 0.0)
        dz = max(abs(pz) - hz, 0.0)
        surface_distance = sqrt(dx * dx + dy * dy + dz * dz)
    relation = "inside_target" if inside else "outside_target"
    if not inside and abs(abs(pz) - hz) < 1e-6 and abs(px) <= hx and abs(py) <= hy:
        relation = "on_target_face"
    elif not inside and surface_distance <= 1.0:
        relation = "near_target_surface"
    return {
        "source_relation": relation,
        "distance_to_target_center_mm": round(sqrt(px * px + py * py + pz * pz), 6),
        "surface_distance_mm": round(surface_distance, 6),
    }


def _analyze_tubs_source(mapping: dict[str, Any]) -> dict[str, Any]:
    radius = _scalar(mapping.get("geometry.params.child_rmax"))
    half_length = _scalar(mapping.get("geometry.params.child_hz"))
    position = _vector3(mapping.get("source.position"))
    if radius is None or half_length is None or position is None:
        return {}
    px, py, pz = position
    radial = sqrt(px * px + py * py)
    inside = radial < radius and abs(pz) < half_length
    if inside:
        radial_gap = radius - radial
        axial_gap = half_length - abs(pz)
        surface_distance = min(radial_gap, axial_gap)
    else:
        radial_gap = max(radial - radius, 0.0)
        axial_gap = max(abs(pz) - half_length, 0.0)
        surface_distance = sqrt(radial_gap * radial_gap + axial_gap * axial_gap)
    relation = "inside_target" if inside else "outside_target"
    if not inside and (abs(radial - radius) < 1e-6 or abs(abs(pz) - half_length) < 1e-6):
        relation = "on_target_face"
    elif not inside and surface_distance <= 1.0:
        relation = "near_target_surface"
    return {
        "source_relation": relation,
        "distance_to_target_center_mm": round(sqrt(px * px + py * py + pz * pz), 6),
        "surface_distance_mm": round(surface_distance, 6),
    }


def _analyze_spatial_context(
    geometry_updates: tuple[UpdateOp, ...],
    source_updates: tuple[UpdateOp, ...],
    geometry_meta: dict[str, Any],
    source_meta: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if not geometry_meta.get("compile_ok") or not source_meta.get("compile_ok"):
        return (), {}
    if geometry_meta.get("finalization_status") != "ready" or source_meta.get("finalization_status") != "ready":
        return (), {}

    mapping = _updates_to_mapping(geometry_updates + source_updates)
    structure = str(mapping.get("geometry.structure") or geometry_meta.get("structure") or "")
    if structure == "single_box":
        spatial_meta = _analyze_box_source(mapping)
    elif structure == "single_tubs":
        spatial_meta = _analyze_tubs_source(mapping)
    else:
        spatial_meta = {}

    warnings: list[str] = []
    relation = spatial_meta.get("source_relation")
    if relation == "inside_target":
        warnings.append("source_inside_target")
    elif relation == "on_target_face":
        warnings.append("source_on_target_face")
    elif relation == "near_target_surface":
        warnings.append("source_near_target_surface")
    return tuple(warnings), spatial_meta


def _apply_spatial_gates(
    source_updates: tuple[UpdateOp, ...],
    source_targets: tuple[str, ...],
    source_meta: dict[str, Any],
    warnings: tuple[str, ...],
    spatial_meta: dict[str, Any],
) -> tuple[tuple[UpdateOp, ...], tuple[str, ...], dict[str, Any]]:
    severe = {"source_inside_target", "source_on_target_face"}
    if not any(item in severe for item in warnings):
        gated_meta = dict(source_meta)
        if warnings:
            gated_meta["spatial_warnings"] = list(warnings)
        if spatial_meta:
            gated_meta["spatial_meta"] = dict(spatial_meta)
        return source_updates, source_targets, gated_meta

    gated_meta = dict(source_meta)
    gated_meta["compile_ok"] = False
    gated_meta["finalization_status"] = "review"
    errors = list(gated_meta.get("errors", []))
    if "spatial_source_target_conflict" not in errors:
        errors.append("spatial_source_target_conflict")
    gated_meta["errors"] = errors
    gated_meta["spatial_warnings"] = list(warnings)
    gated_meta["spatial_meta"] = dict(spatial_meta)
    return (), (), gated_meta


def build_v2_spatial_updates(frame: SlotFrame, *, turn_id: int) -> SpatialV2Result:
    geometry_updates, geometry_targets, geometry_meta = build_v2_geometry_updates(frame, turn_id=turn_id)
    source_updates, source_targets, source_meta = build_v2_source_updates(frame, turn_id=turn_id)
    warnings, spatial_meta = _analyze_spatial_context(
        tuple(geometry_updates),
        tuple(source_updates),
        geometry_meta,
        source_meta,
    )
    source_updates, source_targets, source_meta = _apply_spatial_gates(
        tuple(source_updates),
        tuple(source_targets),
        source_meta,
        warnings,
        spatial_meta,
    )
    return SpatialV2Result(
        geometry_updates=tuple(geometry_updates),
        geometry_targets=tuple(geometry_targets),
        geometry_meta=dict(geometry_meta),
        source_updates=tuple(source_updates),
        source_targets=tuple(source_targets),
        source_meta=dict(source_meta),
        warnings=warnings,
        spatial_meta=spatial_meta,
    )

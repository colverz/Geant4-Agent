from __future__ import annotations

from core.geometry.family_catalog import GEOMETRY_KIND_TO_STRUCTURE
from core.orchestrator.types import Producer, UpdateOp
from core.slots.slot_frame import SlotFrame


def build_legacy_geometry_updates(frame: SlotFrame, *, turn_id: int) -> tuple[list[UpdateOp], list[str]]:
    updates: list[UpdateOp] = []
    target_paths: list[str] = []

    if frame.geometry.kind:
        structure = GEOMETRY_KIND_TO_STRUCTURE.get(frame.geometry.kind)
        if structure:
            updates.append(
                UpdateOp(
                    path="geometry.structure",
                    op="set",
                    value=structure,
                    producer=Producer.SLOT_MAPPER,
                    confidence=frame.confidence or 0.8,
                    turn_id=turn_id,
                )
            )
            target_paths.extend(["geometry.structure", "geometry.root_name"])

    if frame.geometry.size_triplet_mm:
        for label, value in zip(("module_x", "module_y", "module_z"), frame.geometry.size_triplet_mm):
            updates.append(
                UpdateOp(
                    path=f"geometry.params.{label}",
                    op="set",
                    value=float(value),
                    producer=Producer.SLOT_MAPPER,
                    confidence=frame.confidence or 0.8,
                    turn_id=turn_id,
                )
            )
            target_paths.append(f"geometry.params.{label}")

    for attr, path in (
        ("radius_mm", "geometry.params.child_rmax"),
        ("half_length_mm", "geometry.params.child_hz"),
        ("radius1_mm", "geometry.params.rmax1"),
        ("radius2_mm", "geometry.params.rmax2"),
        ("x1_mm", "geometry.params.x1"),
        ("x2_mm", "geometry.params.x2"),
        ("y1_mm", "geometry.params.y1"),
        ("y2_mm", "geometry.params.y2"),
        ("z_mm", "geometry.params.module_z"),
        ("tilt_x_deg", "geometry.params.tilt_x"),
        ("tilt_y_deg", "geometry.params.tilt_y"),
    ):
        value = getattr(frame.geometry, attr)
        if value is None:
            continue
        updates.append(
            UpdateOp(
                path=path,
                op="set",
                value=float(value),
                producer=Producer.SLOT_MAPPER,
                confidence=frame.confidence or 0.8,
                turn_id=turn_id,
            )
        )
        target_paths.append(path)

    if frame.geometry.z_planes_mm and len(frame.geometry.z_planes_mm) == 3:
        for label, value in zip(("z1", "z2", "z3"), frame.geometry.z_planes_mm):
            path = f"geometry.params.{label}"
            updates.append(
                UpdateOp(path=path, op="set", value=float(value), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
            )
            target_paths.append(path)

    if frame.geometry.radii_mm and len(frame.geometry.radii_mm) == 3:
        for label, value in zip(("r1", "r2", "r3"), frame.geometry.radii_mm):
            path = f"geometry.params.{label}"
            updates.append(
                UpdateOp(path=path, op="set", value=float(value), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
            )
            target_paths.append(path)

    for slot_name, param_name in (
        ("trap_x1_mm", "trap_x1"),
        ("trap_x2_mm", "trap_x2"),
        ("trap_x3_mm", "trap_x3"),
        ("trap_x4_mm", "trap_x4"),
        ("trap_y1_mm", "trap_y1"),
        ("trap_y2_mm", "trap_y2"),
        ("trap_z_mm", "trap_z"),
        ("para_x_mm", "para_x"),
        ("para_y_mm", "para_y"),
        ("para_z_mm", "para_z"),
        ("para_alpha_deg", "para_alpha"),
        ("para_theta_deg", "para_theta"),
        ("para_phi_deg", "para_phi"),
        ("torus_major_radius_mm", "torus_rtor"),
        ("torus_minor_radius_mm", "torus_rmax"),
        ("ellipsoid_ax_mm", "ellipsoid_ax"),
        ("ellipsoid_by_mm", "ellipsoid_by"),
        ("ellipsoid_cz_mm", "ellipsoid_cz"),
        ("elltube_ax_mm", "elltube_ax"),
        ("elltube_by_mm", "elltube_by"),
        ("elltube_hz_mm", "elltube_hz"),
    ):
        value = getattr(frame.geometry, slot_name)
        if value is None:
            continue
        path = f"geometry.params.{param_name}"
        updates.append(
            UpdateOp(path=path, op="set", value=float(value), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append(path)

    if frame.geometry.polyhedra_sides is not None:
        path = "geometry.params.polyhedra_nsides"
        updates.append(
            UpdateOp(path=path, op="set", value=int(frame.geometry.polyhedra_sides), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append(path)

    return updates, target_paths

from __future__ import annotations

from typing import Any

from core.contracts.slots import SlotFrame
from core.orchestrator.path_ops import get_path


def _source_frame_has_content(frame: SlotFrame) -> bool:
    return any(
        (
            frame.source.kind,
            frame.source.particle,
            frame.source.energy_mev is not None,
            frame.source.position_mm,
            frame.source.direction_vec,
        )
    )


def _geometry_frame_has_content(frame: SlotFrame) -> bool:
    return any(
        (
            frame.geometry.kind,
            frame.geometry.size_triplet_mm,
            frame.geometry.radius_mm is not None,
            frame.geometry.half_length_mm is not None,
            frame.materials.primary,
        )
    )


def _merge_slot_frame_with_memory(frame: SlotFrame, slot_memory: dict[str, Any]) -> SlotFrame:
    if not isinstance(slot_memory, dict) or not slot_memory:
        return frame

    geometry_memory = slot_memory.get("geometry") if isinstance(slot_memory.get("geometry"), dict) else {}
    source_memory = slot_memory.get("source") if isinstance(slot_memory.get("source"), dict) else {}
    material_memory = slot_memory.get("materials") if isinstance(slot_memory.get("materials"), dict) else {}

    if _geometry_frame_has_content(frame):
        if frame.geometry.kind is None and isinstance(geometry_memory.get("kind"), str):
            frame.geometry.kind = geometry_memory["kind"]
        if frame.geometry.size_triplet_mm is None and isinstance(geometry_memory.get("size_triplet_mm"), list):
            frame.geometry.size_triplet_mm = list(geometry_memory["size_triplet_mm"])
        if frame.geometry.radius_mm is None and geometry_memory.get("radius_mm") is not None:
            frame.geometry.radius_mm = float(geometry_memory["radius_mm"])
        if frame.geometry.half_length_mm is None and geometry_memory.get("half_length_mm") is not None:
            frame.geometry.half_length_mm = float(geometry_memory["half_length_mm"])
        if frame.materials.primary is None and isinstance(material_memory.get("primary"), str):
            frame.materials.primary = material_memory["primary"]

    if _source_frame_has_content(frame):
        memory_kind = source_memory.get("kind")
        explicit_kind = frame.source.kind
        memory_position = source_memory.get("position_mm")
        explicit_position = frame.source.position_mm
        source_changed = bool(
            isinstance(explicit_kind, str)
            and isinstance(memory_kind, str)
            and explicit_kind
            and memory_kind
            and explicit_kind != memory_kind
        )
        position_changed = bool(
            isinstance(explicit_position, list)
            and isinstance(memory_position, list)
            and explicit_position
            and memory_position
            and list(explicit_position) != list(memory_position)
        )
        if not source_changed:
            if frame.source.kind is None and isinstance(memory_kind, str):
                frame.source.kind = memory_kind
            if frame.source.particle is None and isinstance(source_memory.get("particle"), str):
                frame.source.particle = source_memory["particle"]
            if frame.source.energy_mev is None and source_memory.get("energy_mev") is not None:
                frame.source.energy_mev = float(source_memory["energy_mev"])
            if frame.source.position_mm is None and isinstance(memory_position, list):
                frame.source.position_mm = list(memory_position)
            if (
                frame.source.direction_vec is None
                and not position_changed
                and isinstance(source_memory.get("direction_vec"), list)
            ):
                frame.source.direction_vec = list(source_memory["direction_vec"])

    return frame


def _refresh_slot_memory(
    existing_memory: dict[str, Any] | None,
    *,
    config: dict[str, Any],
    slot_frame: SlotFrame | None,
    allow_frame_overlay: bool,
) -> dict[str, Any]:
    memory: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    materials: dict[str, Any] = {}
    source: dict[str, Any] = {}

    structure = get_path(config, "geometry.structure")
    params = get_path(config, "geometry.params", {})
    if isinstance(structure, str) and structure:
        if structure == "single_box":
            geometry["kind"] = "box"
        elif structure == "single_tubs":
            geometry["kind"] = "cylinder"
    if isinstance(params, dict):
        if all(params.get(key) is not None for key in ("module_x", "module_y", "module_z")):
            geometry["size_triplet_mm"] = [
                float(params["module_x"]),
                float(params["module_y"]),
                float(params["module_z"]),
            ]
        if params.get("child_rmax") is not None:
            geometry["radius_mm"] = float(params["child_rmax"])
        if params.get("child_hz") is not None:
            geometry["half_length_mm"] = float(params["child_hz"])

    selected_materials = get_path(config, "materials.selected_materials", [])
    if isinstance(selected_materials, list) and selected_materials and isinstance(selected_materials[0], str):
        materials["primary"] = selected_materials[0]

    source_type = get_path(config, "source.type")
    if isinstance(source_type, str) and source_type:
        source["kind"] = source_type
    source_particle = get_path(config, "source.particle")
    if isinstance(source_particle, str) and source_particle:
        source["particle"] = source_particle
    source_energy = get_path(config, "source.energy")
    if source_energy is not None:
        source["energy_mev"] = float(source_energy)
    source_position = get_path(config, "source.position")
    if isinstance(source_position, dict):
        pos_value = source_position.get("value")
        if isinstance(pos_value, list) and len(pos_value) == 3 and all(value is not None for value in pos_value):
            source["position_mm"] = [float(pos_value[0]), float(pos_value[1]), float(pos_value[2])]
    source_direction = get_path(config, "source.direction")
    if isinstance(source_direction, dict):
        dir_value = source_direction.get("value")
        if isinstance(dir_value, list) and len(dir_value) == 3 and all(value is not None for value in dir_value):
            source["direction_vec"] = [float(dir_value[0]), float(dir_value[1]), float(dir_value[2])]

    if allow_frame_overlay and slot_frame is not None:
        if frame_kind := getattr(slot_frame.geometry, "kind", None):
            geometry["kind"] = frame_kind
        if isinstance(slot_frame.geometry.size_triplet_mm, list):
            geometry["size_triplet_mm"] = list(slot_frame.geometry.size_triplet_mm)
        if slot_frame.geometry.radius_mm is not None:
            geometry["radius_mm"] = float(slot_frame.geometry.radius_mm)
        if slot_frame.geometry.half_length_mm is not None:
            geometry["half_length_mm"] = float(slot_frame.geometry.half_length_mm)
        if slot_frame.materials.primary:
            materials["primary"] = slot_frame.materials.primary
        if slot_frame.source.kind:
            source["kind"] = slot_frame.source.kind
        if slot_frame.source.particle:
            source["particle"] = slot_frame.source.particle
        if slot_frame.source.energy_mev is not None:
            source["energy_mev"] = float(slot_frame.source.energy_mev)
        if isinstance(slot_frame.source.position_mm, list):
            source["position_mm"] = list(slot_frame.source.position_mm)
        if isinstance(slot_frame.source.direction_vec, list):
            source["direction_vec"] = list(slot_frame.source.direction_vec)

    if geometry:
        memory["geometry"] = geometry
    if materials:
        memory["materials"] = materials
    if source:
        memory["source"] = source
    return memory

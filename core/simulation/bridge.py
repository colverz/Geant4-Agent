from __future__ import annotations

from typing import Any

from core.simulation.spec import (
    GeometryRuntimeSpec,
    PhysicsRuntimeSpec,
    RunControlSpec,
    ScoringSpec,
    SimulationSpec,
    SourceRuntimeSpec,
)


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_vector3(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                return (float(raw[0]), float(raw[1]), float(raw[2]))
            except (TypeError, ValueError):
                return fallback
        keys = ("x", "y", "z")
        if all(key in value for key in keys):
            try:
                return (float(value["x"]), float(value["y"]), float(value["z"]))
            except (TypeError, ValueError):
                return fallback
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _first_material(config: dict[str, Any]) -> str:
    materials = config.get("materials", {}) if isinstance(config.get("materials"), dict) else {}
    vmap = materials.get("volume_material_map")
    if isinstance(vmap, dict):
        for _, material in vmap.items():
            if material:
                return str(material)
    if isinstance(vmap, list):
        for item in vmap:
            if isinstance(item, dict) and item.get("material"):
                return str(item["material"])
    selected = materials.get("selected_materials")
    if isinstance(selected, list):
        for item in selected:
            if item:
                return str(item)
    return "G4_Cu"


def _physics_list_name(config: dict[str, Any]) -> str:
    physics_list = config.get("physics_list")
    if isinstance(physics_list, dict) and physics_list.get("name"):
        return str(physics_list["name"])
    physics = config.get("physics")
    if isinstance(physics, dict) and physics.get("physics_list"):
        return str(physics["physics_list"])
    if isinstance(physics_list, str) and physics_list.strip():
        return physics_list.strip()
    return "FTFP_BERT"


def build_simulation_spec(config: dict[str, Any], *, events: int = 1, mode: str = "batch") -> SimulationSpec:
    raw = config if isinstance(config, dict) else {}
    geometry = raw.get("geometry", {}) if isinstance(raw.get("geometry"), dict) else {}
    params = geometry.get("params", {}) if isinstance(geometry.get("params"), dict) else {}
    source = raw.get("source", {}) if isinstance(raw.get("source"), dict) else {}
    scoring = raw.get("scoring", {}) if isinstance(raw.get("scoring"), dict) else {}

    structure = str(geometry.get("structure") or "single_box")
    geometry_spec = GeometryRuntimeSpec(
        structure=structure,
        material=_first_material(raw),
        size_x_mm=_coerce_float(geometry.get("size_triplet_mm", [None, None, None])[0], _coerce_float(params.get("module_x"), 50.0))
        if isinstance(geometry.get("size_triplet_mm"), (list, tuple)) and len(geometry.get("size_triplet_mm")) >= 1
        else _coerce_float(params.get("module_x"), 50.0),
        size_y_mm=_coerce_float(geometry.get("size_triplet_mm", [None, None, None])[1], _coerce_float(params.get("module_y"), 50.0))
        if isinstance(geometry.get("size_triplet_mm"), (list, tuple)) and len(geometry.get("size_triplet_mm")) >= 2
        else _coerce_float(params.get("module_y"), 50.0),
        size_z_mm=_coerce_float(geometry.get("size_triplet_mm", [None, None, None])[2], _coerce_float(params.get("module_z"), 50.0))
        if isinstance(geometry.get("size_triplet_mm"), (list, tuple)) and len(geometry.get("size_triplet_mm")) >= 3
        else _coerce_float(params.get("module_z"), 50.0),
        radius_mm=_coerce_float(params.get("child_rmax") or geometry.get("radius_mm"), 25.0),
        half_length_mm=_coerce_float(params.get("child_hz") or geometry.get("half_length_mm"), 50.0),
    )

    source_spec = SourceRuntimeSpec(
        source_type=str(source.get("type") or "point"),
        particle=str(source.get("particle") or "gamma"),
        energy_mev=_coerce_float(source.get("energy"), 1.0),
        position_mm=_coerce_vector3(source.get("position"), (0.0, 0.0, -100.0)),
        direction_vec=_coerce_vector3(source.get("direction"), (0.0, 0.0, 1.0)),
    )

    scoring_enabled = bool(scoring.get("target_edep", True))
    volume_names = scoring.get("volume_names")
    if isinstance(volume_names, (list, tuple)):
        cleaned = tuple(str(name) for name in volume_names if str(name))
    else:
        cleaned = ("Target",)

    return SimulationSpec(
        geometry=geometry_spec,
        source=source_spec,
        physics=PhysicsRuntimeSpec(physics_list=_physics_list_name(raw)),
        run=RunControlSpec(events=max(1, int(events)), mode=str(mode or "batch")),
        scoring=ScoringSpec(target_edep=scoring_enabled, volume_names=cleaned or ("Target",)),
    )

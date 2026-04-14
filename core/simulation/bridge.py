from __future__ import annotations

from typing import Any

from core.simulation.spec import (
    DetectorRuntimeSpec,
    GeometryRuntimeSpec,
    PhysicsRuntimeSpec,
    RunControlSpec,
    ScoringPlaneSpec,
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


def _run_control_spec(config: dict[str, Any], *, events: int, mode: str) -> RunControlSpec:
    simulation = config.get("simulation", {}) if isinstance(config.get("simulation"), dict) else {}
    raw_run = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    simulation_run = simulation.get("run", {}) if isinstance(simulation.get("run"), dict) else {}

    seed_value = (
        raw_run.get("seed")
        if "seed" in raw_run
        else simulation_run.get("seed")
        if "seed" in simulation_run
        else simulation.get("seed")
    )
    try:
        seed = int(seed_value) if seed_value is not None else 1337
    except (TypeError, ValueError):
        seed = 1337

    return RunControlSpec(
        events=max(1, int(events)),
        mode=str(mode or "batch"),
        seed=seed,
    )


def _root_volume_name(config: dict[str, Any]) -> str:
    geometry = config.get("geometry", {}) if isinstance(config.get("geometry"), dict) else {}
    raw = geometry.get("root_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "Target"


def _detector_spec(config: dict[str, Any]) -> DetectorRuntimeSpec | None:
    raw_detector = config.get("simulation", {}) if isinstance(config.get("simulation"), dict) else {}
    if isinstance(raw_detector, dict):
        raw_detector = raw_detector.get("detector")
    if raw_detector is None:
        raw_detector = config.get("detector")
    if not isinstance(raw_detector, dict):
        return None
    if raw_detector.get("enabled") is False:
        return None

    size_triplet = raw_detector.get("size_triplet_mm")
    if isinstance(size_triplet, (list, tuple)) and len(size_triplet) >= 3:
        size_x = _coerce_float(size_triplet[0], 20.0)
        size_y = _coerce_float(size_triplet[1], 20.0)
        size_z = _coerce_float(size_triplet[2], 2.0)
    else:
        params = raw_detector.get("params", {}) if isinstance(raw_detector.get("params"), dict) else {}
        size_x = _coerce_float(raw_detector.get("size_x_mm"), _coerce_float(params.get("module_x"), 20.0))
        size_y = _coerce_float(raw_detector.get("size_y_mm"), _coerce_float(params.get("module_y"), 20.0))
        size_z = _coerce_float(raw_detector.get("size_z_mm"), _coerce_float(params.get("module_z"), 2.0))

    return DetectorRuntimeSpec(
        volume_name=str(raw_detector.get("name") or "Detector"),
        material=str(raw_detector.get("material") or "G4_Si"),
        position_mm=_coerce_vector3(raw_detector.get("position"), (0.0, 0.0, 100.0)),
        size_x_mm=size_x,
        size_y_mm=size_y,
        size_z_mm=size_z,
    )


def _scoring_spec(
    config: dict[str, Any],
    root_volume_name: str,
    detector_spec: DetectorRuntimeSpec | None,
) -> ScoringSpec:
    scoring = config.get("scoring", {}) if isinstance(config.get("scoring"), dict) else {}
    scoring_enabled = bool(scoring.get("target_edep", True))
    detector_crossings = bool(scoring.get("detector_crossings", True))
    plane_crossings = bool(scoring.get("plane_crossings", False))
    raw_plane = scoring.get("plane") if isinstance(scoring.get("plane"), dict) else {}
    scoring_plane = None
    if plane_crossings:
        scoring_plane = ScoringPlaneSpec(
            name=str(raw_plane.get("name") or "ScoringPlane"),
            z_mm=_coerce_float(raw_plane.get("z_mm"), 0.0),
        )
    volume_names = scoring.get("volume_names")
    if isinstance(volume_names, (list, tuple)):
        cleaned = tuple(str(name).strip() for name in volume_names if str(name).strip())
    else:
        cleaned = ()

    role_map: dict[str, tuple[str, ...]] = {"target": (root_volume_name,)}
    if detector_spec is not None:
        role_map["detector"] = (detector_spec.volume_name,)
    raw_roles = scoring.get("volume_roles")
    if isinstance(raw_roles, dict):
        for role, raw_names in raw_roles.items():
            role_name = str(role).strip()
            if not role_name:
                continue
            if isinstance(raw_names, (list, tuple)):
                names = tuple(str(name).strip() for name in raw_names if str(name).strip())
            elif isinstance(raw_names, str) and raw_names.strip():
                names = (raw_names.strip(),)
            else:
                names = ()
            if names:
                role_map[role_name] = names

    all_names: list[str] = []
    for names in role_map.values():
        for name in names:
            if name not in all_names:
                all_names.append(name)
    for name in cleaned:
        if name not in all_names:
            all_names.append(name)

    return ScoringSpec(
        target_edep=scoring_enabled,
        detector_crossings=detector_crossings,
        plane_crossings=plane_crossings,
        scoring_plane=scoring_plane,
        volume_names=tuple(all_names) or (root_volume_name,),
        volume_roles=role_map,
    )


def build_simulation_spec(config: dict[str, Any], *, events: int = 1, mode: str = "batch") -> SimulationSpec:
    raw = config if isinstance(config, dict) else {}
    geometry = raw.get("geometry", {}) if isinstance(raw.get("geometry"), dict) else {}
    params = geometry.get("params", {}) if isinstance(geometry.get("params"), dict) else {}
    source = raw.get("source", {}) if isinstance(raw.get("source"), dict) else {}

    structure = str(geometry.get("structure") or "single_box")
    root_volume_name = _root_volume_name(raw)
    detector_spec = _detector_spec(raw)
    geometry_spec = GeometryRuntimeSpec(
        structure=structure,
        material=_first_material(raw),
        root_volume_name=root_volume_name,
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

    return SimulationSpec(
        geometry=geometry_spec,
        source=source_spec,
        physics=PhysicsRuntimeSpec(physics_list=_physics_list_name(raw)),
        run=_run_control_spec(raw, events=events, mode=mode),
        scoring=_scoring_spec(raw, root_volume_name, detector_spec),
        detector=detector_spec,
    )

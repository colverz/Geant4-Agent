from __future__ import annotations

from typing import Any

from core.simulation.spec import (
    BeamDivergenceSpec,
    BeamModelSpec,
    BeamSpotSpec,
    DetectorRuntimeSpec,
    GeometryRuntimeSpec,
    PhysicsRuntimeSpec,
    RunControlSpec,
    RuntimeVolumeSpec,
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


def _nonnegative_float(value: Any, fallback: float = 0.0) -> float:
    return max(0.0, _coerce_float(value, fallback))


def _choice(value: Any, *, allowed: set[str], fallback: str) -> str:
    candidate = str(value or fallback).strip().lower()
    return candidate if candidate in allowed else fallback


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


def _coerce_int(value: Any, fallback: int = 0) -> int:
    try:
        if value is None:
            return int(fallback)
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _shape_name(value: Any, fallback: str = "box") -> str:
    raw = str(value or fallback).strip().lower()
    aliases = {
        "single_box": "box",
        "box": "box",
        "cube": "box",
        "single_tubs": "tubs",
        "tubs": "tubs",
        "tube": "tubs",
        "cylinder": "tubs",
    }
    return aliases.get(raw, fallback)


def _volume_from_dict(raw: dict[str, Any], *, fallback_name: str, fallback_material: str) -> RuntimeVolumeSpec:
    size = raw.get("size_mm", raw.get("size_triplet_mm"))
    size_mm = _coerce_vector3(size, (50.0, 50.0, 50.0)) if size is not None else None
    return RuntimeVolumeSpec(
        name=str(raw.get("name") or fallback_name),
        shape=_shape_name(raw.get("shape", raw.get("type")), "box"),
        material=str(raw.get("material") or fallback_material),
        role=str(raw.get("role") or "").strip().lower(),
        parent=str(raw.get("parent") or "World"),
        position_mm=_coerce_vector3(raw.get("position_mm", raw.get("position")), (0.0, 0.0, 0.0)),
        rotation_deg=_coerce_vector3(raw.get("rotation_deg", raw.get("rotation")), (0.0, 0.0, 0.0)),
        size_mm=size_mm,
        radius_mm=_coerce_float(raw.get("radius_mm", raw.get("outer_radius_mm")), 25.0)
        if raw.get("radius_mm", raw.get("outer_radius_mm")) is not None
        else None,
        inner_radius_mm=_nonnegative_float(raw.get("inner_radius_mm"), 0.0),
        half_length_mm=_coerce_float(raw.get("half_length_mm"), 50.0)
        if raw.get("half_length_mm") is not None
        else None,
        copy_no=_coerce_int(raw.get("copy_no"), 0),
    )


def _volume_roles(volumes: tuple[RuntimeVolumeSpec, ...], root_volume_name: str) -> dict[str, tuple[str, ...]]:
    roles: dict[str, list[str]] = {}
    for volume in volumes:
        role = str(volume.role or "").strip().lower()
        if not role:
            continue
        roles.setdefault(role, [])
        if volume.name not in roles[role]:
            roles[role].append(volume.name)
    if "target" not in roles:
        if any(volume.name == root_volume_name for volume in volumes):
            roles["target"] = [root_volume_name]
        else:
            roles["target"] = [volume.name for volume in volumes if volume.role != "detector"]
    return {role: tuple(names) for role, names in roles.items() if names}


def _layer_stack_volumes(
    geometry: dict[str, Any],
    *,
    root_volume_name: str,
    fallback_material: str,
    size_x_mm: float,
    size_y_mm: float,
) -> tuple[RuntimeVolumeSpec, ...]:
    raw_layers = geometry.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        return ()
    total_thickness = 0.0
    parsed: list[tuple[dict[str, Any], float]] = []
    for layer in raw_layers:
        if not isinstance(layer, dict):
            continue
        thickness = _nonnegative_float(layer.get("thickness_mm"), 0.0)
        if thickness <= 0.0:
            continue
        parsed.append((layer, thickness))
        total_thickness += thickness
    if total_thickness <= 0.0:
        return ()
    volumes: list[RuntimeVolumeSpec] = []
    z_cursor = -0.5 * total_thickness
    for index, (layer, thickness) in enumerate(parsed):
        center_z = z_cursor + thickness * 0.5
        z_cursor += thickness
        name = str(layer.get("name") or f"{root_volume_name}_Layer{index + 1}")
        volumes.append(
            RuntimeVolumeSpec(
                name=name,
                shape="box",
                material=str(layer.get("material") or fallback_material),
                role=str(layer.get("role") or "shield").strip().lower(),
                parent="World",
                position_mm=(0.0, 0.0, center_z),
                size_mm=(size_x_mm, size_y_mm, thickness),
                copy_no=index,
            )
        )
    return tuple(volumes)


def _step_wedge_volumes(
    geometry: dict[str, Any],
    *,
    root_volume_name: str,
    fallback_material: str,
    size_y_mm: float,
) -> tuple[RuntimeVolumeSpec, ...]:
    raw_steps = geometry.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return ()
    widths: list[float] = []
    heights: list[float] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        widths.append(_nonnegative_float(step.get("width_mm"), 0.0))
        heights.append(_nonnegative_float(step.get("thickness_mm", step.get("height_mm")), 0.0))
    if not widths or any(value <= 0.0 for value in widths) or any(value <= 0.0 for value in heights):
        return ()
    total_width = sum(widths)
    x_cursor = -0.5 * total_width
    volumes: list[RuntimeVolumeSpec] = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            continue
        width = widths[index]
        thickness = heights[index]
        center_x = x_cursor + width * 0.5
        x_cursor += width
        role = str(step.get("role") or ("region_a" if index == 0 else "region_b")).strip().lower()
        volumes.append(
            RuntimeVolumeSpec(
                name=str(step.get("name") or f"{root_volume_name}_Step{index + 1}"),
                shape="box",
                material=str(step.get("material") or fallback_material),
                role=role,
                parent="World",
                position_mm=(center_x, 0.0, 0.0),
                size_mm=(width, size_y_mm, thickness),
                copy_no=index,
            )
        )
    return tuple(volumes)


def _geometry_volumes(
    geometry: dict[str, Any],
    *,
    structure: str,
    root_volume_name: str,
    material: str,
    size_x_mm: float,
    size_y_mm: float,
    size_z_mm: float,
    radius_mm: float,
    half_length_mm: float,
    detector_spec: DetectorRuntimeSpec | None,
) -> tuple[RuntimeVolumeSpec, ...]:
    raw_volumes = geometry.get("volumes")
    if isinstance(raw_volumes, list) and raw_volumes:
        volumes = tuple(
            _volume_from_dict(item, fallback_name=f"Volume{index + 1}", fallback_material=material)
            for index, item in enumerate(raw_volumes)
            if isinstance(item, dict)
        )
    elif structure == "multi_layer_stack":
        volumes = _layer_stack_volumes(
            geometry,
            root_volume_name=root_volume_name,
            fallback_material=material,
            size_x_mm=size_x_mm,
            size_y_mm=size_y_mm,
        )
    elif structure == "step_wedge":
        volumes = _step_wedge_volumes(
            geometry,
            root_volume_name=root_volume_name,
            fallback_material=material,
            size_y_mm=size_y_mm,
        )
    else:
        root_shape = _shape_name(structure, "box")
        volumes = (
            RuntimeVolumeSpec(
                name=root_volume_name,
                shape=root_shape,
                material=material,
                role="target",
                parent="World",
                size_mm=(size_x_mm, size_y_mm, size_z_mm) if root_shape == "box" else None,
                radius_mm=radius_mm if root_shape == "tubs" else None,
                half_length_mm=half_length_mm if root_shape == "tubs" else None,
            ),
        )
    if detector_spec is not None and all(volume.name != detector_spec.volume_name for volume in volumes):
        volumes = (
            *volumes,
            RuntimeVolumeSpec(
                name=detector_spec.volume_name,
                shape="box",
                material=detector_spec.material,
                role="detector",
                parent="World",
                position_mm=detector_spec.position_mm,
                size_mm=(detector_spec.size_x_mm, detector_spec.size_y_mm, detector_spec.size_z_mm),
            ),
        )
    return volumes


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
    geometry_roles: dict[str, tuple[str, ...]] | None = None,
    geometry_volume_names: tuple[str, ...] = (),
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

    role_map: dict[str, tuple[str, ...]] = dict(geometry_roles or {"target": (root_volume_name,)})
    if "target" not in role_map:
        role_map["target"] = (root_volume_name,)
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
    for name in geometry_volume_names:
        if name not in all_names:
            all_names.append(name)
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
        requests=tuple(item for item in scoring.get("requests", ()) if isinstance(item, dict))
        if isinstance(scoring.get("requests"), (list, tuple))
        else (),
        depth_bins=scoring.get("depth_bins") if isinstance(scoring.get("depth_bins"), dict) else None,
        derived_metrics=tuple(str(item) for item in scoring.get("derived_metrics", ()) if str(item).strip())
        if isinstance(scoring.get("derived_metrics"), (list, tuple))
        else (),
    )


def build_simulation_spec(config: dict[str, Any], *, events: int = 1, mode: str = "batch") -> SimulationSpec:
    raw = config if isinstance(config, dict) else {}
    geometry = raw.get("geometry", {}) if isinstance(raw.get("geometry"), dict) else {}
    params = geometry.get("params", {}) if isinstance(geometry.get("params"), dict) else {}
    source = raw.get("source", {}) if isinstance(raw.get("source"), dict) else {}

    structure = str(geometry.get("structure") or "single_box")
    root_volume_name = _root_volume_name(raw)
    detector_spec = _detector_spec(raw)
    material = _first_material(raw)
    size_x_mm = (
        _coerce_float(geometry.get("size_triplet_mm", [None, None, None])[0], _coerce_float(params.get("module_x"), 50.0))
        if isinstance(geometry.get("size_triplet_mm"), (list, tuple)) and len(geometry.get("size_triplet_mm")) >= 1
        else _coerce_float(params.get("module_x"), 50.0)
    )
    size_y_mm = (
        _coerce_float(geometry.get("size_triplet_mm", [None, None, None])[1], _coerce_float(params.get("module_y"), 50.0))
        if isinstance(geometry.get("size_triplet_mm"), (list, tuple)) and len(geometry.get("size_triplet_mm")) >= 2
        else _coerce_float(params.get("module_y"), 50.0)
    )
    size_z_mm = (
        _coerce_float(geometry.get("size_triplet_mm", [None, None, None])[2], _coerce_float(params.get("module_z"), 50.0))
        if isinstance(geometry.get("size_triplet_mm"), (list, tuple)) and len(geometry.get("size_triplet_mm")) >= 3
        else _coerce_float(params.get("module_z"), 50.0)
    )
    radius_mm = _coerce_float(params.get("child_rmax") or geometry.get("radius_mm"), 25.0)
    half_length_mm = _coerce_float(params.get("child_hz") or geometry.get("half_length_mm"), 50.0)
    volumes = _geometry_volumes(
        geometry,
        structure=structure,
        root_volume_name=root_volume_name,
        material=material,
        size_x_mm=size_x_mm,
        size_y_mm=size_y_mm,
        size_z_mm=size_z_mm,
        radius_mm=radius_mm,
        half_length_mm=half_length_mm,
        detector_spec=detector_spec,
    )
    geometry_roles = _volume_roles(volumes, root_volume_name)
    geometry_spec = GeometryRuntimeSpec(
        structure=structure,
        material=material,
        root_volume_name=root_volume_name,
        size_x_mm=size_x_mm,
        size_y_mm=size_y_mm,
        size_z_mm=size_z_mm,
        radius_mm=radius_mm,
        half_length_mm=half_length_mm,
        volumes=volumes,
        roles=geometry_roles,
    )

    source_spec = SourceRuntimeSpec(
        source_type=str(source.get("type") or "point"),
        particle=str(source.get("particle") or "gamma"),
        energy_mev=_coerce_float(source.get("energy"), 1.0),
        position_mm=_coerce_vector3(source.get("position"), (0.0, 0.0, -100.0)),
        direction_vec=_coerce_vector3(source.get("direction"), (0.0, 0.0, 1.0)),
        beam_model=BeamModelSpec(
            spot=BeamSpotSpec(
                radius_mm=_nonnegative_float(
                    source.get("spot_radius_mm", source.get("beam_spot_radius_mm")),
                    0.0,
                ),
                profile=_choice(
                    source.get("spot_profile", source.get("beam_spot_profile")),
                    allowed={"uniform_disk", "gaussian"},
                    fallback="uniform_disk",
                ),
                sigma_mm=_nonnegative_float(
                    source.get("spot_sigma_mm", source.get("beam_spot_sigma_mm")),
                    0.0,
                ),
            ),
            divergence=BeamDivergenceSpec(
                half_angle_deg=_nonnegative_float(
                    source.get("divergence_half_angle_deg", source.get("divergence_deg")),
                    0.0,
                ),
                profile=_choice(
                    source.get("divergence_profile", source.get("beam_divergence_profile")),
                    allowed={"uniform_cone", "gaussian"},
                    fallback="uniform_cone",
                ),
                sigma_deg=_nonnegative_float(
                    source.get("divergence_sigma_deg", source.get("beam_divergence_sigma_deg")),
                    0.0,
                ),
            ),
        ),
    )

    return SimulationSpec(
        geometry=geometry_spec,
        source=source_spec,
        physics=PhysicsRuntimeSpec(physics_list=_physics_list_name(raw)),
        run=_run_control_spec(raw, events=events, mode=mode),
        scoring=_scoring_spec(
            raw,
            root_volume_name,
            detector_spec,
            geometry_roles=geometry_roles,
            geometry_volume_names=tuple(volume.name for volume in volumes),
        ),
        detector=detector_spec,
    )

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.simulation import SimulationSpec, build_simulation_spec


def _build_run_manifest(spec: SimulationSpec) -> dict[str, Any]:
    return {
        "bridge": "simulation_bridge",
        "geometry_root_volume": spec.geometry.root_volume_name,
        "detector_enabled": spec.detector is not None,
        "detector_volume_name": spec.detector.volume_name if spec.detector is not None else None,
        "scoring_volume_names": list(spec.scoring.volume_names),
        "scoring_roles": {role: list(names) for role, names in spec.scoring.volume_roles.items()},
    }


def _coerce_vector3(value: Any, fallback: list[float]) -> list[float]:
    if isinstance(value, dict):
        if isinstance(value.get("value"), list) and len(value["value"]) >= 3:
            try:
                return [float(value["value"][0]), float(value["value"][1]), float(value["value"][2])]
            except (TypeError, ValueError):
                return list(fallback)
        keys = ("x", "y", "z")
        if all(key in value for key in keys):
            try:
                return [float(value["x"]), float(value["y"]), float(value["z"])]
            except (TypeError, ValueError):
                return list(fallback)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return list(fallback)
    return list(fallback)


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


def build_runtime_payload(config: dict[str, Any] | SimulationSpec) -> dict[str, Any]:
    if isinstance(config, SimulationSpec):
        spec = config
        raw_config: dict[str, Any] = {}
    else:
        raw_config = deepcopy(config) if isinstance(config, dict) else {}
        spec = build_simulation_spec(raw_config)

    payload = {
        "geometry": {
            "structure": spec.geometry.structure,
            "material": spec.geometry.material,
            "root_volume_name": spec.geometry.root_volume_name,
            "size_x_mm": spec.geometry.size_x_mm,
            "size_y_mm": spec.geometry.size_y_mm,
            "size_z_mm": spec.geometry.size_z_mm,
            "radius_mm": spec.geometry.radius_mm,
            "half_length_mm": spec.geometry.half_length_mm,
        },
        "detector": (
            {
                "enabled": True,
                "volume_name": spec.detector.volume_name,
                "material": spec.detector.material,
                "position_mm": list(spec.detector.position_mm),
                "size_x_mm": spec.detector.size_x_mm,
                "size_y_mm": spec.detector.size_y_mm,
                "size_z_mm": spec.detector.size_z_mm,
            }
            if spec.detector is not None
            else {"enabled": False}
        ),
        "source": {
            "type": spec.source.source_type,
            "particle": spec.source.particle,
            "energy_mev": spec.source.energy_mev,
            "position_mm": list(spec.source.position_mm),
            "direction_vec": list(spec.source.direction_vec),
        },
        "physics": {
            "list": spec.physics.physics_list,
        },
        "run": {
            "events": spec.run.events,
            "mode": spec.run.mode,
            "seed": spec.run.seed,
        },
        "run_manifest": _build_run_manifest(spec),
        "scoring": {
            "target_edep": spec.scoring.target_edep,
            "detector_crossings": spec.scoring.detector_crossings,
            "volume_names": list(spec.scoring.volume_names),
            "volume_roles": {role: list(names) for role, names in spec.scoring.volume_roles.items()},
        },
        # Legacy flat fields kept for compatibility with current wrappers and tests.
        "structure": spec.geometry.structure,
        "material": spec.geometry.material,
        "root_volume_name": spec.geometry.root_volume_name,
        "particle": spec.source.particle,
        "source_type": spec.source.source_type,
        "physics_list": spec.physics.physics_list,
        "energy": spec.source.energy_mev,
        "position": {
            "x": spec.source.position_mm[0],
            "y": spec.source.position_mm[1],
            "z": spec.source.position_mm[2],
        },
        "direction": {
            "x": spec.source.direction_vec[0],
            "y": spec.source.direction_vec[1],
            "z": spec.source.direction_vec[2],
        },
        "detector_enabled": spec.detector is not None,
        "raw_config": raw_config,
    }

    if spec.detector is not None:
        payload["detector_name"] = spec.detector.volume_name
        payload["detector_material"] = spec.detector.material
        payload["detector_position"] = {
            "x": spec.detector.position_mm[0],
            "y": spec.detector.position_mm[1],
            "z": spec.detector.position_mm[2],
        }
        payload["detector_size_x"] = float(spec.detector.size_x_mm)
        payload["detector_size_y"] = float(spec.detector.size_y_mm)
        payload["detector_size_z"] = float(spec.detector.size_z_mm)

    if spec.geometry.structure == "single_tubs":
        payload["radius"] = float(spec.geometry.radius_mm or 25.0)
        payload["half_length"] = float(spec.geometry.half_length_mm or 50.0)
    else:
        payload["size_x"] = float(spec.geometry.size_x_mm or 50.0)
        payload["size_y"] = float(spec.geometry.size_y_mm or 50.0)
        payload["size_z"] = float(spec.geometry.size_z_mm or 50.0)

    return payload

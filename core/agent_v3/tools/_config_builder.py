from __future__ import annotations

import re
from typing import Any


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(fallback)


def _events_from_config(config: dict[str, Any], fallback: int) -> int:
    run = config.get("run") if isinstance(config.get("run"), dict) else {}
    return _coerce_int(run.get("events"), fallback)


def _energy_mev(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(kev|mev|gev)\b", text)
    if not match:
        return 1.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "kev":
        return value / 1000.0
    if unit == "gev":
        return value * 1000.0
    return value


# DEPRECATED: legacy regex-based geometry. LLM now outputs structured volumes.
# Kept as fallback for old-format designs.
def _dimensions_mm(text: str) -> tuple[float, float, float]:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)?\s*[x×]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)?\s*[x×]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)?",
        text,
    )
    if not match:
        return (10.0, 10.0, 10.0)
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def _augment_goal_for_design(goal: str) -> str:
    text = goal.lower()
    hints: list[str] = []
    if any(token in goal for token in ("铅", "屏蔽", "透射")) or any(
        token in text for token in ("lead", "shield", "attenuation", "transmission")
    ):
        hints.append("lead shielding gamma transmission silicon detector")
    if any(token in goal for token in ("探测器", "响应", "闪烁体", "硅")) or any(
        token in text for token in ("detector", "response", "scintillator", "silicon")
    ):
        hints.append("silicon detector response detector crossing detector edep")
    if any(token in goal for token in ("剂量", "水箱", "水模", "医学", "质子")) or any(
        token in text for token in ("dose", "water phantom", "medical", "proton")
    ):
        hints.append("water phantom dose depth bins proton beam")
    if any(token in goal for token in ("无损", "缺陷", "空洞", "对比度", "管道")) or any(
        token in text for token in ("ndt", "void", "defect", "contrast", "pipe", "corrosion")
    ):
        hints.append("ndt void contrast detector transmission")
    if not hints:
        return goal
    return f"{goal}\n\nDesign hints: {'; '.join(dict.fromkeys(hints))}."


def build_recommended_config_from_design(candidate: dict[str, Any], *, events: int = 1000, accept_defaults: bool = False) -> tuple[dict[str, Any], list[str]]:
    if candidate.get("next_action") != "build_candidate_config" and not accept_defaults:
        return {}, []
    setup = candidate.get("recommended_setup") if isinstance(candidate.get("recommended_setup"), dict) else {}
    goal = str(candidate.get("goal") or "")
    low = goal.lower()

    # ── New path: LLM provided structured geometry (volumes list) ──
    geometry_raw = setup.get("geometry")
    if isinstance(geometry_raw, dict) and geometry_raw.get("volumes"):
        return _build_from_llm_geometry(candidate, setup, goal, events)

    # ── Legacy path: string geometry ID ──
    # DEPRECATED: LLM structured geometry is preferred. This path handles old-format
    # designs where geometry is a single string ID (single_box, step_wedge, etc.).
    material = str(setup.get("target_material") or setup.get("material") or "").strip() or "G4_WATER"
    environment_material = str(setup.get("environment_material") or "").strip()
    void_material = str(setup.get("void_material") or "").strip()
    geometry = str(geometry_raw or "single_box").strip() or "single_box"
    source_kind = str(setup.get("source") or "beam").strip() or "beam"
    root_name = "Target"
    structure = {
        "void": "embedded_void",
        "inclusion": "embedded_inclusion",
        "multi_layer": "multi_layer_stack",
    }.get(geometry, geometry)
    particle = "gamma"
    for name in ("neutron", "proton", "electron", "gamma"):
        if name in low:
            particle = name
            break
    energy_mev = _energy_mev(low)
    dimensions = _dimensions_mm(low)
    assumptions = [
        f"{structure} geometry is used for the runnable draft",
        "default dimensions are 10 x 10 x 10 mm unless dimensions were provided",
        "source starts at (0, 0, -20) mm and points along +z",
        "physics list defaults to FTFP_BERT",
    ]
    if environment_material:
        assumptions.append(f"environment material is represented as {environment_material}")
    detector_enabled = "detector" in " ".join(str(item) for item in candidate.get("observables", [])).lower() or bool(
        isinstance(setup.get("detector"), dict) and setup.get("detector", {}).get("enabled")
    )
    config: dict[str, Any] = {
        "geometry": {
            "structure": "single_box" if structure == "slab" else structure,
            "root_name": root_name,
            "params": {
                "module_x": dimensions[0],
                "module_y": dimensions[1],
                "module_z": dimensions[2],
            },
        },
        "materials": {
            "selected_materials": [material],
            "volume_material_map": {root_name: material},
        },
        "source": {
            "type": source_kind if source_kind in {"point", "beam", "isotropic"} else "beam",
            "particle": particle,
            "energy": energy_mev,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "physics": {"physics_list": "FTFP_BERT"},
        "output": {"format": "json"},
        "run": {"events": max(1, int(events)), "seed": 1337},
    }
    for extra_material in (environment_material, void_material):
        if extra_material and extra_material not in config["materials"]["selected_materials"]:
            config["materials"]["selected_materials"].append(extra_material)
    # If environment material is specified, create a World volume around the target
    if environment_material:
        world_size = [max(60.0, dimensions[0] * 2.0), max(60.0, dimensions[1] * 2.0), max(60.0, dimensions[2] * 2.0)]
        config["geometry"]["world"] = {
            "name": "World",
            "material": environment_material,
            "size_mm": world_size,
        }
        config["materials"]["volume_material_map"]["World"] = environment_material
        config["materials"]["world_material"] = environment_material
    if structure == "step_wedge":
        _add_step_wedge(config, material, dimensions)
    elif structure == "embedded_void":
        _add_embedded_void(config, material, void_material, dimensions, root_name)
    elif structure == "multi_layer_stack":
        _add_multi_layer_stack(config, material, dimensions)
    if detector_enabled:
        detector_material = "G4_Si"
        detector = setup.get("detector") if isinstance(setup.get("detector"), dict) else {}
        if detector.get("material"):
            detector_material = str(detector["material"])
        config["simulation"] = {
            "detector": {
                "enabled": True,
                "name": "Detector",
                "material": detector_material,
                "position": {"type": "vector", "value": [0.0, 0.0, max(40.0, dimensions[2] + 20.0)]},
                "size_triplet_mm": [max(20.0, dimensions[0]), max(20.0, dimensions[1]), 2.0],
            }
        }
        config["materials"]["volume_material_map"]["Detector"] = detector_material
        if detector_material not in config["materials"]["selected_materials"]:
            config["materials"]["selected_materials"].append(detector_material)
    config["scoring"] = _scoring_config(candidate, structure, detector_enabled, dimensions, root_name)
    return config, assumptions


def _apply_config_overrides(config: dict[str, Any], overrides: Any) -> dict[str, Any]:
    if not isinstance(overrides, dict) or not overrides:
        return {}
    applied: dict[str, Any] = {}
    source = config.get("source") if isinstance(config.get("source"), dict) else {}
    geometry = config.get("geometry") if isinstance(config.get("geometry"), dict) else {}
    geometry_params = geometry.get("params") if isinstance(geometry.get("params"), dict) else {}
    materials = config.get("materials") if isinstance(config.get("materials"), dict) else {}
    volume_material_map = materials.get("volume_material_map") if isinstance(materials.get("volume_material_map"), dict) else {}
    if "source_energy_mev" in overrides:
        try:
            energy = float(overrides["source_energy_mev"])
        except (TypeError, ValueError):
            energy = 0.0
        if energy > 0:
            source["energy"] = energy
            config["source"] = source
            applied["source_energy_mev"] = energy
    if "run_events" in overrides:
        events = _coerce_int(overrides["run_events"], _events_from_config(config, 1000))
        run = config.get("run") if isinstance(config.get("run"), dict) else {}
        run["events"] = events
        config["run"] = run
        applied["run_events"] = events
    if "target_material" in overrides:
        material = str(overrides["target_material"] or "").strip()
        if material:
            selected = materials.get("selected_materials") if isinstance(materials.get("selected_materials"), list) else []
            root_name = str(geometry.get("root_name") or "Target")
            if material not in selected:
                selected.insert(0, material)
            else:
                selected = [material] + [item for item in selected if item != material]
            materials["selected_materials"] = selected
            volume_material_map[root_name] = material
            materials["volume_material_map"] = volume_material_map
            config["materials"] = materials
            applied["target_material"] = material
    if "geometry_dimensions_mm" in overrides and isinstance(overrides["geometry_dimensions_mm"], list) and len(overrides["geometry_dimensions_mm"]) == 3:
        dims = [float(item) for item in overrides["geometry_dimensions_mm"]]
        geometry_params["module_x"], geometry_params["module_y"], geometry_params["module_z"] = dims
        geometry["params"] = geometry_params
        config["geometry"] = geometry
        applied["geometry_dimensions_mm"] = dims
    elif "target_thickness_mm" in overrides:
        try:
            thickness = float(overrides["target_thickness_mm"])
        except (TypeError, ValueError):
            thickness = 0.0
        if thickness > 0:
            geometry_params["module_z"] = thickness
            geometry["params"] = geometry_params
            config["geometry"] = geometry
            applied["target_thickness_mm"] = thickness
    return applied


def _simulation_spec_summary(spec: Any) -> dict[str, Any]:
    return {
        "geometry": {
            "structure": spec.geometry.structure,
            "material": spec.geometry.material,
            "root_volume_name": spec.geometry.root_volume_name,
            "volume_count": len(spec.geometry.volumes),
        },
        "source": {
            "type": spec.source.source_type,
            "particle": spec.source.particle,
            "energy_mev": spec.source.energy_mev,
        },
        "physics": {"list": spec.physics.physics_list},
        "run": {"events": spec.run.events, "mode": spec.run.mode, "seed": spec.run.seed},
        "detector_enabled": spec.detector is not None,
        "scoring": {
            "target_edep": spec.scoring.target_edep,
            "detector_crossings": spec.scoring.detector_crossings,
            "plane_crossings": spec.scoring.plane_crossings,
            "volume_names": list(spec.scoring.volume_names),
            "derived_metrics": list(spec.scoring.derived_metrics),
        },
    }


def _add_step_wedge(config: dict[str, Any], material: str, dimensions: tuple[float, float, float]) -> None:
    step_width = max(5.0, dimensions[0] / 3.0)
    config["geometry"]["steps"] = [
        {"name": "ThinStep", "width_mm": step_width, "thickness_mm": max(2.0, dimensions[2] * 0.25), "material": material, "role": "region_a"},
        {"name": "MidStep", "width_mm": step_width, "thickness_mm": max(4.0, dimensions[2] * 0.5), "material": material, "role": "target"},
        {"name": "ThickStep", "width_mm": step_width, "thickness_mm": max(6.0, dimensions[2]), "material": material, "role": "region_b"},
    ]
    config["materials"]["volume_material_map"].update({"ThinStep": material, "MidStep": material, "ThickStep": material})


def _add_embedded_void(
    config: dict[str, Any],
    material: str,
    void_material: str,
    dimensions: tuple[float, float, float],
    root_name: str,
) -> None:
    void_size = [max(2.0, dimensions[0] * 0.25), max(2.0, dimensions[1] * 0.25), max(2.0, dimensions[2] * 0.25)]
    config["geometry"]["volumes"] = [
        {"name": root_name, "shape": "box", "material": material, "role": "region_b", "size_mm": list(dimensions)},
        {
            "name": "VoidRegion",
            "shape": "box",
            "material": void_material or "G4_AIR",
            "role": "region_a",
            "parent": root_name,
            "size_mm": void_size,
        },
    ]
    config["materials"]["volume_material_map"]["VoidRegion"] = void_material or "G4_AIR"


def _add_multi_layer_stack(config: dict[str, Any], material: str, dimensions: tuple[float, float, float]) -> None:
    half = max(1.0, dimensions[2] * 0.5)
    second = "G4_POLYETHYLENE" if material != "G4_POLYETHYLENE" else "G4_Pb"
    config["geometry"]["layers"] = [
        {"name": "LayerA", "material": material, "thickness_mm": half, "role": "shield"},
        {"name": "LayerB", "material": second, "thickness_mm": half, "role": "shield"},
    ]
    config["materials"]["selected_materials"] = [material, second]
    config["materials"]["volume_material_map"].update({"LayerA": material, "LayerB": second})


def _scoring_config(
    candidate: dict[str, Any],
    structure: str,
    detector_enabled: bool,
    dimensions: tuple[float, float, float],
    root_name: str,
) -> dict[str, Any]:
    observables = set(str(item) for item in candidate.get("observables", []) if str(item))
    scoring: dict[str, Any] = {"target_edep": True}
    if detector_enabled:
        scoring["detector_crossings"] = True
    if "plane_crossing_count" in observables:
        scoring["plane_crossings"] = True
        scoring["plane"] = {"name": "ExitPlane", "z_mm": max(40.0, dimensions[2] + 10.0)}
    if "depth_bins" in observables:
        scoring["depth_bins"] = {"axis": "z", "bins": 20, "range_mm": [-0.5 * dimensions[2], 0.5 * dimensions[2]]}
    if structure == "step_wedge":
        scoring["volume_names"] = ["ThinStep", "MidStep", "ThickStep"]
        scoring["volume_roles"] = {"region_a": ["ThinStep"], "target": ["MidStep"], "region_b": ["ThickStep"]}
    elif structure == "embedded_void":
        scoring["volume_names"] = [root_name, "VoidRegion"]
        scoring["volume_roles"] = {"target": [root_name], "region_a": ["VoidRegion"], "region_b": [root_name]}
    elif structure == "multi_layer_stack":
        scoring["volume_names"] = ["LayerA", "LayerB"]
        scoring["volume_roles"] = {"target": ["LayerA", "LayerB"], "shield": ["LayerA", "LayerB"]}
    return scoring


def _build_from_llm_geometry(candidate: dict[str, Any], setup: dict[str, Any], goal: str, events: int) -> tuple[dict[str, Any], list[str]]:
    """Build config from LLM-structured geometry with volumes list."""
    from core.agent_v3.geometry_validator import validate_and_normalize
    geometry_spec = setup.get("geometry") if isinstance(setup.get("geometry"), dict) else {}
    world_cfg, warnings = validate_and_normalize(geometry_spec)

    volumes = world_cfg.get("volumes", [])
    if not volumes:
        return {}, ["geometry validation produced no valid volumes"]

    env_material = world_cfg.get("material", "G4_Galactic")
    primary_volume = volumes[0]
    primary_material = primary_volume.get("material", "G4_WATER")

    # Collect all materials from volumes
    selected_materials: list[str] = []
    volume_material_map: dict[str, str] = {}
    for vol in volumes:
        mat = str(vol.get("material") or "")
        name = str(vol.get("name") or "Volume")
        if mat and mat not in selected_materials:
            selected_materials.append(mat)
        volume_material_map[name] = mat
    if env_material and env_material not in selected_materials:
        selected_materials.append(env_material)

    # Source: LLM explicit choice overrides inference, otherwise infer from scenario
    explicit_source = str(setup.get("source_type") or "").strip()
    inferred_source = _infer_source_type(goal)
    source_kind = explicit_source if explicit_source in ("beam", "point", "isotropic") else inferred_source
    # Particle: use LLM extraction or infer from goal
    llm_particle = str(setup.get("source_particle") or setup.get("particle") or "").strip()
    if llm_particle:
        particle = llm_particle
    else:
        particle = "gamma"
        for name in ("neutron", "proton", "electron", "gamma", "alpha"):
            if name in goal.lower():
                particle = name
                break
    # Energy: use LLM extraction or defaults
    llm_energy = setup.get("source_energy_mev") or setup.get("energy_mev") or setup.get("energy")
    if llm_energy is not None:
        try:
            energy_mev = float(llm_energy)
        except (TypeError, ValueError):
            energy_mev = _scenario_default_energy(goal, particle)
    else:
        energy_mev = _scenario_default_energy(goal, particle)
    # Source position: LLM placement or compute from geometry
    source_placement = geometry_spec.get("source_placement") or {}
    if isinstance(source_placement, dict):
        src_pos = list(source_placement.get("position_mm", [0.0, 0.0, -200.0]))
        src_dir = list(source_placement.get("direction", [0.0, 0.0, 1.0]))
    else:
        # Compute sensible default: place source upstream of the first volume
        v0_dims = primary_volume.get("dimensions", {})
        if primary_volume.get("shape") == "sphere":
            r = float(v0_dims.get("radius_mm", 50.0))
            src_pos = [0.0, 0.0, -(r + 50.0)]
        elif primary_volume.get("shape") in ("tubs", "cylinder"):
            hl = float(v0_dims.get("half_length_mm", 50.0))
            src_pos = [0.0, 0.0, -(hl + 50.0)]
        else:
            z = float(v0_dims.get("size_z_mm", 10.0))
            src_pos = [0.0, 0.0, -(z / 2.0 + 50.0)]
        src_dir = [0.0, 0.0, 1.0]

    # Normalize volumes to bridge-compatible format (top-level keys, not dimensions dict)
    bridge_volumes = [_normalize_for_bridge(v) for v in volumes]

    # Build geometry config: pass volumes + world directly
    geo_config: dict[str, Any] = {
        "structure": "llm_structured",
        "root_name": primary_volume.get("name", "Target"),
        "params": _flatten_volume_dims(primary_volume),
        "world": {"name": "World", "material": env_material, "size_mm": world_cfg.get("size_mm", [100, 100, 100])},
        "volumes": bridge_volumes,
        "llm_geometry": True,
    }

    detector_enabled = "detector" in " ".join(str(item) for item in candidate.get("observables", [])).lower()

    # Physics list: use LLM choice if provided, else default
    physics_list = str(setup.get("physics_list") or setup.get("source_physics_list") or "FTFP_BERT").strip()
    if physics_list not in ("FTFP_BERT", "Shielding", "QGSP_BERT_HP", "FTFP_BERT_HP", "QGSP_BIC",
                            "FTFP_BERT_ATL", "FTFP_INCLXX", "QGSP_BERT", "QGSP_BIC", "QGS_BIC"):
        physics_list = "FTFP_BERT"

    config: dict[str, Any] = {
        "geometry": geo_config,
        "materials": {"selected_materials": selected_materials, "volume_material_map": volume_material_map},
        "source": {
            "type": source_kind if source_kind in {"point", "beam", "isotropic"} else "beam",
            "particle": particle,
            "energy": energy_mev,
            "position": {"type": "vector", "value": list(src_pos)},
            "direction": {"type": "vector", "value": list(src_dir)},
        },
        "physics": {"physics_list": physics_list},
        "output": {"format": "json"},
        "run": {"events": max(1, int(events)), "seed": 1337},
    }

    if detector_enabled:
        det_material = str(setup.get("detector_material") or (setup.get("detector") or {}).get("material") or "G4_Si").strip()
        # Detector position: downstream of target
        v0_dims = primary_volume.get("dimensions", {})
        if primary_volume.get("shape") == "sphere":
            det_z = float(v0_dims.get("radius_mm", 50.0)) + 20.0
        elif primary_volume.get("shape") in ("tubs", "cylinder"):
            det_z = float(v0_dims.get("half_length_mm", 50.0)) + 20.0
        else:
            det_z = float(v0_dims.get("size_z_mm", 10.0)) / 2.0 + 20.0
        # Detector size: match target cross-section
        if primary_volume.get("shape") == "sphere":
            r = float(v0_dims.get("radius_mm", 50.0))
            det_x, det_y = r * 0.5, r * 0.5
        elif primary_volume.get("shape") in ("tubs", "cylinder"):
            r = float(v0_dims.get("radius_mm", 25.0))
            det_x, det_y = r * 2.0, r * 2.0
        else:
            det_x = float(v0_dims.get("size_x_mm", 10.0))
            det_y = float(v0_dims.get("size_y_mm", 10.0))
        config["simulation"] = {"detector": {"enabled": True, "name": "Detector", "material": det_material,
            "position": {"type": "vector", "value": [0.0, 0.0, det_z]},
            "size_triplet_mm": [det_x, det_y, 2.0]}}
        if det_material not in selected_materials:
            config["materials"]["selected_materials"].append(det_material)
        config["materials"]["volume_material_map"]["Detector"] = det_material

    # Scoring: derive from observables and geometry
    observables = candidate.get("observables") or []
    scoring: dict[str, Any] = {"target_edep": True}
    if detector_enabled or any("detector" in str(o).lower() for o in observables):
        scoring["detector_crossings"] = True
    if any("plane" in str(o).lower() or "crossing" in str(o).lower() or "transmission" in str(o).lower() for o in observables):
        scoring["plane_crossings"] = True
        scoring["plane"] = {"name": "ExitPlane", "z_mm": _compute_plane_z(primary_volume)}
    if any("depth" in str(o).lower() or "dose" in str(o).lower() or "profile" in str(o).lower() for o in observables):
        z_range = _compute_depth_range(primary_volume)
        scoring["depth_bins"] = {"axis": "z", "bins": 20, "range_mm": z_range}
    if any("contrast" in str(o).lower() or "void" in str(o).lower() or "inclusion" in str(o).lower() for o in observables):
        scoring["region_contrast"] = True
    # Volume names and roles from LLM volumes
    vol_names = [str(v.get("name") or "") for v in volumes if v.get("name")]
    if vol_names:
        scoring["volume_names"] = vol_names
    # Build role map from volume properties
    roles: dict[str, list[str]] = {}
    for v in volumes:
        name = str(v.get("name") or "")
        role = str(v.get("role") or "").strip().lower()
        if role and name:
            roles.setdefault(role, []).append(name)
    if not roles:
        roles["target"] = vol_names
    scoring["volume_roles"] = roles
    config["scoring"] = scoring

    assumptions = [f"LLM-structured geometry with {len(volumes)} volume(s)"]
    assumptions.extend(warnings)
    return config, assumptions


def _compute_plane_z(vol: dict[str, Any]) -> float:
    """Compute z-position for scoring plane (just behind the target volume)."""
    dims = vol.get("dimensions", {})
    shape = vol.get("shape", "box")
    if shape == "sphere":
        return float(dims.get("radius_mm", 50.0)) + 10.0
    if shape in ("tubs", "cylinder"):
        return float(dims.get("half_length_mm", 50.0)) + 10.0
    return float(dims.get("size_z_mm", 10.0)) / 2.0 + 10.0


def _compute_depth_range(vol: dict[str, Any]) -> list[float]:
    """Compute z-range for depth bins (through the target volume)."""
    dims = vol.get("dimensions", {})
    shape = vol.get("shape", "box")
    if shape == "sphere":
        r = float(dims.get("radius_mm", 50.0))
        return [-r, r]
    if shape in ("tubs", "cylinder"):
        hl = float(dims.get("half_length_mm", 50.0))
        return [-hl, hl]
    z = float(dims.get("size_z_mm", 10.0))
    return [-z / 2.0, z / 2.0]


def _infer_source_type(goal: str) -> str:
    """Infer source type from physics scenario."""
    g = goal.lower()
    if any(t in g for t in ("空间", "太空", "宇宙", "space", "isotropic", "各向同性", "astronaut", "宇航")):
        return "isotropic"
    if any(t in g for t in ("点源", "point", "check source", "校准")):
        return "point"
    return "beam"


def _scenario_default_energy(goal: str, particle: str) -> float:
    """Provide scenario-appropriate default energy."""
    g = goal.lower()
    if particle == "proton" or any(t in g for t in ("质子", "proton", "治疗", "medical", "剂量", "dose")):
        return 150.0
    if particle == "neutron" or any(t in g for t in ("中子", "neutron")):
        return 2.0
    if any(t in g for t in ("ndt", "无损", "检测", "defect", "void", "空洞", "夹杂")):
        return 0.5
    if any(t in g for t in ("kev", "keV")):
        return 0.1
    if any(t in g for t in ("gev", "GeV")):
        return 10.0
    return 1.0


def _normalize_for_bridge(vol: dict[str, Any]) -> dict[str, Any]:
    """Convert LLM volume format (dimensions dict) to bridge-compatible format (top-level keys)."""
    out = {k: v for k, v in vol.items() if k != "dimensions"}
    dims = vol.get("dimensions") if isinstance(vol.get("dimensions"), dict) else {}
    shape = str(vol.get("shape") or "box")
    # Box → size_mm triplet
    if shape == "box":
        out["size_mm"] = [
            float(dims.get("size_x_mm", 10.0)),
            float(dims.get("size_y_mm", 10.0)),
            float(dims.get("size_z_mm", 10.0)),
        ]
    # Sphere → radius_mm
    if shape == "sphere":
        out["radius_mm"] = float(dims.get("radius_mm", 50.0))
    # Tubs/Cylinder → radius_mm + half_length_mm
    if shape in ("tubs", "cylinder"):
        out["radius_mm"] = float(dims.get("radius_mm", 25.0))
        out["half_length_mm"] = float(dims.get("half_length_mm", 50.0))
    # Cons → radius1/2 + half_length
    if shape == "cons":
        out["outer_radius_mm"] = float(dims.get("radius1_mm", 20.0))
        out["inner_radius_mm"] = float(dims.get("radius2_mm", 30.0))
        out["half_length_mm"] = float(dims.get("half_length_mm", 50.0))
    # TRD → 5 params
    if shape == "trd":
        out["size_mm"] = [
            float(dims.get("x1_mm", 10.0)),
            float(dims.get("y1_mm", 10.0)),
            float(dims.get("z_mm", 30.0)),
        ]
    # Fallback for unknown shapes: extract whatever dimensions exist
    if shape not in ("box", "sphere", "tubs", "cylinder", "cons", "trd"):
        size_x = float(dims.get("size_x_mm", dims.get("module_x", 10.0)))
        size_y = float(dims.get("size_y_mm", dims.get("module_y", 10.0)))
        size_z = float(dims.get("size_z_mm", dims.get("module_z", 10.0)))
        out["size_mm"] = [size_x, size_y, size_z]
    return out


def _flatten_volume_dims(vol: dict[str, Any]) -> dict[str, Any]:
    """Extract box-compatible params from a volume for legacy bridge compatibility."""
    dims = vol.get("dimensions", {})
    shape = vol.get("shape", "box")
    result: dict[str, Any] = {}
    if shape == "box":
        result["module_x"] = dims.get("size_x_mm", 10.0)
        result["module_y"] = dims.get("size_y_mm", 10.0)
        result["module_z"] = dims.get("size_z_mm", 10.0)
    elif shape in ("tubs", "cylinder"):
        result["child_rmax"] = dims.get("radius_mm", 25.0)
        result["child_hz"] = dims.get("half_length_mm", 50.0)
        result["module_x"] = float(dims.get("radius_mm", 25.0)) * 2.0
        result["module_y"] = float(dims.get("radius_mm", 25.0)) * 2.0
        result["module_z"] = float(dims.get("half_length_mm", 50.0)) * 2.0
    elif shape == "sphere":
        r = dims.get("radius_mm", 50.0)
        result["child_rmax"] = r
        result["module_x"] = float(r) * 2.0
        result["module_y"] = float(r) * 2.0
        result["module_z"] = float(r) * 2.0
    else:
        # Unknown shape: extract whatever dimensions are available
        for k, v in dims.items():
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                result[k] = 10.0
    return result


__all__ = [
    "build_recommended_config_from_design",
    "_augment_goal_for_design",
    "_apply_config_overrides",
    "_coerce_int",
    "_events_from_config",
    "_simulation_spec_summary",
]

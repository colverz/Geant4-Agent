"""Thin validation layer for LLM-generated geometry specifications.

Does NOT do dictionary mapping of shapes or materials.
LLM is trusted to produce valid Geant4 parameters; this only checks safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Load known materials once
def _load_known_materials() -> set[str]:
    try:
        data = json.loads(Path("knowledge/data/materials_geant4_nist.json").read_text(encoding="utf-8"))
        return set(data.get("materials", []))
    except Exception:
        return set()

KNOWN_MATERIALS: set[str] = _load_known_materials()

SHAPE_DIMENSION_KEYS = {
    "box": ("size_x_mm", "size_y_mm", "size_z_mm"),
    "sphere": ("radius_mm",),
    "tubs": ("radius_mm", "half_length_mm"),
    "cylinder": ("radius_mm", "half_length_mm"),
    "cons": ("radius1_mm", "radius2_mm", "half_length_mm"),
    "trd": ("x1_mm", "x2_mm", "y1_mm", "y2_mm", "z_mm"),
}

SHAPE_DEFAULTS = {
    "box": {"size_x_mm": 10.0, "size_y_mm": 10.0, "size_z_mm": 10.0},
    "sphere": {"radius_mm": 50.0},
    "tubs": {"radius_mm": 25.0, "half_length_mm": 50.0},
    "cylinder": {"radius_mm": 25.0, "half_length_mm": 50.0},
    "cons": {"radius1_mm": 20.0, "radius2_mm": 30.0, "half_length_mm": 50.0},
    "trd": {"x1_mm": 10.0, "x2_mm": 20.0, "y1_mm": 10.0, "y2_mm": 20.0, "z_mm": 30.0},
}

SAFE_BOX = {"name": "Target", "shape": "box", "material": "G4_WATER",
            "dimensions": {"size_x_mm": 10, "size_y_mm": 10, "size_z_mm": 10}}


def validate_and_normalize(geometry_spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate LLM geometry output and return (normalized_config, warnings).

    Checks: required fields, positive dimensions, known materials, world sizing.
    Does NOT restrict shapes to a fixed list — LLM is free to use any shape name.
    Unknown shapes get box default dimensions.
    """
    warnings: list[str] = []
    volumes = geometry_spec.get("volumes")
    if not isinstance(volumes, list) or not volumes:
        return _fallback_config(), ["geometry missing 'volumes' list"]

    normalized_volumes = []
    max_extent = 0.0
    for i, vol in enumerate(volumes):
        if not isinstance(vol, dict):
            warnings.append(f"volume[{i}] is not a dict, skipping")
            continue
        nvol, vw = _normalize_volume(vol, i)
        normalized_volumes.append(nvol)
        warnings.extend(vw)
        extent = _volume_extent(nvol)
        if extent > max_extent:
            max_extent = extent

    env_material = _env_material(geometry_spec)
    if env_material not in KNOWN_MATERIALS and env_material:
        warnings.append(f"environment material '{env_material}' not in known G4 materials")

    world_size = max(max_extent * 2.5, 100.0)
    world_config = {
        "name": "World",
        "material": env_material,
        "size_mm": [world_size, world_size, world_size],
        "volumes": normalized_volumes,
    }

    if not env_material:
        warnings.append("no environment material specified, defaulting to G4_Galactic")
        world_config["material"] = "G4_Galactic"

    return world_config, warnings


def _normalize_volume(vol: dict[str, Any], index: int) -> tuple[dict[str, Any], list[str]]:
    w: list[str] = []
    name = str(vol.get("name") or f"Volume{index}")
    shape = str(vol.get("shape") or "box").strip().lower()
    material = str(vol.get("material") or "G4_WATER").strip()

    if material not in KNOWN_MATERIALS and material:
        w.append(f"volume '{name}': material '{material}' not in known G4 list")

    dims = vol.get("dimensions")
    if not isinstance(dims, dict):
        dims = {}

    # Use shape-appropriate defaults for any missing dimensions
    expected_keys = SHAPE_DIMENSION_KEYS.get(shape, ("size_x_mm", "size_y_mm", "size_z_mm"))
    defaults = SHAPE_DEFAULTS.get(shape, SHAPE_DEFAULTS["box"])
    normalized_dims: dict[str, float] = {}
    for key in expected_keys:
        val = dims.get(key, defaults.get(key, 10.0))
        try:
            fval = float(val)
        except (TypeError, ValueError):
            fval = float(defaults.get(key, 10.0))
            w.append(f"volume '{name}': invalid dimension {key}={val}, using {fval}")
        if fval <= 0:
            fval = float(defaults.get(key, 10.0))
            w.append(f"volume '{name}': non-positive {key}={val}, using {fval}")
        normalized_dims[key] = fval

    position = vol.get("position_mm") or vol.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        pos = [float(position[0]), float(position[1]), float(position[2])]
    else:
        pos = [0.0, 0.0, 0.0]

    return {
        "name": name,
        "shape": shape,
        "material": material,
        "dimensions": normalized_dims,
        "position_mm": pos,
    }, w


def _volume_extent(vol: dict[str, Any]) -> float:
    """Estimate the maximum spatial extent of a volume for world sizing."""
    dims = vol.get("dimensions", {})
    shape = vol.get("shape", "box")
    if shape in ("sphere",):
        return abs(float(dims.get("radius_mm", 50.0))) * 2.2
    if shape in ("tubs", "cylinder"):
        r = abs(float(dims.get("radius_mm", 25.0)))
        h = abs(float(dims.get("half_length_mm", 50.0))) * 2
        return max(r * 2.2, h * 1.2)
    # box and others: use max dimension
    vals = [abs(float(v)) for v in dims.values() if isinstance(v, (int, float))]
    return max(vals) * 1.5 if vals else 100.0


def _env_material(spec: dict[str, Any]) -> str:
    env = spec.get("environment")
    if isinstance(env, dict):
        return str(env.get("material") or "G4_Galactic")
    return "G4_Galactic"


def _fallback_config() -> dict[str, Any]:
    return {
        "name": "World",
        "material": "G4_Galactic",
        "size_mm": [100.0, 100.0, 100.0],
        "volumes": [SAFE_BOX],
    }

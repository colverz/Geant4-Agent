from __future__ import annotations

from core.orchestrator.path_ops import get_path
from core.orchestrator.types import Phase
from core.validation.geometry_registry import get_geometry_family


_DEFAULT_GEOMETRY_REQUIRED = [
    "geometry.structure",
    "geometry.params.module_x",
    "geometry.params.module_y",
    "geometry.params.module_z",
]

_SHARED_REQUIRED = [
    "materials.volume_material_map",
    "source.type",
    "source.particle",
    "source.energy",
    "source.position",
    "physics.physics_list",
    "output.format",
    "output.path",
]


def _geometry_required_paths(config: dict | None = None) -> list[str]:
    if not isinstance(config, dict):
        return list(_DEFAULT_GEOMETRY_REQUIRED)
    structure = str(get_path(config, "geometry.structure", "") or "")
    family = get_geometry_family(structure)
    family_required = sorted(family.get("required_paths", set()))
    if not structure or not family_required:
        return list(_DEFAULT_GEOMETRY_REQUIRED)
    return ["geometry.structure", *family_required]


def get_minimal_required_paths(config: dict | None = None) -> list[str]:
    source_required = [
        "source.type",
        "source.particle",
        "source.energy",
        "source.position",
    ]
    source_type = str(get_path(config or {}, "source.type", "") or "").lower()
    if source_type in {"point", "beam", "plane", ""}:
        source_required.append("source.direction")
    return _geometry_required_paths(config) + [
        "materials.volume_material_map",
        *source_required,
        "physics.physics_list",
        "output.format",
        "output.path",
    ]


def get_local_required_paths(phase: Phase, *, config: dict | None = None) -> list[str]:
    if phase == Phase.GEOMETRY:
        return _geometry_required_paths(config)
    if phase == Phase.MATERIALS:
        return ["materials.volume_material_map"]
    if phase == Phase.SOURCE:
        required = [
            "source.type",
            "source.particle",
            "source.energy",
            "source.position",
        ]
        source_type = str(get_path(config or {}, "source.type", "") or "").lower()
        if source_type in {"point", "beam", "plane", ""}:
            required.append("source.direction")
        return required
    if phase == Phase.PHYSICS:
        return ["physics.physics_list"]
    if phase == Phase.OUTPUT:
        return ["output.format", "output.path"]
    return []

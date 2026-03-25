from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.domain.geometry import GEOMETRY_KIND_TO_STRUCTURE
from core.domain.geometry_family import GEOMETRY_FAMILY_REGISTRY


@dataclass(frozen=True)
class GeometryParamDefinition:
    name: str
    slot_fields: tuple[str, ...]
    config_param_keys: tuple[str, ...] = ()
    runtime_aliases: tuple[str, ...] = ()
    required: bool = False
    unit: str = "mm"
    value_kind: str = "scalar"
    arity: int = 1
    description: str = ""

    def normalize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if self.value_kind == "triplet":
            if isinstance(value, (list, tuple)) and len(value) == self.arity:
                return [float(value[0]), float(value[1]), float(value[2])]
            return None
        if self.value_kind == "integer":
            return int(value)
        return float(value)


@dataclass(frozen=True)
class GeometryCatalogEntry:
    structure: str
    user_kind_aliases: tuple[str, ...]
    structure_aliases: tuple[str, ...]
    required_slot_fields: tuple[str, ...]
    params: tuple[GeometryParamDefinition, ...]
    allowed_paths: frozenset[str] = field(default_factory=frozenset)
    required_paths: frozenset[str] = field(default_factory=frozenset)


_CATALOG: dict[str, GeometryCatalogEntry] = {
    "single_box": GeometryCatalogEntry(
        structure="single_box",
        user_kind_aliases=("box", "cube"),
        structure_aliases=("single_box",),
        required_slot_fields=("kind", "size_triplet_mm"),
        params=(
            GeometryParamDefinition(
                name="size_triplet_mm",
                slot_fields=("size_triplet_mm",),
                config_param_keys=("module_x", "module_y", "module_z"),
                runtime_aliases=("size_x", "size_y", "size_z"),
                required=True,
                value_kind="triplet",
                arity=3,
                description="Full edge lengths along x/y/z in millimetres.",
            ),
        ),
        allowed_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_box"]["allowed_paths"]),
        required_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_box"]["required_paths"]),
    ),
    "single_tubs": GeometryCatalogEntry(
        structure="single_tubs",
        user_kind_aliases=("cylinder", "tubs"),
        structure_aliases=("single_tubs",),
        required_slot_fields=("kind", "radius_mm", "half_length_mm"),
        params=(
            GeometryParamDefinition(
                name="radius_mm",
                slot_fields=("radius_mm",),
                config_param_keys=("child_rmax",),
                runtime_aliases=("radius",),
                required=True,
                description="Outer radius in millimetres.",
            ),
            GeometryParamDefinition(
                name="half_length_mm",
                slot_fields=("half_length_mm",),
                config_param_keys=("child_hz",),
                runtime_aliases=("half_length",),
                required=True,
                description="Half length along z in millimetres.",
            ),
        ),
        allowed_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_tubs"]["allowed_paths"]),
        required_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_tubs"]["required_paths"]),
    ),
    "single_orb": GeometryCatalogEntry(
        structure="single_orb",
        user_kind_aliases=("orb",),
        structure_aliases=("single_orb",),
        required_slot_fields=("kind", "radius_mm"),
        params=(
            GeometryParamDefinition(
                name="radius_mm",
                slot_fields=("radius_mm",),
                config_param_keys=("child_rmax",),
                runtime_aliases=("radius",),
                required=True,
                description="Orb radius in millimetres.",
            ),
        ),
        allowed_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_orb"]["allowed_paths"]),
        required_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_orb"]["required_paths"]),
    ),
    "single_cons": GeometryCatalogEntry(
        structure="single_cons",
        user_kind_aliases=("cons",),
        structure_aliases=("single_cons",),
        required_slot_fields=("kind", "radius1_mm", "radius2_mm", "half_length_mm"),
        params=(
            GeometryParamDefinition(
                name="radius1_mm",
                slot_fields=("radius1_mm",),
                config_param_keys=("rmax1",),
                required=True,
                description="Lower outer radius in millimetres.",
            ),
            GeometryParamDefinition(
                name="radius2_mm",
                slot_fields=("radius2_mm",),
                config_param_keys=("rmax2",),
                required=True,
                description="Upper outer radius in millimetres.",
            ),
            GeometryParamDefinition(
                name="half_length_mm",
                slot_fields=("half_length_mm",),
                config_param_keys=("child_hz",),
                runtime_aliases=("half_length",),
                required=True,
                description="Half length along z in millimetres.",
            ),
        ),
        allowed_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_cons"]["allowed_paths"]),
        required_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_cons"]["required_paths"]),
    ),
    "single_trd": GeometryCatalogEntry(
        structure="single_trd",
        user_kind_aliases=("trd",),
        structure_aliases=("single_trd",),
        required_slot_fields=("kind", "x1_mm", "x2_mm", "y1_mm", "y2_mm", "z_mm"),
        params=(
            GeometryParamDefinition(
                name="x1_mm",
                slot_fields=("x1_mm",),
                config_param_keys=("x1",),
                required=True,
                description="Half x-length at -z face in millimetres.",
            ),
            GeometryParamDefinition(
                name="x2_mm",
                slot_fields=("x2_mm",),
                config_param_keys=("x2",),
                required=True,
                description="Half x-length at +z face in millimetres.",
            ),
            GeometryParamDefinition(
                name="y1_mm",
                slot_fields=("y1_mm",),
                config_param_keys=("y1",),
                required=True,
                description="Half y-length at -z face in millimetres.",
            ),
            GeometryParamDefinition(
                name="y2_mm",
                slot_fields=("y2_mm",),
                config_param_keys=("y2",),
                required=True,
                description="Half y-length at +z face in millimetres.",
            ),
            GeometryParamDefinition(
                name="z_mm",
                slot_fields=("z_mm",),
                config_param_keys=("module_z",),
                required=True,
                description="Half z-length in millimetres.",
            ),
        ),
        allowed_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_trd"]["allowed_paths"]),
        required_paths=frozenset(GEOMETRY_FAMILY_REGISTRY["single_trd"]["required_paths"]),
    ),
}


def resolve_geometry_structure(kind_or_structure: str | None) -> str | None:
    text = str(kind_or_structure or "").strip().lower()
    if not text:
        return None
    if text in _CATALOG:
        return text
    for structure, entry in _CATALOG.items():
        if text in entry.structure_aliases:
            return structure
    mapped = GEOMETRY_KIND_TO_STRUCTURE.get(text)
    if mapped in _CATALOG:
        return mapped
    for structure, entry in _CATALOG.items():
        if text in entry.user_kind_aliases:
            return structure
    return None


def get_geometry_catalog_entry(structure: str | None) -> GeometryCatalogEntry | None:
    resolved = resolve_geometry_structure(structure)
    if not resolved:
        return None
    return _CATALOG.get(resolved)


def iter_geometry_catalog() -> tuple[GeometryCatalogEntry, ...]:
    return tuple(_CATALOG.values())

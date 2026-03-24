from __future__ import annotations

from core.domain.geometry_family import GEOMETRY_FAMILY_REGISTRY, get_geometry_family
from core.orchestrator.path_ops import flatten, get_path, remove_path
from core.validation.error_codes import E_SCOPE_LEAK


def prune_out_of_scope_params(config: dict, family_registry: dict | None = None) -> tuple[dict, list[dict]]:
    registry = family_registry or GEOMETRY_FAMILY_REGISTRY
    structure = str(get_path(config, "geometry.structure", "") or "")
    family = registry.get(structure, {"allowed_paths": set()})
    allowed = set(family.get("allowed_paths", set()))

    params = get_path(config, "geometry.params", {})
    if not isinstance(params, dict):
        return config, []
    flat = flatten({"geometry": {"params": params}})
    errors: list[dict] = []
    for path in flat.keys():
        if path not in allowed:
            remove_path(config, path)
            errors.append(
                {
                    "code": E_SCOPE_LEAK,
                    "path": path,
                    "message": f"param not allowed for structure={structure}",
                }
            )
    return config, errors


__all__ = ["GEOMETRY_FAMILY_REGISTRY", "get_geometry_family", "prune_out_of_scope_params"]

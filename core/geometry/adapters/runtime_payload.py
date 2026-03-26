from __future__ import annotations

from core.geometry.catalog import get_geometry_catalog_entry
from core.geometry.spec import GeometrySpec


def geometry_spec_to_runtime_geometry(spec: GeometrySpec) -> dict[str, object]:
    if not spec.runtime_ready:
        raise ValueError(f"geometry_spec_not_runtime_ready:{spec.structure}:{spec.finalization_status}")
    payload: dict[str, object] = {"structure": spec.structure}
    entry = get_geometry_catalog_entry(spec.structure)
    if entry is None:
        return payload

    for param in entry.params:
        value = spec.params.get(param.name)
        if value is None:
            continue
        if param.name.endswith("_triplet_mm") and isinstance(value, list) and len(param.runtime_aliases) == 3:
            for alias, item in zip(param.runtime_aliases, value):
                payload[alias] = float(item)
            continue
        alias = param.runtime_aliases[0] if param.runtime_aliases else None
        if alias:
            payload[alias] = float(value)
    return payload

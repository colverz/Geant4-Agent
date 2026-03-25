from __future__ import annotations

from typing import Any

from core.geometry.adapters.config_fragment import geometry_spec_to_config_fragment
from core.geometry.spec import GeometrySpec


def diff_geometry_config_fragment(
    spec: GeometrySpec,
    legacy_geometry_config: dict[str, Any] | None,
) -> dict[str, Any]:
    fragment = geometry_spec_to_config_fragment(spec).get("geometry", {})
    legacy_geometry = legacy_geometry_config if isinstance(legacy_geometry_config, dict) else {}
    new_params = fragment.get("params", {}) if isinstance(fragment.get("params"), dict) else {}
    old_params = legacy_geometry.get("params", {}) if isinstance(legacy_geometry.get("params"), dict) else {}

    mismatches: list[dict[str, Any]] = []
    all_param_keys = sorted(set(new_params.keys()) | set(old_params.keys()))
    for key in all_param_keys:
        old_value = old_params.get(key)
        new_value = new_params.get(key)
        if old_value != new_value:
            mismatches.append({"field": f"geometry.params.{key}", "legacy": old_value, "new": new_value})

    if legacy_geometry.get("structure") != fragment.get("structure"):
        mismatches.append(
            {
                "field": "geometry.structure",
                "legacy": legacy_geometry.get("structure"),
                "new": fragment.get("structure"),
            }
        )

    return {
        "matches": not mismatches,
        "new_geometry": fragment,
        "legacy_geometry": legacy_geometry,
        "mismatches": mismatches,
    }


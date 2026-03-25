from __future__ import annotations

from core.geometry.spec import GeometrySpec


def geometry_spec_to_runtime_geometry(spec: GeometrySpec) -> dict[str, object]:
    payload: dict[str, object] = {"structure": spec.structure}

    if spec.structure == "single_box":
        triplet = spec.params.get("size_triplet_mm")
        if isinstance(triplet, list) and len(triplet) == 3:
            payload["size_x"] = float(triplet[0])
            payload["size_y"] = float(triplet[1])
            payload["size_z"] = float(triplet[2])
    elif spec.structure == "single_tubs":
        if "radius_mm" in spec.params:
            payload["radius"] = float(spec.params["radius_mm"])
        if "half_length_mm" in spec.params:
            payload["half_length"] = float(spec.params["half_length_mm"])

    return payload


from __future__ import annotations

from typing import Any


INDUSTRIAL_RUNTIME_CONTRACT_PATHS = (
    "geometry.structure",
    "geometry.material",
    "geometry.root_volume_name",
    "geometry.size_x_mm",
    "geometry.size_y_mm",
    "geometry.size_z_mm",
    "detector.enabled",
    "detector.volume_name",
    "detector.material",
    "detector.position_mm",
    "detector.size_x_mm",
    "detector.size_y_mm",
    "detector.size_z_mm",
    "source.type",
    "source.particle",
    "source.energy_mev",
    "source.position_mm",
    "source.direction_vec",
    "physics.list",
    "run.events",
    "run.seed",
    "scoring.target_edep",
    "scoring.detector_crossings",
    "scoring.plane_crossings",
    "scoring.plane.z_mm",
)


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return _numeric_equal(left, right)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_value_equal(l_item, r_item) for l_item, r_item in zip(left, right))
    return left == right


def compare_candidate_runtime_contract(
    candidate_payload: dict[str, Any],
    expected_payload: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for path in INDUSTRIAL_RUNTIME_CONTRACT_PATHS:
        expected = _get_path(expected_payload, path)
        candidate = _get_path(candidate_payload, path)
        if expected is None and candidate is None:
            continue
        if not _value_equal(candidate, expected):
            mismatches.append({"path": path, "expected": expected, "actual": candidate})
    return {
        "ok": not mismatches,
        "checked_paths": list(INDUSTRIAL_RUNTIME_CONTRACT_PATHS),
        "mismatches": mismatches,
    }


__all__ = ["INDUSTRIAL_RUNTIME_CONTRACT_PATHS", "compare_candidate_runtime_contract"]

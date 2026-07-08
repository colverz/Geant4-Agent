from __future__ import annotations

from dataclasses import asdict, dataclass
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


V3_INDUSTRIAL_CANDIDATE_CONTRACT_SCHEMA_VERSION = "geant4_agent_v3_industrial_candidate_contract.v1"


@dataclass(frozen=True, slots=True)
class V3IndustrialCandidateRequirements:
    target_material: str
    target_thickness_mm: float
    source_type: str
    particle: str
    energy_mev: float
    direction_vec: tuple[float, float, float]
    detector_required: bool
    detector_material: str | None
    target_edep_required: bool
    detector_crossings_required: bool
    plane_crossings_required: bool
    physics_list: str
    events: int
    seed: int
    schema_version: str = V3_INDUSTRIAL_CANDIDATE_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_runtime_payload(cls, payload: dict[str, Any]) -> "V3IndustrialCandidateRequirements":
        direction = _vector3(_get_path(payload, "source.direction_vec"), fallback=(0.0, 0.0, 1.0))
        detector_required = bool(_get_path(payload, "detector.enabled"))
        return cls(
            target_material=str(_get_path(payload, "geometry.material") or ""),
            target_thickness_mm=float(_get_path(payload, "geometry.size_z_mm") or 0.0),
            source_type=str(_get_path(payload, "source.type") or ""),
            particle=str(_get_path(payload, "source.particle") or ""),
            energy_mev=float(_get_path(payload, "source.energy_mev") or 0.0),
            direction_vec=direction,
            detector_required=detector_required,
            detector_material=str(_get_path(payload, "detector.material") or "") if detector_required else None,
            target_edep_required=bool(_get_path(payload, "scoring.target_edep")),
            detector_crossings_required=bool(_get_path(payload, "scoring.detector_crossings")),
            plane_crossings_required=bool(_get_path(payload, "scoring.plane_crossings")),
            physics_list=str(_get_path(payload, "physics.list") or ""),
            events=int(_get_path(payload, "run.events") or 0),
            seed=int(_get_path(payload, "run.seed") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _vector3(value: Any, *, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return fallback
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return fallback


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


def compare_v3_candidate_runtime_contract(
    candidate_payload: dict[str, Any],
    expected_payload: dict[str, Any],
) -> dict[str, Any]:
    requirements = V3IndustrialCandidateRequirements.from_runtime_payload(expected_payload)
    mismatches: list[dict[str, Any]] = []

    def require_equal(field: str, actual: Any, expected: Any) -> None:
        if not _value_equal(actual, expected):
            mismatches.append({"field": field, "expected": expected, "actual": actual})

    require_equal("target.material", _get_path(candidate_payload, "geometry.material"), requirements.target_material)
    require_equal(
        "target.thickness_mm",
        _get_path(candidate_payload, "geometry.size_z_mm"),
        requirements.target_thickness_mm,
    )
    require_equal("source.type", _get_path(candidate_payload, "source.type"), requirements.source_type)
    require_equal("source.particle", _get_path(candidate_payload, "source.particle"), requirements.particle)
    require_equal("source.energy_mev", _get_path(candidate_payload, "source.energy_mev"), requirements.energy_mev)
    require_equal("source.direction_vec", _get_path(candidate_payload, "source.direction_vec"), list(requirements.direction_vec))
    require_equal("physics.list", _get_path(candidate_payload, "physics.list"), requirements.physics_list)
    require_equal("run.events", _get_path(candidate_payload, "run.events"), requirements.events)
    require_equal("run.seed", _get_path(candidate_payload, "run.seed"), requirements.seed)

    detector_enabled = bool(_get_path(candidate_payload, "detector.enabled"))
    require_equal("detector.required", detector_enabled, requirements.detector_required)
    if requirements.detector_required:
        require_equal("detector.material", _get_path(candidate_payload, "detector.material"), requirements.detector_material)

    required_scoring = {
        "scoring.target_edep": requirements.target_edep_required,
        "scoring.detector_crossings": requirements.detector_crossings_required,
        "scoring.plane_crossings": requirements.plane_crossings_required,
    }
    for field, required in required_scoring.items():
        if required and not bool(_get_path(candidate_payload, field)):
            mismatches.append({"field": field, "expected": True, "actual": _get_path(candidate_payload, field)})

    _check_z_order(candidate_payload, requirements, mismatches)
    return {
        "schema_version": V3_INDUSTRIAL_CANDIDATE_CONTRACT_SCHEMA_VERSION,
        "ok": not mismatches,
        "requirements": requirements.to_dict(),
        "mismatches": mismatches,
        "allowed_variations": [
            "geometry.root_volume_name",
            "geometry.transverse_size",
            "source.upstream_distance",
            "detector.downstream_distance",
            "detector.transverse_size",
            "additional_scoring",
        ],
    }


def _check_z_order(
    candidate_payload: dict[str, Any],
    requirements: V3IndustrialCandidateRequirements,
    mismatches: list[dict[str, Any]],
) -> None:
    direction = _vector3(_get_path(candidate_payload, "source.direction_vec"), fallback=(0.0, 0.0, 0.0))
    if not _value_equal(list(direction), list(requirements.direction_vec)) or direction[2] <= 0:
        return
    source_position = _vector3(_get_path(candidate_payload, "source.position_mm"), fallback=(0.0, 0.0, 0.0))
    target_half_z = requirements.target_thickness_mm / 2.0
    if source_position[2] >= -target_half_z:
        mismatches.append(
            {
                "field": "source.upstream_position",
                "expected": f"z < {-target_half_z:g}",
                "actual": source_position[2],
            }
        )
    if requirements.detector_required:
        detector_position = _vector3(_get_path(candidate_payload, "detector.position_mm"), fallback=(0.0, 0.0, 0.0))
        if detector_position[2] <= target_half_z:
            mismatches.append(
                {
                    "field": "detector.downstream_position",
                    "expected": f"z > {target_half_z:g}",
                    "actual": detector_position[2],
                }
            )


__all__ = [
    "INDUSTRIAL_RUNTIME_CONTRACT_PATHS",
    "V3_INDUSTRIAL_CANDIDATE_CONTRACT_SCHEMA_VERSION",
    "V3IndustrialCandidateRequirements",
    "compare_candidate_runtime_contract",
    "compare_v3_candidate_runtime_contract",
]

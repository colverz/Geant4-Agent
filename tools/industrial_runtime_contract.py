from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
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


V3_INDUSTRIAL_CANDIDATE_CONTRACT_SCHEMA_VERSION = "geant4_agent_v3_industrial_candidate_contract.v2"


class DetectorPolicy(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True, slots=True)
class V3IndustrialCandidateRequirements:
    primary_material: str | None
    required_materials: tuple[str, ...]
    target_thickness_mm: float | None
    source_type: str
    particle: str
    energy_mev: float
    direction_vec: tuple[float, float, float]
    detector_policy: DetectorPolicy
    detector_material: str | None
    target_edep_required: bool
    detector_crossings_required: bool
    plane_crossings_required: bool
    depth_bins_required: bool
    region_contrast_required: bool
    physics_list: str
    events: int
    seed: int
    schema_version: str = V3_INDUSTRIAL_CANDIDATE_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_runtime_payload(cls, payload: dict[str, Any]) -> "V3IndustrialCandidateRequirements":
        direction = _vector3(_get_path(payload, "source.direction_vec"), fallback=(0.0, 0.0, 1.0))
        detector_required = bool(_get_path(payload, "detector.enabled"))
        target_material = str(_get_path(payload, "geometry.material") or "")
        required_materials = tuple(item for item in (target_material,) if item)
        return cls(
            primary_material=target_material or None,
            required_materials=required_materials,
            target_thickness_mm=float(_get_path(payload, "geometry.size_z_mm") or 0.0),
            source_type=str(_get_path(payload, "source.type") or ""),
            particle=str(_get_path(payload, "source.particle") or ""),
            energy_mev=float(_get_path(payload, "source.energy_mev") or 0.0),
            direction_vec=direction,
            detector_policy=DetectorPolicy.REQUIRED if detector_required else DetectorPolicy.FORBIDDEN,
            detector_material=str(_get_path(payload, "detector.material") or "") if detector_required else None,
            target_edep_required=bool(_get_path(payload, "scoring.target_edep")),
            detector_crossings_required=bool(_get_path(payload, "scoring.detector_crossings")),
            plane_crossings_required=bool(_get_path(payload, "scoring.plane_crossings")),
            depth_bins_required=bool(_get_path(payload, "scoring.depth_bins")),
            region_contrast_required=bool(_get_path(payload, "scoring.region_contrast")),
            physics_list=str(_get_path(payload, "physics.list") or ""),
            events=int(_get_path(payload, "run.events") or 0),
            seed=int(_get_path(payload, "run.seed") or 0),
        )

    @classmethod
    def from_case(
        cls,
        case: dict[str, Any],
        expected_payload: dict[str, Any],
    ) -> "V3IndustrialCandidateRequirements":
        base = cls.from_runtime_payload(expected_payload)
        raw = case.get("semantic_requirements") if isinstance(case.get("semantic_requirements"), dict) else {}
        target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        detector = raw.get("detector") if isinstance(raw.get("detector"), dict) else {}
        required_scoring = {str(item) for item in raw.get("required_scoring") or [] if str(item)}
        thickness = target.get("thickness_mm") if "thickness_mm" in target else base.target_thickness_mm
        detector_policy = _detector_policy(detector, fallback=base.detector_policy)
        raw_materials = raw.get("required_materials")
        required_materials = (
            tuple(dict.fromkeys(str(item) for item in raw_materials if str(item).strip()))
            if isinstance(raw_materials, list)
            else base.required_materials
        )
        primary_material = target.get("material") if "material" in target else base.primary_material
        return cls(
            primary_material=str(primary_material) if primary_material else None,
            required_materials=required_materials,
            target_thickness_mm=float(thickness) if thickness is not None else None,
            source_type=str(source.get("type") or base.source_type),
            particle=str(source.get("particle") or base.particle),
            energy_mev=float(source.get("energy_mev") if source.get("energy_mev") is not None else base.energy_mev),
            direction_vec=_vector3(source.get("direction_vec"), fallback=base.direction_vec),
            detector_policy=detector_policy,
            detector_material=(
                str(detector.get("material") or base.detector_material or "")
                if detector_policy is DetectorPolicy.REQUIRED
                else None
            ),
            target_edep_required=("target_edep" in required_scoring) if required_scoring else base.target_edep_required,
            detector_crossings_required=(
                "detector_crossings" in required_scoring
                if required_scoring
                else base.detector_crossings_required
            ),
            plane_crossings_required=(
                "plane_crossings" in required_scoring if required_scoring else base.plane_crossings_required
            ),
            depth_bins_required="depth_bins" in required_scoring,
            region_contrast_required="region_contrast" in required_scoring,
            physics_list=base.physics_list,
            events=base.events,
            seed=base.seed,
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


def _detector_policy(detector: dict[str, Any], *, fallback: DetectorPolicy) -> DetectorPolicy:
    raw_policy = detector.get("policy")
    if raw_policy is not None:
        try:
            return DetectorPolicy(str(raw_policy).strip().lower())
        except ValueError:
            return fallback
    if "required" in detector:
        return DetectorPolicy.REQUIRED if bool(detector["required"]) else DetectorPolicy.FORBIDDEN
    return fallback


def _candidate_materials(payload: dict[str, Any]) -> set[str]:
    materials: set[str] = set()
    for path in ("geometry.material", "geometry.world_material", "geometry.world.material", "detector.material"):
        value = _get_path(payload, path)
        if value:
            materials.add(str(value))
    volumes = _get_path(payload, "geometry.volumes")
    if isinstance(volumes, list):
        materials.update(
            str(volume["material"])
            for volume in volumes
            if isinstance(volume, dict) and volume.get("material")
        )
    return materials


def _has_executable_depth_bins(payload: dict[str, Any]) -> bool:
    raw_names = _get_path(payload, "scoring.volume_roles.depth_bin")
    volumes = _get_path(payload, "geometry.volumes")
    if not isinstance(raw_names, list) or not raw_names or not isinstance(volumes, list):
        return False
    volume_names = {
        str(volume.get("name"))
        for volume in volumes
        if isinstance(volume, dict) and volume.get("name")
    }
    return all(str(name) in volume_names for name in raw_names)


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
    *,
    requirements: V3IndustrialCandidateRequirements | None = None,
) -> dict[str, Any]:
    requirements = requirements or V3IndustrialCandidateRequirements.from_runtime_payload(expected_payload)
    mismatches: list[dict[str, Any]] = []

    def require_equal(field: str, actual: Any, expected: Any) -> None:
        if not _value_equal(actual, expected):
            mismatches.append({"field": field, "expected": expected, "actual": actual})

    if requirements.primary_material is not None:
        require_equal("target.material", _get_path(candidate_payload, "geometry.material"), requirements.primary_material)
    candidate_materials = _candidate_materials(candidate_payload)
    missing_materials = sorted(set(requirements.required_materials) - candidate_materials)
    if missing_materials:
        mismatches.append(
            {
                "field": "materials.required",
                "expected": list(requirements.required_materials),
                "actual": sorted(candidate_materials),
            }
        )
    if requirements.target_thickness_mm is not None:
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
    if requirements.detector_policy is DetectorPolicy.REQUIRED and not detector_enabled:
        mismatches.append({"field": "detector.policy", "expected": "required", "actual": "absent"})
    elif requirements.detector_policy is DetectorPolicy.FORBIDDEN and detector_enabled:
        mismatches.append({"field": "detector.policy", "expected": "forbidden", "actual": "present"})
    if requirements.detector_policy is DetectorPolicy.REQUIRED:
        require_equal("detector.material", _get_path(candidate_payload, "detector.material"), requirements.detector_material)

    required_scoring = {
        "scoring.target_edep": requirements.target_edep_required,
        "scoring.detector_crossings": requirements.detector_crossings_required,
        "scoring.plane_crossings": requirements.plane_crossings_required,
        "scoring.depth_bins": requirements.depth_bins_required,
        "scoring.region_contrast": requirements.region_contrast_required,
    }
    for field, required in required_scoring.items():
        if required and not bool(_get_path(candidate_payload, field)):
            mismatches.append({"field": field, "expected": True, "actual": _get_path(candidate_payload, field)})
    if requirements.depth_bins_required and not _has_executable_depth_bins(candidate_payload):
        mismatches.append(
            {
                "field": "scoring.depth_bins.executable",
                "expected": "named depth-bin volumes",
                "actual": _get_path(candidate_payload, "scoring.volume_roles.depth_bin"),
            }
        )

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
    candidate_thickness = _get_path(candidate_payload, "geometry.size_z_mm")
    try:
        target_half_z = float(candidate_thickness) / 2.0
    except (TypeError, ValueError):
        mismatches.append({"field": "target.thickness_mm", "expected": "positive number", "actual": candidate_thickness})
        return
    if source_position[2] >= -target_half_z:
        mismatches.append(
            {
                "field": "source.upstream_position",
                "expected": f"z < {-target_half_z:g}",
                "actual": source_position[2],
            }
        )
    detector_enabled = bool(_get_path(candidate_payload, "detector.enabled"))
    if detector_enabled:
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
    "DetectorPolicy",
    "V3IndustrialCandidateRequirements",
    "compare_candidate_runtime_contract",
    "compare_v3_candidate_runtime_contract",
]

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from core.simulation import build_simulation_spec
from mcp.geant4.runtime_payload import build_runtime_payload

INDUSTRIAL_RUNTIME_COMPILER_SCHEMA_VERSION = "geant4_agent_industrial_runtime_compiler.v1"

DIRECT_METRIC_PATHS = {
    "events_completed": "result_summary.run.events_completed",
    "target_edep_total_mev": "result_summary.scoring.target.target_edep_total_mev",
    "detector_crossing_count": "result_summary.scoring.detector_crossing.detector_crossing_count",
    "detector_edep_total_mev": "result_summary.scoring.roles.detector.edep_total_mev",
    "plane_crossing_count": "result_summary.scoring.plane_crossing.plane_crossing_count",
}

DERIVED_METRIC_INPUTS = {
    "transmission_factor": ["events_completed", "detector_crossing_count"],
    "acceptance_fraction": ["events_completed", "detector_crossing_count"],
}

UNSUPPORTED_FEATURE_HINTS = {
    "multi_thickness_geometry": "step_wedge_geometry_not_supported_by_current_single_volume_runtime",
    "void_geometry": "embedded_void_geometry_not_supported_by_current_single_volume_runtime",
    "region_scoring": "region_scoring_not_supported_by_current_runtime_summary",
    "curved_geometry": "curved_pipe_geometry_not_supported_by_current_single_volume_runtime",
    "paired_scenario": "paired_runtime_runs_not_supported_by_current_benchmark_executor",
    "defect_region": "defect_region_geometry_not_supported_by_current_single_volume_runtime",
    "multi_material_geometry": "inclusion_geometry_not_supported_by_current_single_volume_runtime",
    "multi_layer_geometry": "multi_layer_geometry_not_supported_by_current_runtime_payload",
    "depth_binned_scoring": "depth_binned_scoring_not_supported_by_current_runtime_summary",
    "bragg_peak_proxy": "bragg_peak_metric_not_supported_without_depth_binned_scoring",
    "plane_spatial_scoring": "plane_spatial_distribution_metrics_not_supported_by_current_runtime_summary",
    "collimator_geometry": "collimator_aperture_geometry_not_supported_by_current_runtime_payload",
    "aperture_acceptance": "aperture_acceptance_requires_collimator_geometry",
    "isotropic_source": "isotropic_source_sampling_not_supported_by_current_source_runtime",
    "solid_angle_acceptance": "solid_angle_acceptance_requires_isotropic_source_support",
    "multi_turn_update": "multi_turn_benchmark_execution_not_supported_by_current_runtime_compiler",
    "paired_runtime_runs": "paired_runtime_runs_not_supported_by_current_benchmark_executor",
    "metric_delta": "paired_metric_delta_not_supported_without_multi_run_execution",
    "source_energy_preservation": "multi_turn_source_energy_preservation_not_supported_by_current_benchmark_executor",
    "paired_metric_delta": "paired_metric_delta_not_supported_without_multi_run_execution",
}


def _vec(x: float, y: float, z: float) -> dict[str, Any]:
    return {"type": "vector", "value": [float(x), float(y), float(z)]}


def _base_config(
    *,
    root_name: str,
    structure: str = "single_box",
    material: str,
    size_mm: tuple[float, float, float] = (100.0, 100.0, 10.0),
    particle: str = "gamma",
    energy_mev: float = 1.0,
    source_type: str = "beam",
    source_position_mm: tuple[float, float, float] = (0.0, 0.0, -100.0),
    source_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    detector: dict[str, Any] | None = None,
    scoring_plane_z_mm: float | None = None,
    target_edep: bool = True,
    detector_crossings: bool = True,
    seed: int = 1337,
) -> dict[str, Any]:
    volume_material_map = {root_name: material}
    if detector is not None:
        volume_material_map[str(detector.get("name") or "Detector")] = str(detector.get("material") or "G4_Si")
    return {
        "geometry": {
            "structure": structure,
            "root_name": root_name,
            "size_triplet_mm": [float(size_mm[0]), float(size_mm[1]), float(size_mm[2])],
        },
        "materials": {
            "selected_materials": [material],
            "volume_material_map": volume_material_map,
        },
        "simulation": {
            "detector": deepcopy(detector) if detector is not None else {"enabled": False},
            "run": {"seed": int(seed)},
        },
        "source": {
            "type": source_type,
            "particle": particle,
            "energy": float(energy_mev),
            "position": _vec(*source_position_mm),
            "direction": _vec(*source_direction),
        },
        "physics_list": {"name": "FTFP_BERT"},
        "run": {"seed": int(seed)},
        "scoring": {
            "target_edep": bool(target_edep),
            "detector_crossings": bool(detector_crossings and detector is not None),
            "plane_crossings": scoring_plane_z_mm is not None,
            "plane": {"name": "ScoringPlane", "z_mm": float(scoring_plane_z_mm or 0.0)}
            if scoring_plane_z_mm is not None
            else None,
            "volume_names": [root_name, *([str(detector.get("name") or "Detector")] if detector else [])],
            "volume_roles": {
                "target": [root_name],
                **({"detector": [str(detector.get("name") or "Detector")]} if detector else {}),
            },
        },
    }


def _detector(
    *,
    name: str = "Detector",
    material: str = "G4_Si",
    z_mm: float = 50.0,
    size_mm: tuple[float, float, float] = (20.0, 20.0, 2.0),
) -> dict[str, Any]:
    return {
        "enabled": True,
        "name": name,
        "material": material,
        "position": _vec(0.0, 0.0, z_mm),
        "size_triplet_mm": [float(size_mm[0]), float(size_mm[1]), float(size_mm[2])],
    }


def _case_config(case_id: str) -> dict[str, Any] | None:
    if case_id == "shielding_lead_gamma_transmission":
        return _base_config(
            root_name="LeadShield",
            material="G4_Pb",
            size_mm=(100.0, 100.0, 10.0),
            particle="gamma",
            energy_mev=1.0,
            detector=_detector(z_mm=50.0),
        )
    if case_id == "shielding_concrete_gamma_transmission":
        return _base_config(
            root_name="ConcreteShield",
            material="G4_CONCRETE",
            size_mm=(120.0, 120.0, 100.0),
            particle="gamma",
            energy_mev=2.0,
            detector=_detector(z_mm=130.0),
        )
    if case_id == "shielding_polyethylene_neutron_moderation":
        return _base_config(
            root_name="PolyethyleneShield",
            material="G4_POLYETHYLENE",
            size_mm=(100.0, 100.0, 50.0),
            particle="neutron",
            energy_mev=5.0,
            detector=None,
            scoring_plane_z_mm=80.0,
        )
    if case_id == "medical_proton_water_depth_dose":
        return _base_config(
            root_name="WaterPhantom",
            material="G4_WATER",
            size_mm=(100.0, 100.0, 300.0),
            particle="proton",
            energy_mev=150.0,
            source_position_mm=(0.0, 0.0, -180.0),
            detector=None,
            scoring_plane_z_mm=20.0,
        )
    if case_id == "medical_electron_water_surface_dose":
        return _base_config(
            root_name="WaterPhantom",
            material="G4_WATER",
            size_mm=(100.0, 100.0, 60.0),
            particle="e-",
            energy_mev=12.0,
            source_position_mm=(0.0, 0.0, -50.0),
            detector=None,
        )
    if case_id == "medical_gamma_water_depth_bins":
        return _base_config(
            root_name="WaterPhantom",
            material="G4_WATER",
            size_mm=(100.0, 100.0, 200.0),
            particle="gamma",
            energy_mev=6.0,
            source_position_mm=(0.0, 0.0, -150.0),
            detector=None,
        )
    if case_id == "detector_silicon_gamma_response":
        return _base_config(
            root_name="AirGap",
            material="G4_AIR",
            size_mm=(100.0, 100.0, 80.0),
            particle="gamma",
            energy_mev=1.0,
            detector=_detector(material="G4_Si", z_mm=60.0, size_mm=(20.0, 20.0, 1.0)),
            target_edep=False,
        )
    if case_id == "detector_scintillator_gamma_response":
        return _base_config(
            root_name="AirGap",
            material="G4_AIR",
            size_mm=(100.0, 100.0, 80.0),
            particle="gamma",
            energy_mev=0.662,
            detector=_detector(
                material="G4_PLASTIC_SC_VINYLTOLUENE",
                z_mm=60.0,
                size_mm=(30.0, 30.0, 5.0),
            ),
            target_edep=False,
        )
    if case_id == "beam_gaussian_spread_plane":
        config = _base_config(
            root_name="AirWorld",
            material="G4_AIR",
            size_mm=(200.0, 200.0, 220.0),
            particle="gamma",
            energy_mev=1.0,
            detector=None,
            scoring_plane_z_mm=100.0,
            target_edep=False,
        )
        config["source"].update(
            {
                "spot_profile": "gaussian",
                "spot_sigma_mm": 3.0,
                "divergence_profile": "gaussian",
                "divergence_sigma_deg": 0.0,
            }
        )
        return config
    return None


def _unsupported_features(case: dict[str, Any]) -> list[str]:
    pressures = case.get("capability_pressure")
    if not isinstance(pressures, list):
        return []
    reasons: list[str] = []
    for item in pressures:
        key = str(item)
        reason = UNSUPPORTED_FEATURE_HINTS.get(key)
        if reason:
            reasons.append(reason)
    return list(dict.fromkeys(reasons))


def _metric_plan(case: dict[str, Any]) -> dict[str, Any]:
    golden_metrics = case.get("golden_metrics")
    required = list(golden_metrics.keys()) if isinstance(golden_metrics, dict) else []
    supported: dict[str, Any] = {}
    unsupported: dict[str, str] = {}
    for metric in required:
        if metric in DIRECT_METRIC_PATHS:
            supported[metric] = {"kind": "direct", "path": DIRECT_METRIC_PATHS[metric]}
            continue
        if metric in DERIVED_METRIC_INPUTS:
            inputs = DERIVED_METRIC_INPUTS[metric]
            supported[metric] = {"kind": "derived", "inputs": inputs}
            continue
        unsupported[metric] = "metric_not_available_from_current_structured_runtime_result"
    return {
        "required": required,
        "supported": supported,
        "unsupported": unsupported,
        "all_supported": not unsupported,
    }


def compile_industrial_case_to_runtime(
    case: dict[str, Any],
    *,
    runtime_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_defaults = runtime_defaults if isinstance(runtime_defaults, dict) else {}
    case_id = str(case.get("id") or "")
    if case.get("domain") == "unsupported_boundary" or case.get("golden_required") is False:
        return {
            "schema_version": INDUSTRIAL_RUNTIME_COMPILER_SCHEMA_VERSION,
            "case_id": case_id,
            "status": "unsupported_capability",
            "failure_category": "unsupported_capability",
            "unsupported_features": list(case.get("capability_pressure") or []),
            "metric_plan": _metric_plan(case),
        }

    config = _case_config(case_id)
    unsupported = _unsupported_features(case)
    if config is None:
        return {
            "schema_version": INDUSTRIAL_RUNTIME_COMPILER_SCHEMA_VERSION,
            "case_id": case_id,
            "status": "unsupported_capability",
            "failure_category": "spec_compile_error",
            "unsupported_features": unsupported or ["runtime_blueprint_not_available_for_case"],
            "metric_plan": _metric_plan(case),
        }

    events = int(runtime_defaults.get("events", 10000) or 10000)
    seed = int(runtime_defaults.get("seed", 1337) or 1337)
    physics_list = str(runtime_defaults.get("physics_list") or "FTFP_BERT")
    config["run"]["seed"] = seed
    config["simulation"]["run"]["seed"] = seed
    config["physics_list"] = {"name": physics_list}
    config["scoring"]["volume_names"] = list(dict.fromkeys(config["scoring"].get("volume_names") or []))
    metric_plan = _metric_plan(case)
    spec = build_simulation_spec(config, events=events, mode="batch")
    runtime_payload = build_runtime_payload(spec)
    status = "compiled" if metric_plan["all_supported"] and not unsupported else "compiled_with_gaps"
    return {
        "schema_version": INDUSTRIAL_RUNTIME_COMPILER_SCHEMA_VERSION,
        "case_id": case_id,
        "status": status,
        "failure_category": None if status == "compiled" else "missing_metric",
        "unsupported_features": unsupported,
        "metric_plan": metric_plan,
        "config": config,
        "simulation_spec": asdict(spec),
        "runtime_payload": runtime_payload,
    }


def summarize_compile_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    unsupported_features: dict[str, int] = {}
    unsupported_metrics: dict[str, int] = {}
    for result in results:
        for feature in result.get("unsupported_features") or []:
            name = str(feature)
            unsupported_features[name] = unsupported_features.get(name, 0) + 1
        metric_plan = result.get("metric_plan")
        unsupported = metric_plan.get("unsupported") if isinstance(metric_plan, dict) else {}
        if isinstance(unsupported, dict):
            for metric in unsupported:
                unsupported_metrics[str(metric)] = unsupported_metrics.get(str(metric), 0) + 1
    return {
        "schema_version": INDUSTRIAL_RUNTIME_COMPILER_SCHEMA_VERSION,
        "status_counts": dict(sorted(counts.items())),
        "unsupported_features": dict(sorted(unsupported_features.items())),
        "unsupported_metrics": dict(sorted(unsupported_metrics.items())),
    }

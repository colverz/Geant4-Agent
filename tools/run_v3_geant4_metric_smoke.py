from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from typing import Any

from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from core.simulation import build_runtime_smoke_report
from mcp.geant4.adapter import InMemoryGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.runtime_discovery import discover_local_geant4_runtime
from mcp.geant4.server import Geant4McpServer


V3_GEANT4_METRIC_SMOKE_SCHEMA_VERSION = "geant4_agent_v3_metric_smoke.v1"


def _base_config(*, material: str = "G4_Cu", energy_mev: float = 1.0, events: int = 5) -> dict[str, Any]:
    return {
        "geometry": {
            "structure": "single_box",
            "root_name": "Target",
            "params": {"module_x": 10.0, "module_y": 10.0, "module_z": 10.0},
        },
        "materials": {"selected_materials": [material], "volume_material_map": {"Target": material}},
        "source": {
            "type": "beam",
            "particle": "gamma",
            "energy": energy_mev,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "physics": {"physics_list": "FTFP_BERT"},
        "output": {"format": "json"},
        "run": {"events": events, "seed": 1337},
        "scoring": {"target_edep": True, "detector_crossings": False, "plane_crossings": False},
    }


def _metric_cases(events: int) -> list[dict[str, Any]]:
    detector_case = _base_config(material="G4_Galactic", energy_mev=1.0, events=events)
    detector_case["materials"]["selected_materials"].append("G4_Si")
    detector_case["materials"]["volume_material_map"]["Detector"] = "G4_Si"
    detector_case["simulation"] = {
        "detector": {
            "enabled": True,
            "name": "Detector",
            "material": "G4_Si",
            "position": {"type": "vector", "value": [0.0, 0.0, 30.0]},
            "size_triplet_mm": [40.0, 40.0, 2.0],
        }
    }
    detector_case["scoring"] = {"target_edep": True, "detector_crossings": True, "plane_crossings": False}

    plane_case = _base_config(material="G4_Galactic", energy_mev=1.0, events=events)
    plane_case["scoring"] = {
        "target_edep": True,
        "detector_crossings": False,
        "plane_crossings": True,
        "plane": {"name": "ExitPlane", "z_mm": 5.0},
    }

    high_energy_case = _base_config(material="G4_Cu", energy_mev=2.0, events=events)
    low_energy_case = _base_config(material="G4_Cu", energy_mev=0.5, events=events)

    return [
        {
            "id": "target_edep_copper_1mev",
            "description": "Copper target energy-deposition scoring.",
            "config": _base_config(material="G4_Cu", energy_mev=1.0, events=events),
            "required_metrics": ["target_edep_total_mev"],
        },
        {
            "id": "target_edep_copper_2mev",
            "description": "Same target with higher source energy to verify source-energy propagation.",
            "config": high_energy_case,
            "required_metrics": ["target_edep_total_mev"],
        },
        {
            "id": "target_edep_copper_0p5mev",
            "description": "Same target with lower source energy to verify source-energy propagation.",
            "config": low_energy_case,
            "required_metrics": ["target_edep_total_mev"],
        },
        {
            "id": "detector_crossing_vacuum_gamma",
            "description": "Downstream silicon detector crossing count.",
            "config": detector_case,
            "required_metrics": ["detector_crossing_count"],
        },
        {
            "id": "plane_crossing_vacuum_gamma",
            "description": "Scoring-plane crossing count.",
            "config": plane_case,
            "required_metrics": ["plane_crossing_count"],
        },
    ]


def run_metric_smoke_case(server: Geant4McpServer, case: dict[str, Any], *, events: int) -> dict[str, Any]:
    config = deepcopy(case["config"])
    validate_obs = server.call_tool(ToolCallRequest(tool_name="validate_config", arguments={"config": config, "events": events}))
    if validate_obs.status != RuntimeActionStatus.COMPLETED or not bool((validate_obs.payload or {}).get("ok")):
        return _case_failure(case, "validate_config_failed", validate_obs)
    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": config}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        return _case_failure(case, "apply_config_patch_failed", apply_obs)
    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        return _case_failure(case, "initialize_run_failed", init_obs)
    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": events}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        return _case_failure(case, "run_beam_failed", run_obs)
    summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
    if summary_obs.status != RuntimeActionStatus.COMPLETED:
        return _case_failure(case, "summarize_last_result_failed", summary_obs)
    report = build_runtime_smoke_report(
        events=events,
        run_payload=run_obs.payload,
        summary_payload=summary_obs.payload,
    )
    result_summary = report.get("result_summary") if isinstance(report.get("result_summary"), dict) else {}
    missing = _missing_metrics(report, case["required_metrics"])
    return {
        "id": case["id"],
        "description": case["description"],
        "ok": not missing and bool(report.get("ok")),
        "missing_metrics": missing,
        "events_completed": report.get("events_completed"),
        "completion_fraction": report.get("completion_fraction"),
        "configuration": report.get("configuration"),
        "source": _source_summary(result_summary, config),
        "scoring": _scoring_summary(result_summary),
        "key_metrics": report.get("key_metrics"),
        "artifact_dir": report.get("artifact_dir"),
        "run_summary_path": report.get("run_summary_path"),
        "result_summary_sections": list(result_summary.keys()),
    }


def run_v3_geant4_metric_smoke(*, events: int = 5, auto_discover_runtime: bool = True, require_runtime: bool = True) -> dict[str, Any]:
    discovery = discover_local_geant4_runtime() if auto_discover_runtime else None
    env = discovery.env() if discovery is not None and discovery.found else None
    adapter = build_geant4_adapter_from_env(env)
    if isinstance(adapter, InMemoryGeant4Adapter):
        return {
            "schema_version": V3_GEANT4_METRIC_SMOKE_SCHEMA_VERSION,
            "ok": not require_runtime,
            "skipped": True,
            "skip_reason": "local_process_runtime_required",
            "runtime_discovery": discovery.to_dict() if discovery is not None else None,
            "case_results": [],
        }
    server = Geant4McpServer(adapter=adapter)
    case_results = [run_metric_smoke_case(server, case, events=events) for case in _metric_cases(events)]
    return {
        "schema_version": V3_GEANT4_METRIC_SMOKE_SCHEMA_VERSION,
        "ok": all(bool(item.get("ok")) for item in case_results),
        "skipped": False,
        "events": events,
        "runtime_discovery": discovery.to_dict() if discovery is not None else None,
        "case_results": case_results,
    }


def _missing_metrics(report: dict[str, Any], required_metrics: list[str]) -> list[str]:
    metrics = report.get("key_metrics") if isinstance(report.get("key_metrics"), dict) else {}
    missing = []
    for key in required_metrics:
        if metrics.get(key) is None:
            missing.append(key)
    return missing


def _source_summary(result_summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source = result_summary.get("source") if isinstance(result_summary.get("source"), dict) else {}
    config_source = config.get("source") if isinstance(config.get("source"), dict) else {}
    return {
        "type": source.get("type") or config_source.get("type"),
        "particle": source.get("particle") or config_source.get("particle"),
        "energy_mev": source.get("energy_mev") or config_source.get("energy"),
        "primary_count": source.get("primary_count"),
    }


def _scoring_summary(result_summary: dict[str, Any]) -> dict[str, Any]:
    scoring = result_summary.get("scoring") if isinstance(result_summary.get("scoring"), dict) else {}
    target = scoring.get("target") if isinstance(scoring.get("target"), dict) else {}
    detector = scoring.get("detector_crossing") if isinstance(scoring.get("detector_crossing"), dict) else {}
    plane = scoring.get("plane_crossing") if isinstance(scoring.get("plane_crossing"), dict) else {}
    return {
        "target_edep_enabled": target.get("target_edep_enabled"),
        "target_edep_mean_mev_per_event": target.get("target_edep_mean_mev_per_event"),
        "detector_crossings_enabled": detector.get("detector_crossings_enabled"),
        "detector_crossing_mean_per_event": detector.get("detector_crossing_mean_per_event"),
        "plane_crossings_enabled": plane.get("plane_crossings_enabled"),
        "plane_crossing_mean_per_event": plane.get("plane_crossing_mean_per_event"),
    }


def _case_failure(case: dict[str, Any], reason: str, obs: Any) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "description": case.get("description"),
        "ok": False,
        "failure_reason": reason,
        "message": str(getattr(obs, "message", "") or ""),
        "payload": getattr(obs, "payload", {}) or {},
        "errors": list(getattr(obs, "errors", []) or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Geant4 metric smoke cases for v3 agent runtime confidence.")
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--no-auto-discover-runtime", action="store_true")
    parser.add_argument("--allow-skip", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_v3_geant4_metric_smoke(
        events=max(1, int(args.events)),
        auto_discover_runtime=not args.no_auto_discover_runtime,
        require_runtime=not args.allow_skip,
    )
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        if result.get("skipped"):
            print(f"SKIP: {result.get('skip_reason')}")
        else:
            for item in result.get("case_results", []):
                print(f"{item['id']}: ok={item.get('ok')} events={item.get('events_completed')} metrics={item.get('key_metrics')}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

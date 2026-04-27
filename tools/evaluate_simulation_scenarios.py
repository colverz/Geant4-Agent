from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.simulation import SIMULATION_RESULT_SCHEMA_VERSION, build_runtime_smoke_report, build_simulation_spec, simulation_result_from_dict
from mcp.geant4.runtime_payload import build_runtime_payload


DEFAULT_SCENARIO_CASEBANK = Path("docs/eval/simulation_scenario_casebank.json")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_run_summary(runtime_payload: dict[str, Any], events: int) -> dict[str, Any]:
    scoring = runtime_payload.get("scoring", {}) if isinstance(runtime_payload.get("scoring"), dict) else {}
    detector_enabled = bool(runtime_payload.get("detector_enabled"))
    plane_enabled = bool(scoring.get("plane_crossings"))
    detector_crossings_enabled = bool(scoring.get("detector_crossings"))
    return {
        "run_ok": True,
        "events_requested": events,
        "events_completed": events,
        "schema_version": SIMULATION_RESULT_SCHEMA_VERSION,
        "geometry_structure": runtime_payload["structure"],
        "material": runtime_payload["material"],
        "particle": runtime_payload["particle"],
        "source_type": runtime_payload["source_type"],
        "source_position_mm": [
            runtime_payload["position"]["x"],
            runtime_payload["position"]["y"],
            runtime_payload["position"]["z"],
        ],
        "source_direction": [
            runtime_payload["direction"]["x"],
            runtime_payload["direction"]["y"],
            runtime_payload["direction"]["z"],
        ],
        "source_primary_count": events,
        "source_sampled_position_mean_mm": [
            runtime_payload["position"]["x"],
            runtime_payload["position"]["y"],
            runtime_payload["position"]["z"],
        ],
        "source_sampled_direction_mean": [
            runtime_payload["direction"]["x"],
            runtime_payload["direction"]["y"],
            runtime_payload["direction"]["z"],
        ],
        "physics_list": runtime_payload["physics_list"],
        "events": events,
        "mode": runtime_payload["run"]["mode"],
        "run_seed": runtime_payload["run"]["seed"],
        "run_manifest": runtime_payload["run_manifest"],
        "detector": (
            {
                "enabled": True,
                "volume_name": runtime_payload["detector_name"],
                "material": runtime_payload["detector_material"],
                "position_mm": [
                    runtime_payload["detector_position"]["x"],
                    runtime_payload["detector_position"]["y"],
                    runtime_payload["detector_position"]["z"],
                ],
                "size_mm": [
                    runtime_payload["detector_size_x"],
                    runtime_payload["detector_size_y"],
                    runtime_payload["detector_size_z"],
                ],
            }
            if detector_enabled
            else {"enabled": False}
        ),
        "scoring": {
            "target_edep_enabled": bool(scoring.get("target_edep")),
            "detector_crossings_enabled": detector_crossings_enabled,
            "plane_crossings_enabled": plane_enabled,
            "plane_crossing_name": (scoring.get("plane") or {}).get("name") if isinstance(scoring.get("plane"), dict) else None,
            "plane_crossing_z_mm": (scoring.get("plane") or {}).get("z_mm") if isinstance(scoring.get("plane"), dict) else None,
            "plane_crossing_count": events if plane_enabled else 0,
            "plane_crossing_events": events if plane_enabled else 0,
            "detector_crossing_count": events if detector_crossings_enabled and detector_enabled else 0,
            "detector_crossing_events": events if detector_crossings_enabled and detector_enabled else 0,
            "target_edep_total_mev": 0.0,
            "target_hit_events": 0,
            "target_step_count": 0,
            "target_track_entries": events,
        },
    }


def _validate_scenario(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = case.get("expected", {}) if isinstance(case.get("expected"), dict) else {}
    events = int(case.get("events", 1))
    spec = build_simulation_spec(case.get("config", {}), events=events)
    runtime_payload = build_runtime_payload(spec)

    checks = {
        "structure": runtime_payload.get("structure"),
        "material": runtime_payload.get("material"),
        "source_type": runtime_payload.get("source_type"),
        "particle": runtime_payload.get("particle"),
        "energy": runtime_payload.get("energy"),
        "physics_list": runtime_payload.get("physics_list"),
        "detector_enabled": runtime_payload.get("detector_enabled"),
    }
    for key, actual in checks.items():
        if key in expected and actual != expected[key]:
            errors.append(f"{key}:expected={expected[key]!r}:actual={actual!r}")

    scoring = runtime_payload.get("scoring", {}) if isinstance(runtime_payload.get("scoring"), dict) else {}
    expected_scoring = set(expected.get("scoring", []))
    for scorer in expected_scoring:
        if not bool(scoring.get(scorer)):
            errors.append(f"missing_enabled_scorer:{scorer}")

    run_summary = _fake_run_summary(runtime_payload, events)
    result = simulation_result_from_dict(run_summary)
    result_payload = result.to_payload()
    smoke_report = build_runtime_smoke_report(
        events=events,
        run_payload={"simulation_result": result_payload, "result_summary": result_payload["result_summary"]},
    )
    summary = smoke_report.get("result_summary") if isinstance(smoke_report.get("result_summary"), dict) else {}
    for section in expected.get("result_summary_sections", []):
        if section not in summary:
            errors.append(f"missing_result_summary_section:{section}")
    if smoke_report.get("events_completed") != events:
        errors.append(f"events_completed:expected={events}:actual={smoke_report.get('events_completed')}")
    if smoke_report.get("configuration", {}).get("particle") != expected.get("particle"):
        errors.append("smoke_report_particle_mismatch")
    return errors


def evaluate_simulation_scenarios(path: Path = DEFAULT_SCENARIO_CASEBANK) -> dict[str, Any]:
    cases = _load_json(path)
    failures: list[dict[str, Any]] = []
    for case in cases:
        errors = _validate_scenario(case)
        if errors:
            failures.append({"id": case.get("id"), "errors": errors})
    return {"name": "simulation_scenarios", "total": len(cases), "failed": len(failures), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate scenario-level simulation bridge coverage.")
    parser.add_argument("--casebank", type=Path, default=DEFAULT_SCENARIO_CASEBANK)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate_simulation_scenarios(args.casebank)
    output = {"ok": report["failed"] == 0, "failed": report["failed"], "reports": [report]}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"{report['name']}: {report['total'] - report['failed']} / {report['total']} passed")
        for failure in report["failures"]:
            print(f"  FAIL {failure['id']}: {failure}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from copy import deepcopy
from typing import Any


RUNTIME_SMOKE_REPORT_SCHEMA_VERSION = "2026-04-24.runtime-smoke.v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def build_runtime_smoke_report(
    *,
    events: int,
    run_payload: dict[str, Any] | None = None,
    summary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the compact report used by manual/live runtime smoke tests.

    This is deliberately a consumption contract, not a replacement for the full
    SimulationResult payload. Agent/UI callers can show this directly while deeper
    analysis still reads result_summary or the raw run artifact.
    """

    run_payload = _as_dict(run_payload)
    summary_payload = _as_dict(summary_payload)
    simulation_result = _as_dict(run_payload.get("simulation_result"))
    result_summary = _first_dict(summary_payload.get("result_summary"), run_payload.get("result_summary"))
    run = _as_dict(result_summary.get("run"))
    configuration = _as_dict(result_summary.get("configuration"))
    scoring = _as_dict(result_summary.get("scoring"))
    target = _as_dict(scoring.get("target"))
    detector_crossing = _as_dict(scoring.get("detector_crossing"))
    plane_crossing = _as_dict(scoring.get("plane_crossing"))

    return {
        "schema_version": RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
        "ok": bool(result_summary) and run.get("ok") is not False,
        "events_requested": int(events),
        "events_completed": run.get("events_completed"),
        "completion_fraction": run.get("completion_fraction"),
        "configuration": {
            "geometry_structure": configuration.get("geometry_structure"),
            "material": configuration.get("material"),
            "source_type": configuration.get("source_type"),
            "particle": configuration.get("particle"),
            "physics_list": configuration.get("physics_list"),
            "detector_enabled": configuration.get("detector_enabled"),
        },
        "key_metrics": {
            "target_edep_total_mev": target.get("target_edep_total_mev"),
            "target_hit_events": target.get("target_hit_events"),
            "target_track_entries": target.get("target_track_entries"),
            "detector_crossing_count": detector_crossing.get("detector_crossing_count"),
            "plane_crossing_count": plane_crossing.get("plane_crossing_count"),
        },
        "artifact_dir": simulation_result.get("artifact_dir") or summary_payload.get("artifact_dir") or "",
        "run_summary_path": simulation_result.get("run_summary_path") or summary_payload.get("run_summary_path") or "",
        "result_summary": deepcopy(result_summary) if result_summary else None,
    }

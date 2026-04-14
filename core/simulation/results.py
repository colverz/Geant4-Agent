from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

SIMULATION_RESULT_SCHEMA_VERSION = "2026-04-14.v1"


def _coerce_triplet(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SimulationScoringResult:
    target_edep_enabled: bool = False
    detector_crossings_enabled: bool = False
    detector_crossing_count: int = 0
    detector_crossing_events: int = 0
    target_edep_total_mev: float = 0.0
    target_edep_mean_mev_per_event: float = 0.0
    target_hit_events: int = 0
    target_step_count: int = 0
    target_track_entries: int = 0
    volume_stats: dict[str, dict[str, float | int]] | None = None
    role_stats: dict[str, dict[str, float | int]] | None = None


@dataclass(frozen=True)
class SimulationDetectorResult:
    enabled: bool = False
    volume_name: str | None = None
    material: str | None = None
    position_mm: tuple[float, float, float] | None = None
    size_mm: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SimulationResult:
    run_ok: bool
    events_requested: int
    events_completed: int
    schema_version: str = SIMULATION_RESULT_SCHEMA_VERSION
    geometry_structure: str | None = None
    material: str | None = None
    particle: str | None = None
    source_type: str | None = None
    source_position_mm: tuple[float, float, float] | None = None
    source_direction: tuple[float, float, float] | None = None
    payload_sha256: str | None = None
    geant4_version: str | None = None
    run_seed: int = 1337
    run_manifest: dict[str, Any] | None = None
    physics_list: str | None = None
    events: int = 0
    mode: str = "batch"
    scoring: SimulationScoringResult = SimulationScoringResult()
    detector: SimulationDetectorResult = SimulationDetectorResult()

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def simulation_result_from_dict(data: dict[str, Any]) -> SimulationResult:
    scoring_data = data.get("scoring", {}) if isinstance(data.get("scoring"), dict) else {}
    raw_volume_stats = scoring_data.get("volume_stats", {})
    raw_role_stats = scoring_data.get("role_stats", {})
    volume_stats: dict[str, dict[str, float | int]] = {}
    role_stats: dict[str, dict[str, float | int]] = {}
    if isinstance(raw_volume_stats, dict):
        for volume_name, raw_stats in raw_volume_stats.items():
            if not isinstance(volume_name, str) or not isinstance(raw_stats, dict):
                continue
            volume_stats[volume_name] = {
                "edep_total_mev": float(raw_stats.get("edep_total_mev", 0.0) or 0.0),
                "edep_mean_mev_per_event": float(raw_stats.get("edep_mean_mev_per_event", 0.0) or 0.0),
                "hit_events": int(raw_stats.get("hit_events", 0) or 0),
                "crossing_events": int(raw_stats.get("crossing_events", 0) or 0),
                "crossing_count": int(raw_stats.get("crossing_count", 0) or 0),
                "step_count": int(raw_stats.get("step_count", 0) or 0),
                "track_entries": int(raw_stats.get("track_entries", 0) or 0),
            }
    if isinstance(raw_role_stats, dict):
        for role_name, raw_stats in raw_role_stats.items():
            if not isinstance(role_name, str) or not isinstance(raw_stats, dict):
                continue
            role_stats[role_name] = {
                "edep_total_mev": float(raw_stats.get("edep_total_mev", 0.0) or 0.0),
                "edep_mean_mev_per_event": float(raw_stats.get("edep_mean_mev_per_event", 0.0) or 0.0),
                "hit_events": int(raw_stats.get("hit_events", 0) or 0),
                "crossing_events": int(raw_stats.get("crossing_events", 0) or 0),
                "crossing_count": int(raw_stats.get("crossing_count", 0) or 0),
                "step_count": int(raw_stats.get("step_count", 0) or 0),
                "track_entries": int(raw_stats.get("track_entries", 0) or 0),
            }
    if not volume_stats:
        volume_stats["Target"] = {
            "edep_total_mev": float(scoring_data.get("target_edep_total_mev", 0.0) or 0.0),
            "edep_mean_mev_per_event": float(scoring_data.get("target_edep_mean_mev_per_event", 0.0) or 0.0),
            "hit_events": int(scoring_data.get("target_hit_events", 0) or 0),
            "crossing_events": 0,
            "crossing_count": 0,
            "step_count": int(scoring_data.get("target_step_count", 0) or 0),
            "track_entries": int(scoring_data.get("target_track_entries", 0) or 0),
        }
    scoring = SimulationScoringResult(
        target_edep_enabled=bool(scoring_data.get("target_edep_enabled", False)),
        detector_crossings_enabled=bool(scoring_data.get("detector_crossings_enabled", False)),
        detector_crossing_count=int(scoring_data.get("detector_crossing_count", 0) or 0),
        detector_crossing_events=int(scoring_data.get("detector_crossing_events", 0) or 0),
        target_edep_total_mev=float(scoring_data.get("target_edep_total_mev", 0.0) or 0.0),
        target_edep_mean_mev_per_event=float(scoring_data.get("target_edep_mean_mev_per_event", 0.0) or 0.0),
        target_hit_events=int(scoring_data.get("target_hit_events", 0) or 0),
        target_step_count=int(scoring_data.get("target_step_count", 0) or 0),
        target_track_entries=int(scoring_data.get("target_track_entries", 0) or 0),
        volume_stats=volume_stats or None,
        role_stats=role_stats or None,
    )
    return SimulationResult(
        schema_version=str(data.get("schema_version") or SIMULATION_RESULT_SCHEMA_VERSION),
        run_ok=bool(data.get("run_ok", False)),
        events_requested=int(data.get("events_requested", 0) or 0),
        events_completed=int(data.get("events_completed", 0) or 0),
        geometry_structure=data.get("geometry_structure"),
        material=data.get("material"),
        particle=data.get("particle"),
        source_type=data.get("source_type"),
        source_position_mm=_coerce_triplet(data.get("source_position_mm")),
        source_direction=_coerce_triplet(data.get("source_direction")),
        payload_sha256=data.get("payload_sha256"),
        geant4_version=data.get("geant4_version"),
        run_seed=int(data.get("run_seed", 1337) or 1337),
        run_manifest=data.get("run_manifest") if isinstance(data.get("run_manifest"), dict) else None,
        physics_list=data.get("physics_list"),
        events=int(data.get("events", 0) or 0),
        mode=str(data.get("mode", "batch") or "batch"),
        scoring=scoring,
        detector=SimulationDetectorResult(
            enabled=bool(data.get("detector", {}).get("enabled", False)) if isinstance(data.get("detector"), dict) else False,
            volume_name=data.get("detector", {}).get("volume_name") if isinstance(data.get("detector"), dict) else None,
            material=data.get("detector", {}).get("material") if isinstance(data.get("detector"), dict) else None,
            position_mm=_coerce_triplet(data.get("detector", {}).get("position_mm")) if isinstance(data.get("detector"), dict) else None,
            size_mm=_coerce_triplet(data.get("detector", {}).get("size_mm")) if isinstance(data.get("detector"), dict) else None,
        ),
    )


def load_simulation_result(summary_path: str | Path) -> SimulationResult:
    summary_file = Path(summary_path)
    with summary_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("run_summary.json must contain a JSON object")
    return simulation_result_from_dict(payload)


def derive_role_stats(
    volume_stats: dict[str, dict[str, float | int]] | None,
    volume_roles: dict[str, list[str] | tuple[str, ...] | str] | None,
) -> dict[str, dict[str, float | int]]:
    if not volume_stats or not volume_roles:
        return {}
    role_stats: dict[str, dict[str, float | int]] = {}
    for role, raw_names in volume_roles.items():
        role_name = str(role).strip()
        if not role_name:
            continue
        if isinstance(raw_names, str):
            names = [raw_names]
        elif isinstance(raw_names, (list, tuple)):
            names = [str(name) for name in raw_names if str(name)]
        else:
            names = []
        if not names:
            continue
        aggregate = {
            "edep_total_mev": 0.0,
            "edep_mean_mev_per_event": 0.0,
            "hit_events": 0,
            "crossing_events": 0,
            "crossing_count": 0,
            "step_count": 0,
            "track_entries": 0,
        }
        matched = False
        for name in names:
            stats = volume_stats.get(name)
            if not isinstance(stats, dict):
                continue
            matched = True
            aggregate["edep_total_mev"] += float(stats.get("edep_total_mev", 0.0) or 0.0)
            aggregate["edep_mean_mev_per_event"] += float(stats.get("edep_mean_mev_per_event", 0.0) or 0.0)
            aggregate["hit_events"] += int(stats.get("hit_events", 0) or 0)
            aggregate["crossing_events"] += int(stats.get("crossing_events", 0) or 0)
            aggregate["crossing_count"] += int(stats.get("crossing_count", 0) or 0)
            aggregate["step_count"] += int(stats.get("step_count", 0) or 0)
            aggregate["track_entries"] += int(stats.get("track_entries", 0) or 0)
        if matched:
            role_stats[role_name] = aggregate
    return role_stats

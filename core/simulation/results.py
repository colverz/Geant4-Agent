from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


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
    target_edep_total_mev: float = 0.0
    target_edep_mean_mev_per_event: float = 0.0
    target_hit_events: int = 0
    target_step_count: int = 0
    target_track_entries: int = 0
    volume_stats: dict[str, dict[str, float | int]] | None = None


@dataclass(frozen=True)
class SimulationResult:
    run_ok: bool
    events_requested: int
    events_completed: int
    geometry_structure: str | None = None
    material: str | None = None
    particle: str | None = None
    source_type: str | None = None
    source_position_mm: tuple[float, float, float] | None = None
    source_direction: tuple[float, float, float] | None = None
    physics_list: str | None = None
    events: int = 0
    mode: str = "batch"
    scoring: SimulationScoringResult = SimulationScoringResult()

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def simulation_result_from_dict(data: dict[str, Any]) -> SimulationResult:
    scoring_data = data.get("scoring", {}) if isinstance(data.get("scoring"), dict) else {}
    raw_volume_stats = scoring_data.get("volume_stats", {})
    volume_stats: dict[str, dict[str, float | int]] = {}
    if isinstance(raw_volume_stats, dict):
        for volume_name, raw_stats in raw_volume_stats.items():
            if not isinstance(volume_name, str) or not isinstance(raw_stats, dict):
                continue
            volume_stats[volume_name] = {
                "edep_total_mev": float(raw_stats.get("edep_total_mev", 0.0) or 0.0),
                "edep_mean_mev_per_event": float(raw_stats.get("edep_mean_mev_per_event", 0.0) or 0.0),
                "hit_events": int(raw_stats.get("hit_events", 0) or 0),
                "step_count": int(raw_stats.get("step_count", 0) or 0),
                "track_entries": int(raw_stats.get("track_entries", 0) or 0),
            }
    if not volume_stats:
        volume_stats["Target"] = {
            "edep_total_mev": float(scoring_data.get("target_edep_total_mev", 0.0) or 0.0),
            "edep_mean_mev_per_event": float(scoring_data.get("target_edep_mean_mev_per_event", 0.0) or 0.0),
            "hit_events": int(scoring_data.get("target_hit_events", 0) or 0),
            "step_count": int(scoring_data.get("target_step_count", 0) or 0),
            "track_entries": int(scoring_data.get("target_track_entries", 0) or 0),
        }
    scoring = SimulationScoringResult(
        target_edep_enabled=bool(scoring_data.get("target_edep_enabled", False)),
        target_edep_total_mev=float(scoring_data.get("target_edep_total_mev", 0.0) or 0.0),
        target_edep_mean_mev_per_event=float(scoring_data.get("target_edep_mean_mev_per_event", 0.0) or 0.0),
        target_hit_events=int(scoring_data.get("target_hit_events", 0) or 0),
        target_step_count=int(scoring_data.get("target_step_count", 0) or 0),
        target_track_entries=int(scoring_data.get("target_track_entries", 0) or 0),
        volume_stats=volume_stats or None,
    )
    return SimulationResult(
        run_ok=bool(data.get("run_ok", False)),
        events_requested=int(data.get("events_requested", 0) or 0),
        events_completed=int(data.get("events_completed", 0) or 0),
        geometry_structure=data.get("geometry_structure"),
        material=data.get("material"),
        particle=data.get("particle"),
        source_type=data.get("source_type"),
        source_position_mm=_coerce_triplet(data.get("source_position_mm")),
        source_direction=_coerce_triplet(data.get("source_direction")),
        physics_list=data.get("physics_list"),
        events=int(data.get("events", 0) or 0),
        mode=str(data.get("mode", "batch") or "batch"),
        scoring=scoring,
    )


def load_simulation_result(summary_path: str | Path) -> SimulationResult:
    summary_file = Path(summary_path)
    with summary_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("run_summary.json must contain a JSON object")
    return simulation_result_from_dict(payload)

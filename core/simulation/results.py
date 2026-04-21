from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

SIMULATION_RESULT_SCHEMA_VERSION = "2026-04-14.v7"


def _coerce_triplet(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _coerce_int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        result[key] = int(raw_value or 0)
    return result


@dataclass(frozen=True)
class SimulationVolumeStatsResult:
    edep_total_mev: float = 0.0
    edep_mean_mev_per_event: float = 0.0
    hit_events: int = 0
    crossing_events: int = 0
    crossing_count: int = 0
    crossing_mean_per_event: float = 0.0
    step_count: int = 0
    track_entries: int = 0

    def to_payload(self) -> dict[str, float | int]:
        return {
            "edep_total_mev": self.edep_total_mev,
            "edep_mean_mev_per_event": self.edep_mean_mev_per_event,
            "hit_events": self.hit_events,
            "crossing_events": self.crossing_events,
            "crossing_count": self.crossing_count,
            "crossing_mean_per_event": self.crossing_mean_per_event,
            "step_count": self.step_count,
            "track_entries": self.track_entries,
        }

    def __getitem__(self, key: str) -> float | int:
        return self.to_payload()[key]

    def __contains__(self, key: object) -> bool:
        return key in self.to_payload()

    def get(self, key: str, default: object = None) -> object:
        return self.to_payload().get(key, default)

    def keys(self) -> Any:
        return self.to_payload().keys()

    def items(self) -> Any:
        return self.to_payload().items()

    def values(self) -> Any:
        return self.to_payload().values()


@dataclass(frozen=True)
class SimulationParticleCrossingResult:
    counts: dict[str, int] | None = None
    events: dict[str, int] | None = None


@dataclass(frozen=True)
class SimulationTargetScoringResult:
    edep_enabled: bool = False
    edep_total_mev: float = 0.0
    edep_mean_mev_per_event: float = 0.0
    hit_events: int = 0
    step_count: int = 0
    track_entries: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_edep_enabled": self.edep_enabled,
            "target_edep_total_mev": self.edep_total_mev,
            "target_edep_mean_mev_per_event": self.edep_mean_mev_per_event,
            "target_hit_events": self.hit_events,
            "target_step_count": self.step_count,
            "target_track_entries": self.track_entries,
        }


@dataclass(frozen=True)
class SimulationDetectorCrossingResult:
    enabled: bool = False
    count: int = 0
    events: int = 0
    mean_per_event: float = 0.0
    particles: SimulationParticleCrossingResult = field(default_factory=SimulationParticleCrossingResult)

    def to_payload(self) -> dict[str, Any]:
        return {
            "detector_crossings_enabled": self.enabled,
            "detector_crossing_count": self.count,
            "detector_crossing_events": self.events,
            "detector_crossing_mean_per_event": self.mean_per_event,
            "detector_crossing_particle_counts": self.particles.counts,
            "detector_crossing_particle_events": self.particles.events,
        }


@dataclass(frozen=True)
class SimulationPlaneCrossingResult:
    enabled: bool = False
    name: str | None = None
    z_mm: float | None = None
    count: int = 0
    events: int = 0
    forward_count: int = 0
    forward_events: int = 0
    reverse_count: int = 0
    reverse_events: int = 0
    mean_per_event: float = 0.0
    particles: SimulationParticleCrossingResult = field(default_factory=SimulationParticleCrossingResult)

    def to_payload(self) -> dict[str, Any]:
        return {
            "plane_crossings_enabled": self.enabled,
            "plane_crossing_name": self.name,
            "plane_crossing_z_mm": self.z_mm,
            "plane_crossing_count": self.count,
            "plane_crossing_events": self.events,
            "plane_crossing_forward_count": self.forward_count,
            "plane_crossing_forward_events": self.forward_events,
            "plane_crossing_reverse_count": self.reverse_count,
            "plane_crossing_reverse_events": self.reverse_events,
            "plane_crossing_mean_per_event": self.mean_per_event,
            "plane_crossing_particle_counts": self.particles.counts,
            "plane_crossing_particle_events": self.particles.events,
        }


def _parse_volume_stats(raw_stats: object) -> SimulationVolumeStatsResult:
    if not isinstance(raw_stats, dict):
        raw_stats = {}
    return SimulationVolumeStatsResult(
        edep_total_mev=float(raw_stats.get("edep_total_mev", 0.0) or 0.0),
        edep_mean_mev_per_event=float(raw_stats.get("edep_mean_mev_per_event", 0.0) or 0.0),
        hit_events=int(raw_stats.get("hit_events", 0) or 0),
        crossing_events=int(raw_stats.get("crossing_events", 0) or 0),
        crossing_count=int(raw_stats.get("crossing_count", 0) or 0),
        crossing_mean_per_event=float(raw_stats.get("crossing_mean_per_event", 0.0) or 0.0),
        step_count=int(raw_stats.get("step_count", 0) or 0),
        track_entries=int(raw_stats.get("track_entries", 0) or 0),
    )


def _parse_volume_stats_map(raw_stats_map: object) -> dict[str, SimulationVolumeStatsResult]:
    if not isinstance(raw_stats_map, dict):
        return {}
    result: dict[str, SimulationVolumeStatsResult] = {}
    for raw_name, raw_stats in raw_stats_map.items():
        name = str(raw_name).strip()
        if not name:
            continue
        result[name] = _parse_volume_stats(raw_stats)
    return result


def _volume_stats_map_to_payload(
    stats_map: dict[str, SimulationVolumeStatsResult] | None,
) -> dict[str, dict[str, float | int]] | None:
    if not stats_map:
        return None
    return {name: stats.to_payload() for name, stats in stats_map.items()}


@dataclass(frozen=True, init=False)
class SimulationScoringResult:
    target: SimulationTargetScoringResult
    detector_crossing: SimulationDetectorCrossingResult
    plane_crossing: SimulationPlaneCrossingResult
    volume_stats: dict[str, SimulationVolumeStatsResult] | None
    role_stats: dict[str, SimulationVolumeStatsResult] | None

    def __init__(
        self,
        target: SimulationTargetScoringResult | None = None,
        detector_crossing: SimulationDetectorCrossingResult | None = None,
        plane_crossing: SimulationPlaneCrossingResult | None = None,
        volume_stats: dict[str, SimulationVolumeStatsResult | dict[str, float | int]] | None = None,
        role_stats: dict[str, SimulationVolumeStatsResult | dict[str, float | int]] | None = None,
        **legacy_fields: Any,
    ) -> None:
        if target is None:
            target = SimulationTargetScoringResult(
                edep_enabled=bool(legacy_fields.pop("target_edep_enabled", False)),
                edep_total_mev=float(legacy_fields.pop("target_edep_total_mev", 0.0) or 0.0),
                edep_mean_mev_per_event=float(legacy_fields.pop("target_edep_mean_mev_per_event", 0.0) or 0.0),
                hit_events=int(legacy_fields.pop("target_hit_events", 0) or 0),
                step_count=int(legacy_fields.pop("target_step_count", 0) or 0),
                track_entries=int(legacy_fields.pop("target_track_entries", 0) or 0),
            )
        else:
            for key in (
                "target_edep_enabled",
                "target_edep_total_mev",
                "target_edep_mean_mev_per_event",
                "target_hit_events",
                "target_step_count",
                "target_track_entries",
            ):
                legacy_fields.pop(key, None)

        if detector_crossing is None:
            detector_crossing = SimulationDetectorCrossingResult(
                enabled=bool(legacy_fields.pop("detector_crossings_enabled", False)),
                count=int(legacy_fields.pop("detector_crossing_count", 0) or 0),
                events=int(legacy_fields.pop("detector_crossing_events", 0) or 0),
                mean_per_event=float(legacy_fields.pop("detector_crossing_mean_per_event", 0.0) or 0.0),
                particles=SimulationParticleCrossingResult(
                    counts=_coerce_int_map(legacy_fields.pop("detector_crossing_particle_counts", None)) or None,
                    events=_coerce_int_map(legacy_fields.pop("detector_crossing_particle_events", None)) or None,
                ),
            )
        else:
            for key in (
                "detector_crossings_enabled",
                "detector_crossing_count",
                "detector_crossing_events",
                "detector_crossing_mean_per_event",
                "detector_crossing_particle_counts",
                "detector_crossing_particle_events",
            ):
                legacy_fields.pop(key, None)

        if plane_crossing is None:
            plane_z = legacy_fields.pop("plane_crossing_z_mm", None)
            plane_crossing = SimulationPlaneCrossingResult(
                enabled=bool(legacy_fields.pop("plane_crossings_enabled", False)),
                name=legacy_fields.pop("plane_crossing_name", None),
                z_mm=float(plane_z) if plane_z is not None else None,
                count=int(legacy_fields.pop("plane_crossing_count", 0) or 0),
                events=int(legacy_fields.pop("plane_crossing_events", 0) or 0),
                forward_count=int(legacy_fields.pop("plane_crossing_forward_count", 0) or 0),
                forward_events=int(legacy_fields.pop("plane_crossing_forward_events", 0) or 0),
                reverse_count=int(legacy_fields.pop("plane_crossing_reverse_count", 0) or 0),
                reverse_events=int(legacy_fields.pop("plane_crossing_reverse_events", 0) or 0),
                mean_per_event=float(legacy_fields.pop("plane_crossing_mean_per_event", 0.0) or 0.0),
                particles=SimulationParticleCrossingResult(
                    counts=_coerce_int_map(legacy_fields.pop("plane_crossing_particle_counts", None)) or None,
                    events=_coerce_int_map(legacy_fields.pop("plane_crossing_particle_events", None)) or None,
                ),
            )
        else:
            for key in (
                "plane_crossings_enabled",
                "plane_crossing_name",
                "plane_crossing_z_mm",
                "plane_crossing_count",
                "plane_crossing_events",
                "plane_crossing_forward_count",
                "plane_crossing_forward_events",
                "plane_crossing_reverse_count",
                "plane_crossing_reverse_events",
                "plane_crossing_mean_per_event",
                "plane_crossing_particle_counts",
                "plane_crossing_particle_events",
            ):
                legacy_fields.pop(key, None)

        if legacy_fields:
            unexpected = ", ".join(sorted(legacy_fields.keys()))
            raise TypeError(f"Unexpected scoring result fields: {unexpected}")

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "detector_crossing", detector_crossing)
        object.__setattr__(self, "plane_crossing", plane_crossing)
        object.__setattr__(self, "volume_stats", _normalize_volume_stats_map(volume_stats))
        object.__setattr__(self, "role_stats", _normalize_volume_stats_map(role_stats))

    @property
    def target_edep_enabled(self) -> bool:
        return self.target.edep_enabled

    @property
    def detector_crossings_enabled(self) -> bool:
        return self.detector_crossing.enabled

    @property
    def plane_crossings_enabled(self) -> bool:
        return self.plane_crossing.enabled

    @property
    def plane_crossing_name(self) -> str | None:
        return self.plane_crossing.name

    @property
    def plane_crossing_z_mm(self) -> float | None:
        return self.plane_crossing.z_mm

    @property
    def plane_crossing_count(self) -> int:
        return self.plane_crossing.count

    @property
    def plane_crossing_events(self) -> int:
        return self.plane_crossing.events

    @property
    def plane_crossing_forward_count(self) -> int:
        return self.plane_crossing.forward_count

    @property
    def plane_crossing_forward_events(self) -> int:
        return self.plane_crossing.forward_events

    @property
    def plane_crossing_reverse_count(self) -> int:
        return self.plane_crossing.reverse_count

    @property
    def plane_crossing_reverse_events(self) -> int:
        return self.plane_crossing.reverse_events

    @property
    def plane_crossing_mean_per_event(self) -> float:
        return self.plane_crossing.mean_per_event

    @property
    def plane_crossing_particle_counts(self) -> dict[str, int] | None:
        return self.plane_crossing.particles.counts

    @property
    def plane_crossing_particle_events(self) -> dict[str, int] | None:
        return self.plane_crossing.particles.events

    @property
    def detector_crossing_count(self) -> int:
        return self.detector_crossing.count

    @property
    def detector_crossing_events(self) -> int:
        return self.detector_crossing.events

    @property
    def detector_crossing_mean_per_event(self) -> float:
        return self.detector_crossing.mean_per_event

    @property
    def detector_crossing_particle_counts(self) -> dict[str, int] | None:
        return self.detector_crossing.particles.counts

    @property
    def detector_crossing_particle_events(self) -> dict[str, int] | None:
        return self.detector_crossing.particles.events

    @property
    def target_edep_total_mev(self) -> float:
        return self.target.edep_total_mev

    @property
    def target_edep_mean_mev_per_event(self) -> float:
        return self.target.edep_mean_mev_per_event

    @property
    def target_hit_events(self) -> int:
        return self.target.hit_events

    @property
    def target_step_count(self) -> int:
        return self.target.step_count

    @property
    def target_track_entries(self) -> int:
        return self.target.track_entries

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        payload.update(self.target.to_payload())
        payload.update(self.detector_crossing.to_payload())
        payload.update(self.plane_crossing.to_payload())
        payload["volume_stats"] = _volume_stats_map_to_payload(self.volume_stats)
        payload["role_stats"] = _volume_stats_map_to_payload(self.role_stats)
        return payload


def _normalize_volume_stats_map(
    raw_stats_map: dict[str, SimulationVolumeStatsResult | dict[str, float | int]] | None,
) -> dict[str, SimulationVolumeStatsResult] | None:
    if not raw_stats_map:
        return None
    result: dict[str, SimulationVolumeStatsResult] = {}
    for raw_name, raw_stats in raw_stats_map.items():
        name = str(raw_name).strip()
        if not name:
            continue
        if isinstance(raw_stats, SimulationVolumeStatsResult):
            result[name] = raw_stats
        else:
            result[name] = _parse_volume_stats(raw_stats)
    return result or None


@dataclass(frozen=True)
class SimulationDetectorResult:
    enabled: bool = False
    volume_name: str | None = None
    material: str | None = None
    position_mm: tuple[float, float, float] | None = None
    size_mm: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SimulationSourceModelResult:
    spot_radius_mm: float = 0.0
    divergence_half_angle_deg: float = 0.0
    spot_profile: str = "uniform_disk"
    spot_sigma_mm: float = 0.0
    divergence_profile: str = "uniform_cone"
    divergence_sigma_deg: float = 0.0


@dataclass(frozen=True)
class SimulationSourceSamplingResult:
    primary_count: int = 0
    sampled_position_mean_mm: tuple[float, float, float] | None = None
    sampled_position_rms_mm: tuple[float, float, float] | None = None
    sampled_direction_mean: tuple[float, float, float] | None = None
    sampled_direction_rms: tuple[float, float, float] | None = None


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
    source_model: SimulationSourceModelResult = field(default_factory=SimulationSourceModelResult)
    source_position_mm: tuple[float, float, float] | None = None
    source_direction: tuple[float, float, float] | None = None
    source_sampling: SimulationSourceSamplingResult = field(default_factory=SimulationSourceSamplingResult)
    payload_sha256: str | None = None
    geant4_version: str | None = None
    run_seed: int = 1337
    run_manifest: dict[str, Any] | None = None
    physics_list: str | None = None
    events: int = 0
    mode: str = "batch"
    scoring: SimulationScoringResult = field(default_factory=SimulationScoringResult)
    detector: SimulationDetectorResult = field(default_factory=SimulationDetectorResult)

    @property
    def source_spot_radius_mm(self) -> float:
        return self.source_model.spot_radius_mm

    @property
    def source_divergence_half_angle_deg(self) -> float:
        return self.source_model.divergence_half_angle_deg

    @property
    def source_spot_profile(self) -> str:
        return self.source_model.spot_profile

    @property
    def source_spot_sigma_mm(self) -> float:
        return self.source_model.spot_sigma_mm

    @property
    def source_divergence_profile(self) -> str:
        return self.source_model.divergence_profile

    @property
    def source_divergence_sigma_deg(self) -> float:
        return self.source_model.divergence_sigma_deg

    @property
    def source_primary_count(self) -> int:
        return self.source_sampling.primary_count

    @property
    def source_sampled_position_mean_mm(self) -> tuple[float, float, float] | None:
        return self.source_sampling.sampled_position_mean_mm

    @property
    def source_sampled_position_rms_mm(self) -> tuple[float, float, float] | None:
        return self.source_sampling.sampled_position_rms_mm

    @property
    def source_sampled_direction_mean(self) -> tuple[float, float, float] | None:
        return self.source_sampling.sampled_direction_mean

    @property
    def source_sampled_direction_rms(self) -> tuple[float, float, float] | None:
        return self.source_sampling.sampled_direction_rms

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scoring"] = self.scoring.to_payload()
        payload["detector"] = asdict(self.detector)
        # Compatibility fields for current API consumers while internal code uses structured objects.
        payload.update(
            {
                "source_spot_radius_mm": self.source_spot_radius_mm,
                "source_divergence_half_angle_deg": self.source_divergence_half_angle_deg,
                "source_spot_profile": self.source_spot_profile,
                "source_spot_sigma_mm": self.source_spot_sigma_mm,
                "source_divergence_profile": self.source_divergence_profile,
                "source_divergence_sigma_deg": self.source_divergence_sigma_deg,
                "source_primary_count": self.source_primary_count,
                "source_sampled_position_mean_mm": self.source_sampled_position_mean_mm,
                "source_sampled_position_rms_mm": self.source_sampled_position_rms_mm,
                "source_sampled_direction_mean": self.source_sampled_direction_mean,
                "source_sampled_direction_rms": self.source_sampled_direction_rms,
            }
        )
        return payload


def simulation_result_from_dict(data: dict[str, Any]) -> SimulationResult:
    scoring_data = data.get("scoring", {}) if isinstance(data.get("scoring"), dict) else {}
    volume_stats = _parse_volume_stats_map(scoring_data.get("volume_stats", {}))
    role_stats = _parse_volume_stats_map(scoring_data.get("role_stats", {}))
    if not volume_stats:
        volume_stats["Target"] = SimulationVolumeStatsResult(
            edep_total_mev=float(scoring_data.get("target_edep_total_mev", 0.0) or 0.0),
            edep_mean_mev_per_event=float(scoring_data.get("target_edep_mean_mev_per_event", 0.0) or 0.0),
            hit_events=int(scoring_data.get("target_hit_events", 0) or 0),
            step_count=int(scoring_data.get("target_step_count", 0) or 0),
            track_entries=int(scoring_data.get("target_track_entries", 0) or 0),
        )

    plane_z = scoring_data.get("plane_crossing_z_mm")
    scoring = SimulationScoringResult(
        target=SimulationTargetScoringResult(
            edep_enabled=bool(scoring_data.get("target_edep_enabled", False)),
            edep_total_mev=float(scoring_data.get("target_edep_total_mev", 0.0) or 0.0),
            edep_mean_mev_per_event=float(scoring_data.get("target_edep_mean_mev_per_event", 0.0) or 0.0),
            hit_events=int(scoring_data.get("target_hit_events", 0) or 0),
            step_count=int(scoring_data.get("target_step_count", 0) or 0),
            track_entries=int(scoring_data.get("target_track_entries", 0) or 0),
        ),
        detector_crossing=SimulationDetectorCrossingResult(
            enabled=bool(scoring_data.get("detector_crossings_enabled", False)),
            count=int(scoring_data.get("detector_crossing_count", 0) or 0),
            events=int(scoring_data.get("detector_crossing_events", 0) or 0),
            mean_per_event=float(scoring_data.get("detector_crossing_mean_per_event", 0.0) or 0.0),
            particles=SimulationParticleCrossingResult(
                counts=_coerce_int_map(scoring_data.get("detector_crossing_particle_counts")) or None,
                events=_coerce_int_map(scoring_data.get("detector_crossing_particle_events")) or None,
            ),
        ),
        plane_crossing=SimulationPlaneCrossingResult(
            enabled=bool(scoring_data.get("plane_crossings_enabled", False)),
            name=scoring_data.get("plane_crossing_name"),
            z_mm=float(plane_z) if plane_z is not None else None,
            count=int(scoring_data.get("plane_crossing_count", 0) or 0),
            events=int(scoring_data.get("plane_crossing_events", 0) or 0),
            forward_count=int(scoring_data.get("plane_crossing_forward_count", 0) or 0),
            forward_events=int(scoring_data.get("plane_crossing_forward_events", 0) or 0),
            reverse_count=int(scoring_data.get("plane_crossing_reverse_count", 0) or 0),
            reverse_events=int(scoring_data.get("plane_crossing_reverse_events", 0) or 0),
            mean_per_event=float(scoring_data.get("plane_crossing_mean_per_event", 0.0) or 0.0),
            particles=SimulationParticleCrossingResult(
                counts=_coerce_int_map(scoring_data.get("plane_crossing_particle_counts")) or None,
                events=_coerce_int_map(scoring_data.get("plane_crossing_particle_events")) or None,
            ),
        ),
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
        source_model=SimulationSourceModelResult(
            spot_radius_mm=float(data.get("source_spot_radius_mm", 0.0) or 0.0),
            divergence_half_angle_deg=float(data.get("source_divergence_half_angle_deg", 0.0) or 0.0),
            spot_profile=str(data.get("source_spot_profile") or "uniform_disk"),
            spot_sigma_mm=float(data.get("source_spot_sigma_mm", 0.0) or 0.0),
            divergence_profile=str(data.get("source_divergence_profile") or "uniform_cone"),
            divergence_sigma_deg=float(data.get("source_divergence_sigma_deg", 0.0) or 0.0),
        ),
        source_position_mm=_coerce_triplet(data.get("source_position_mm")),
        source_direction=_coerce_triplet(data.get("source_direction")),
        source_sampling=SimulationSourceSamplingResult(
            primary_count=int(data.get("source_primary_count", 0) or 0),
            sampled_position_mean_mm=_coerce_triplet(data.get("source_sampled_position_mean_mm")),
            sampled_position_rms_mm=_coerce_triplet(data.get("source_sampled_position_rms_mm")),
            sampled_direction_mean=_coerce_triplet(data.get("source_sampled_direction_mean")),
            sampled_direction_rms=_coerce_triplet(data.get("source_sampled_direction_rms")),
        ),
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
    volume_stats: dict[str, dict[str, float | int] | SimulationVolumeStatsResult] | None,
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
        aggregate: dict[str, float | int] = {
            "edep_total_mev": 0.0,
            "edep_mean_mev_per_event": 0.0,
            "hit_events": 0,
            "crossing_events": 0,
            "crossing_count": 0,
            "crossing_mean_per_event": 0.0,
            "step_count": 0,
            "track_entries": 0,
        }
        matched = False
        for name in names:
            stats = volume_stats.get(name)
            if not isinstance(stats, (dict, SimulationVolumeStatsResult)):
                continue
            matched = True
            aggregate["edep_total_mev"] += float(stats.get("edep_total_mev", 0.0) or 0.0)
            aggregate["edep_mean_mev_per_event"] += float(stats.get("edep_mean_mev_per_event", 0.0) or 0.0)
            aggregate["hit_events"] += int(stats.get("hit_events", 0) or 0)
            aggregate["crossing_events"] += int(stats.get("crossing_events", 0) or 0)
            aggregate["crossing_count"] += int(stats.get("crossing_count", 0) or 0)
            aggregate["crossing_mean_per_event"] += float(stats.get("crossing_mean_per_event", 0.0) or 0.0)
            aggregate["step_count"] += int(stats.get("step_count", 0) or 0)
            aggregate["track_entries"] += int(stats.get("track_entries", 0) or 0)
        if matched:
            role_stats[role_name] = aggregate
    return role_stats

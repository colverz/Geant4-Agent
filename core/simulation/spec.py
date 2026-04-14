from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeometryRuntimeSpec:
    structure: str
    material: str
    root_volume_name: str = "Target"
    size_x_mm: float | None = None
    size_y_mm: float | None = None
    size_z_mm: float | None = None
    radius_mm: float | None = None
    half_length_mm: float | None = None


@dataclass(frozen=True)
class DetectorRuntimeSpec:
    volume_name: str = "Detector"
    material: str = "G4_Si"
    position_mm: tuple[float, float, float] = (0.0, 0.0, 100.0)
    size_x_mm: float = 20.0
    size_y_mm: float = 20.0
    size_z_mm: float = 2.0


@dataclass(frozen=True)
class SourceRuntimeSpec:
    source_type: str
    particle: str
    energy_mev: float
    position_mm: tuple[float, float, float]
    direction_vec: tuple[float, float, float]


@dataclass(frozen=True)
class PhysicsRuntimeSpec:
    physics_list: str


@dataclass(frozen=True)
class RunControlSpec:
    events: int = 1
    mode: str = "batch"
    seed: int = 1337


@dataclass(frozen=True)
class ScoringSpec:
    target_edep: bool = True
    detector_crossings: bool = True
    volume_names: tuple[str, ...] = field(default_factory=lambda: ("Target",))
    volume_roles: dict[str, tuple[str, ...]] = field(default_factory=lambda: {"target": ("Target",)})


@dataclass(frozen=True)
class SimulationSpec:
    geometry: GeometryRuntimeSpec
    source: SourceRuntimeSpec
    physics: PhysicsRuntimeSpec
    run: RunControlSpec = field(default_factory=RunControlSpec)
    scoring: ScoringSpec = field(default_factory=ScoringSpec)
    detector: DetectorRuntimeSpec | None = None

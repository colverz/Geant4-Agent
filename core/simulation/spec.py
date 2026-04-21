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
class BeamSpotSpec:
    profile: str = "uniform_disk"
    radius_mm: float = 0.0
    sigma_mm: float = 0.0


@dataclass(frozen=True)
class BeamDivergenceSpec:
    profile: str = "uniform_cone"
    half_angle_deg: float = 0.0
    sigma_deg: float = 0.0


@dataclass(frozen=True)
class BeamModelSpec:
    spot: BeamSpotSpec = field(default_factory=BeamSpotSpec)
    divergence: BeamDivergenceSpec = field(default_factory=BeamDivergenceSpec)


@dataclass(frozen=True)
class SourceRuntimeSpec:
    source_type: str
    particle: str
    energy_mev: float
    position_mm: tuple[float, float, float]
    direction_vec: tuple[float, float, float]
    beam_model: BeamModelSpec = field(default_factory=BeamModelSpec)

    @property
    def spot_radius_mm(self) -> float:
        return self.beam_model.spot.radius_mm

    @property
    def spot_profile(self) -> str:
        return self.beam_model.spot.profile

    @property
    def spot_sigma_mm(self) -> float:
        return self.beam_model.spot.sigma_mm

    @property
    def divergence_half_angle_deg(self) -> float:
        return self.beam_model.divergence.half_angle_deg

    @property
    def divergence_profile(self) -> str:
        return self.beam_model.divergence.profile

    @property
    def divergence_sigma_deg(self) -> float:
        return self.beam_model.divergence.sigma_deg


@dataclass(frozen=True)
class PhysicsRuntimeSpec:
    physics_list: str


@dataclass(frozen=True)
class RunControlSpec:
    events: int = 1
    mode: str = "batch"
    seed: int = 1337


@dataclass(frozen=True)
class ScoringPlaneSpec:
    name: str = "ScoringPlane"
    z_mm: float = 0.0


@dataclass(frozen=True)
class ScoringSpec:
    target_edep: bool = True
    detector_crossings: bool = True
    plane_crossings: bool = False
    scoring_plane: ScoringPlaneSpec | None = None
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

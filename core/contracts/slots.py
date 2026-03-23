from __future__ import annotations

from dataclasses import dataclass, field

from core.orchestrator.types import Intent


@dataclass
class GeometrySlots:
    kind: str | None = None
    size_triplet_mm: list[float] | None = None
    radius_mm: float | None = None
    half_length_mm: float | None = None
    radius1_mm: float | None = None
    radius2_mm: float | None = None
    x1_mm: float | None = None
    x2_mm: float | None = None
    y1_mm: float | None = None
    y2_mm: float | None = None
    z_mm: float | None = None
    z_planes_mm: list[float] | None = None
    radii_mm: list[float] | None = None
    trap_x1_mm: float | None = None
    trap_x2_mm: float | None = None
    trap_x3_mm: float | None = None
    trap_x4_mm: float | None = None
    trap_y1_mm: float | None = None
    trap_y2_mm: float | None = None
    trap_z_mm: float | None = None
    para_x_mm: float | None = None
    para_y_mm: float | None = None
    para_z_mm: float | None = None
    para_alpha_deg: float | None = None
    para_theta_deg: float | None = None
    para_phi_deg: float | None = None
    torus_major_radius_mm: float | None = None
    torus_minor_radius_mm: float | None = None
    ellipsoid_ax_mm: float | None = None
    ellipsoid_by_mm: float | None = None
    ellipsoid_cz_mm: float | None = None
    elltube_ax_mm: float | None = None
    elltube_by_mm: float | None = None
    elltube_hz_mm: float | None = None
    polyhedra_sides: int | None = None
    tilt_x_deg: float | None = None
    tilt_y_deg: float | None = None


@dataclass
class MaterialsSlots:
    primary: str | None = None


@dataclass
class SourceSlots:
    kind: str | None = None
    particle: str | None = None
    energy_mev: float | None = None
    position_mm: list[float] | None = None
    direction_vec: list[float] | None = None


@dataclass
class PhysicsSlots:
    explicit_list: str | None = None
    recommendation_intent: str | None = None


@dataclass
class OutputSlots:
    format: str | None = None
    path: str | None = None


@dataclass
class SlotFrame:
    intent: Intent = Intent.OTHER
    confidence: float = 0.0
    normalized_text: str = ""
    target_slots: list[str] = field(default_factory=list)
    geometry: GeometrySlots = field(default_factory=GeometrySlots)
    materials: MaterialsSlots = field(default_factory=MaterialsSlots)
    source: SourceSlots = field(default_factory=SourceSlots)
    physics: PhysicsSlots = field(default_factory=PhysicsSlots)
    output: OutputSlots = field(default_factory=OutputSlots)
    notes: list[str] = field(default_factory=list)

    def has_content(self) -> bool:
        return any(
            [
                self.geometry.kind,
                self.geometry.size_triplet_mm,
                self.geometry.radius_mm is not None,
                self.geometry.half_length_mm is not None,
                self.geometry.radius1_mm is not None,
                self.geometry.radius2_mm is not None,
                self.geometry.x1_mm is not None,
                self.geometry.x2_mm is not None,
                self.geometry.y1_mm is not None,
                self.geometry.y2_mm is not None,
                self.geometry.z_mm is not None,
                self.geometry.z_planes_mm,
                self.geometry.radii_mm,
                self.geometry.trap_x1_mm is not None,
                self.geometry.trap_x2_mm is not None,
                self.geometry.trap_x3_mm is not None,
                self.geometry.trap_x4_mm is not None,
                self.geometry.trap_y1_mm is not None,
                self.geometry.trap_y2_mm is not None,
                self.geometry.trap_z_mm is not None,
                self.geometry.para_x_mm is not None,
                self.geometry.para_y_mm is not None,
                self.geometry.para_z_mm is not None,
                self.geometry.para_alpha_deg is not None,
                self.geometry.para_theta_deg is not None,
                self.geometry.para_phi_deg is not None,
                self.geometry.torus_major_radius_mm is not None,
                self.geometry.torus_minor_radius_mm is not None,
                self.geometry.ellipsoid_ax_mm is not None,
                self.geometry.ellipsoid_by_mm is not None,
                self.geometry.ellipsoid_cz_mm is not None,
                self.geometry.elltube_ax_mm is not None,
                self.geometry.elltube_by_mm is not None,
                self.geometry.elltube_hz_mm is not None,
                self.geometry.polyhedra_sides is not None,
                self.geometry.tilt_x_deg is not None,
                self.geometry.tilt_y_deg is not None,
                self.materials.primary,
                self.source.kind,
                self.source.particle,
                self.source.energy_mev is not None,
                self.source.position_mm,
                self.source.direction_vec,
                self.physics.explicit_list,
                self.physics.recommendation_intent,
                self.output.format,
                self.output.path,
            ]
        )

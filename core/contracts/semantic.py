from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeometryFrame:
    structure: Optional[str] = None
    chosen_skeleton: Optional[str] = None
    graph_program: Optional[Dict[str, Any]] = None
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaterialsFrame:
    selected_materials: List[str] = field(default_factory=list)
    volume_material_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class SourceFrame:
    type: Optional[str] = None
    particle: Optional[str] = None
    energy: Optional[Dict[str, float]] = None
    position: Optional[Dict[str, float]] = None
    direction: Optional[Dict[str, float]] = None


@dataclass
class PhysicsFrame:
    physics_list: Optional[str] = None


@dataclass
class OutputFrame:
    format: Optional[str] = None


@dataclass
class EnvironmentFrame:
    temperature: Optional[float] = None
    pressure: Optional[float] = None


@dataclass
class SemanticFrame:
    geometry: GeometryFrame = field(default_factory=GeometryFrame)
    materials: MaterialsFrame = field(default_factory=MaterialsFrame)
    source: SourceFrame = field(default_factory=SourceFrame)
    physics: PhysicsFrame = field(default_factory=PhysicsFrame)
    output: OutputFrame = field(default_factory=OutputFrame)
    environment: EnvironmentFrame = field(default_factory=EnvironmentFrame)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometry": self.geometry.__dict__,
            "materials": self.materials.__dict__,
            "source": self.source.__dict__,
            "physics": self.physics.__dict__,
            "output": self.output.__dict__,
            "environment": self.environment.__dict__,
            "notes": list(self.notes),
        }

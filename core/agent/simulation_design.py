from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


SIMULATION_DESIGN_SCHEMA_VERSION = "geant4_agent_simulation_design_candidate.v1"
SIMULATION_DESIGN_ANNOTATIONS_PATH = Path("knowledge/data/simulation_design_annotations.json")
ALLOWED_NEXT_ACTIONS = {
    "build_candidate_config",
    "ask_user_to_choose_approximation",
    "unsupported_capability",
    "needs_more_information",
}


@dataclass(frozen=True)
class SimulationDesignCandidate:
    schema_version: str = SIMULATION_DESIGN_SCHEMA_VERSION
    goal: str = ""
    recommended_setup: dict[str, Any] = field(default_factory=dict)
    observables: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    simplifications: tuple[str, ...] = ()
    unsupported_capabilities: tuple[str, ...] = ()
    user_decisions_required: tuple[str, ...] = ()
    knowledge_references: tuple[str, ...] = ()
    capability_check: dict[str, Any] = field(default_factory=dict)
    next_action: str = "needs_more_information"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "goal": self.goal,
            "recommended_setup": dict(self.recommended_setup),
            "observables": list(self.observables),
            "assumptions": list(self.assumptions),
            "simplifications": list(self.simplifications),
            "unsupported_capabilities": list(self.unsupported_capabilities),
            "user_decisions_required": list(self.user_decisions_required),
            "knowledge_references": list(self.knowledge_references),
            "capability_check": dict(self.capability_check),
            "next_action": self.next_action,
        }


def load_simulation_design_annotations(path: Path = SIMULATION_DESIGN_ANNOTATIONS_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def default_runtime_capabilities(annotations: dict[str, Any] | None = None) -> dict[str, Any]:
    data = annotations if isinstance(annotations, dict) else load_simulation_design_annotations()
    runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
    return {
        "supported_geometry": list(runtime.get("supported_geometry") or ["single_box"]),
        "supported_sources": list(runtime.get("supported_sources") or ["beam", "point"]),
        "supported_scoring": list(runtime.get("supported_scoring") or ["target_edep", "detector_crossing_count"]),
        "unsupported_capabilities": list(runtime.get("unsupported_capabilities") or []),
    }


def validate_simulation_design_annotations(annotations: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for section in ("materials", "sources", "scoring", "geometry", "runtime"):
        if not isinstance(annotations.get(section), dict):
            errors.append(f"missing_section:{section}")
    materials = annotations.get("materials") if isinstance(annotations.get("materials"), dict) else {}
    for material in (
        "G4_Pb",
        "G4_WATER",
        "G4_Si",
        "G4_AIR",
        "G4_POLYETHYLENE",
        "G4_PLASTIC_SC_VINYLTOLUENE",
    ):
        tags = (materials.get(material) or {}).get("tags") if isinstance(materials.get(material), dict) else None
        if not tags:
            errors.append(f"missing_material_tags:{material}")
    runtime = annotations.get("runtime") if isinstance(annotations.get("runtime"), dict) else {}
    for key in ("supported_geometry", "supported_sources", "supported_scoring", "unsupported_capabilities"):
        if not isinstance(runtime.get(key), list):
            errors.append(f"missing_runtime_capability:{key}")
    return errors


def build_simulation_design_reference_pack(
    user_goal: str,
    runtime_capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    annotations = load_simulation_design_annotations()
    capabilities = runtime_capabilities if isinstance(runtime_capabilities, dict) else default_runtime_capabilities(annotations)
    low = str(user_goal or "").lower()
    material_refs = _select_material_refs(low, annotations)
    source_refs = _select_refs(low, annotations.get("sources", {}), default_keys=("beam", "point"))
    scoring_refs = _select_scoring_refs(low, annotations)
    geometry_refs = _select_geometry_refs(low, annotations)
    return {
        "schema_version": "geant4_agent_simulation_design_reference_pack.v1",
        "materials": material_refs,
        "sources": source_refs,
        "scoring": scoring_refs,
        "geometry": geometry_refs,
        "runtime_capabilities": capabilities,
    }


def build_simulation_design_candidate(
    user_goal: str,
    *,
    current_config: dict[str, Any] | None = None,
    runtime_capabilities: dict[str, Any] | None = None,
) -> SimulationDesignCandidate:
    reference_pack = build_simulation_design_reference_pack(user_goal, runtime_capabilities)
    low = str(user_goal or "").lower()
    setup = _recommended_setup(low)
    observables = _observables(low)
    assumptions = _assumptions(low)
    simplifications = _simplifications(low)
    unsupported = _unsupported_capabilities(low)
    decisions = _user_decisions_required(simplifications, unsupported)
    capability_check = check_simulation_design_capability(
        {
            "recommended_setup": setup,
            "observables": list(observables),
            "simplifications": list(simplifications),
            "unsupported_capabilities": list(unsupported),
        },
        reference_pack["runtime_capabilities"],
    )
    next_action = _next_action(capability_check, decisions)
    return SimulationDesignCandidate(
        goal=str(user_goal or "").strip(),
        recommended_setup=setup,
        observables=observables,
        assumptions=assumptions,
        simplifications=simplifications,
        unsupported_capabilities=tuple(capability_check["unsupported_capabilities"]),
        user_decisions_required=tuple(decisions),
        knowledge_references=tuple(_knowledge_reference_ids(reference_pack)),
        capability_check=capability_check,
        next_action=next_action,
    )


def check_simulation_design_capability(
    candidate: dict[str, Any] | SimulationDesignCandidate,
    runtime_capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = candidate.to_dict() if isinstance(candidate, SimulationDesignCandidate) else dict(candidate)
    capabilities = runtime_capabilities if isinstance(runtime_capabilities, dict) else default_runtime_capabilities()
    setup = payload.get("recommended_setup") if isinstance(payload.get("recommended_setup"), dict) else {}
    geometry = str(setup.get("geometry") or "")
    source = str(setup.get("source") or "")
    observables = [str(item) for item in payload.get("observables") or []]
    explicit_unsupported = [str(item) for item in payload.get("unsupported_capabilities") or []]
    unsupported = list(dict.fromkeys(explicit_unsupported))
    requires_approval = bool(payload.get("user_decisions_required")) or _simplifications_require_approval(
        payload.get("simplifications")
    )

    if geometry and geometry not in set(capabilities.get("supported_geometry") or []):
        if geometry in {"step_wedge", "pipe"}:
            requires_approval = True
        else:
            unsupported.append(f"{geometry}_geometry")
    if source and source not in set(capabilities.get("supported_sources") or []):
        unsupported.append(f"{source}_source")
    supported_scoring = set(capabilities.get("supported_scoring") or [])
    for observable in observables:
        if observable not in supported_scoring and observable not in {"transmission_factor"}:
            unsupported.append(observable)
    unsupported = list(dict.fromkeys(unsupported))
    return {
        "ok": not unsupported and not requires_approval,
        "supported": not unsupported,
        "requires_user_approval": requires_approval,
        "unsupported_capabilities": unsupported,
        "checked_geometry": geometry,
        "checked_source": source,
        "checked_observables": observables,
    }


def _simplifications_require_approval(values: Any) -> bool:
    text = " ".join(str(item) for item in (values or [])).lower()
    if not text:
        return False
    approval_markers = (
        "approximation",
        "approximate",
        "approximated",
        "requires user",
        "user approval",
        "approve",
        "unsupported",
        "multi-run",
        "multiple independent runs",
        "sweep",
        "proxy",
        "slab",
        "step wedge",
        "pipe",
    )
    return any(marker in text for marker in approval_markers)


def _next_action(capability_check: dict[str, Any], decisions: list[str]) -> str:
    if capability_check.get("unsupported_capabilities"):
        return "unsupported_capability"
    if decisions or capability_check.get("requires_user_approval"):
        return "ask_user_to_choose_approximation"
    if capability_check.get("supported"):
        return "build_candidate_config"
    return "needs_more_information"


def _select_material_refs(text: str, annotations: dict[str, Any]) -> list[dict[str, Any]]:
    materials = annotations.get("materials") if isinstance(annotations.get("materials"), dict) else {}
    selected: list[str] = []
    cues = {
        "G4_Pb": ("lead", "shield", "gamma transmission", "铅", "屏蔽"),
        "G4_WATER": ("water", "phantom", "dose", "水", "模体", "剂量"),
        "G4_Si": ("silicon", "detector", "硅", "探测器"),
        "G4_AIR": ("air", "gap", "空气", "气隙"),
        "G4_POLYETHYLENE": ("polyethylene", "neutron", "moderation", "聚乙烯", "中子", "慢化"),
        "G4_PLASTIC_SC_VINYLTOLUENE": ("scintillator", "plastic", "闪烁体", "塑料"),
        "G4_STAINLESS-STEEL": ("steel", "pipe", "wedge", "钢", "管", "楔"),
        "G4_Al": ("aluminum", "aluminium", "void", "铝", "空洞", "缺陷"),
    }
    for material, tokens in cues.items():
        if any(token in text for token in tokens):
            selected.append(material)
    if not selected:
        selected.extend(["G4_Pb", "G4_Si", "G4_AIR"])
    return [_ref_item(material, materials.get(material)) for material in dict.fromkeys(selected) if material in materials]


def _select_refs(text: str, section: Any, *, default_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    data = section if isinstance(section, dict) else {}
    selected = [key for key in data if key in text]
    if not selected:
        selected = list(default_keys)
    return [_ref_item(key, data.get(key)) for key in dict.fromkeys(selected) if key in data]


def _select_scoring_refs(text: str, annotations: dict[str, Any]) -> list[dict[str, Any]]:
    scoring = annotations.get("scoring") if isinstance(annotations.get("scoring"), dict) else {}
    selected = ["target_edep", "detector_crossing_count"]
    if "detector" in text:
        selected.append("detector_edep")
    if "plane" in text:
        selected.append("plane_crossing_count")
    if "contrast" in text or "void" in text or "region" in text or "对比" in text or "空洞" in text or "区域" in text:
        selected.append("region_contrast")
    if "depth" in text or "bragg" in text or "dose" in text or "深度" in text or "剂量" in text:
        selected.append("depth_bins")
    return [_ref_item(key, scoring.get(key)) for key in dict.fromkeys(selected) if key in scoring]


def _select_geometry_refs(text: str, annotations: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = annotations.get("geometry") if isinstance(annotations.get("geometry"), dict) else {}
    selected = ["single_box"]
    for key, tokens in {
        "step_wedge": ("step wedge", "wedge", "阶梯楔", "楔"),
        "pipe": ("pipe", "corrosion", "管", "腐蚀"),
        "void": ("void", "空洞", "孔洞", "缺陷"),
        "inclusion": ("inclusion", "夹杂", "内含物"),
        "multi_layer": ("multi-layer", "graded", "layer", "多层", "分层"),
    }.items():
        if any(token in text for token in tokens):
            selected.append(key)
    return [_ref_item(key, geometry.get(key)) for key in dict.fromkeys(selected) if key in geometry]


def _ref_item(name: str, value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return {
        "id": name,
        "tags": list(payload.get("tags") or []),
        "use_cases": list(payload.get("use_cases") or []),
        "notes": str(payload.get("notes") or ""),
    }


def _recommended_setup(text: str) -> dict[str, Any]:
    geometry = "single_box"
    material = "G4_Cu"
    source = "beam"
    detector = None
    scoring = ["target_edep"]
    has_explicit_target_material = False
    if "lead" in text or "shield" in text or "铅" in text or "屏蔽" in text:
        material = "G4_Pb"
        has_explicit_target_material = True
        detector = {"enabled": True, "material": "G4_Si", "role": "downstream_detector"}
        scoring = ["detector_crossing_count", "detector_edep", "target_edep"]
    if "polyethylene" in text or "neutron" in text or "聚乙烯" in text or "中子" in text:
        material = "G4_POLYETHYLENE"
        has_explicit_target_material = True
        scoring = ["plane_crossing_count", "target_edep"]
    if "water" in text or "phantom" in text or "水" in text or "模体" in text:
        material = "G4_WATER"
        has_explicit_target_material = True
        scoring = ["target_edep", "depth_bins"]
    if "silicon detector" in text or "硅探测器" in text:
        if not has_explicit_target_material:
            material = "G4_AIR"
        detector = {"enabled": True, "material": "G4_Si", "role": "detector"}
        if scoring == ["target_edep"]:
            scoring = ["detector_crossing_count", "detector_edep"]
    if "scintillator" in text or "闪烁体" in text:
        if not has_explicit_target_material:
            material = "G4_AIR"
        detector = {"enabled": True, "material": "G4_PLASTIC_SC_VINYLTOLUENE", "role": "detector"}
        if scoring == ["target_edep"]:
            scoring = ["detector_crossing_count", "detector_edep"]
    if "step wedge" in text or "wedge" in text or "阶梯楔" in text:
        geometry = "step_wedge"
        material = "G4_STAINLESS-STEEL"
    if "pipe" in text or "corrosion" in text or "钢管" in text or "腐蚀" in text:
        geometry = "pipe"
        material = "G4_STAINLESS-STEEL"
        scoring = ["detector_crossing_count", "transmission_factor"]
    if "void" in text or "空洞" in text or "孔洞" in text:
        geometry = "void"
        material = "G4_Al"
        scoring = ["region_contrast", "detector_edep"]
    return {
        "geometry": geometry,
        "material": material,
        "source": source,
        "detector": detector,
        "scoring": scoring,
    }


def _observables(text: str) -> tuple[str, ...]:
    setup = _recommended_setup(text)
    observables = list(setup.get("scoring") or [])
    if "transmission" in text:
        observables.append("transmission_factor")
    return tuple(dict.fromkeys(observables))


def _assumptions(text: str) -> tuple[str, ...]:
    assumptions = ["Use current single-thread deterministic runtime defaults unless the user specifies otherwise."]
    if "gamma" in text and "energy" not in text and "mev" not in text and "kev" not in text:
        assumptions.append("Gamma energy is not explicit and must be confirmed before final configuration.")
    if "detector" in text:
        assumptions.append("Detector is treated as downstream along the source direction when direction is not otherwise specified.")
    return tuple(assumptions)


def _simplifications(text: str) -> tuple[str, ...]:
    simplifications: list[str] = []
    if "pipe" in text or "corrosion" in text:
        simplifications.append("Approximate curved pipe wall as slab thickness paths only if the user approves.")
    return tuple(simplifications)


def _unsupported_capabilities(text: str) -> tuple[str, ...]:
    unsupported: list[str] = []
    if "cad" in text:
        unsupported.append("cad_import")
    if "moving" in text or "motion" in text:
        unsupported.append("moving_geometry")
    return tuple(dict.fromkeys(unsupported))


def _user_decisions_required(simplifications: tuple[str, ...], unsupported: tuple[str, ...]) -> list[str]:
    decisions = [f"Approve simplification: {item}" for item in simplifications]
    if unsupported:
        decisions.append("Choose a supported approximation or defer until runtime capability is added.")
    return decisions


def _knowledge_reference_ids(reference_pack: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for section in ("materials", "sources", "scoring", "geometry"):
        for item in reference_pack.get(section) or []:
            if isinstance(item, dict) and item.get("id"):
                refs.append(f"{section}:{item['id']}")
    return list(dict.fromkeys(refs))


__all__ = [
    "ALLOWED_NEXT_ACTIONS",
    "SIMULATION_DESIGN_SCHEMA_VERSION",
    "SimulationDesignCandidate",
    "build_simulation_design_candidate",
    "build_simulation_design_reference_pack",
    "check_simulation_design_capability",
    "default_runtime_capabilities",
    "load_simulation_design_annotations",
    "validate_simulation_design_annotations",
]

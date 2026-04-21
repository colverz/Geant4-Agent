from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.config.llm_prompt_registry import (
    STRICT_SLOT_PROMPT_PROFILE,
    build_strict_slot_prompt,
)
from core.config.output_format_registry import canonical_output_format
from core.domain.lexicon import BASE_MATERIAL_ALIASES
from core.geometry.family_catalog import SUPPORTED_GEOMETRY_KINDS
from core.orchestrator.types import Intent
from core.slots.slot_frame import SlotFrame
from core.slots.slot_validator import validate_slot_frame
from core.validation.error_codes import (
    E_LLM_SLOT_CALL_FAILED,
    E_LLM_SLOT_EMPTY,
    E_LLM_SLOT_PARSE_FAILED,
    E_LLM_SLOT_SCHEMA_INVALID,
)
from nlu.llm_support.ollama_client import chat, extract_json
from nlu.uncertainty import (
    has_grounded_payload_for_target,
    has_uncertainty_signal,
    infer_unresolved_targets,
)


logger = logging.getLogger(__name__)


_INTENT_MAP = {x.value: x for x in Intent}
_GEOMETRY_ALIASES = {
    "box": "box",
    "cube": "box",
    "cuboid": "box",
    "single_box": "box",
    "cylinder": "cylinder",
    "cylindrical": "cylinder",
    "tubs": "cylinder",
    "single_tubs": "cylinder",
    "sphere": "sphere",
    "single_sphere": "sphere",
    "orb": "orb",
    "single_orb": "orb",
    "cons": "cons",
    "cone": "cons",
    "frustum": "cons",
    "single_cons": "cons",
    "trd": "trd",
    "trapezoid": "trd",
    "trapezoidal": "trd",
    "single_trd": "trd",
    "polycone": "polycone",
    "single_polycone": "polycone",
    "polyhedra": "polyhedra",
    "polyhedron": "polyhedra",
    "single_polyhedra": "polyhedra",
    "cuttubs": "cuttubs",
    "cut tubs": "cuttubs",
    "single_cuttubs": "cuttubs",
    "trap": "trap",
    "single_trap": "trap",
    "para": "para",
    "parallelepiped": "para",
    "skewed box": "para",
    "single_para": "para",
    "torus": "torus",
    "donut": "torus",
    "ring tube": "torus",
    "single_torus": "torus",
    "ellipsoid": "ellipsoid",
    "elliptic": "ellipsoid",
    "single_ellipsoid": "ellipsoid",
    "elltube": "elltube",
    "elliptical tube": "elltube",
    "ellipse tube": "elltube",
    "elliptic tube": "elltube",
    "single_elltube": "elltube",
}
_SOURCE_ALIASES = {
    "point": "point",
    "point source": "point",
    "beam": "beam",
    "plane": "plane",
    "plane source": "plane",
    "isotropic": "isotropic",
}
_PARTICLE_ALIASES = {
    "gamma": "gamma",
    "photon": "gamma",
    "electron": "e-",
    "e-": "e-",
    "proton": "proton",
    "neutron": "neutron",
}
_MATERIAL_ALIASES = dict(BASE_MATERIAL_ALIASES)
_NULL_LITERALS = {"null", "none", "n/a", "na", "unknown", "\u65e0", "\u672a\u77e5", "\u7a7a"}
_AXIS_VECTORS = {
    "+x": ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    "-x": ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    "+y": ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
    "-y": ([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]),
    "+z": ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
    "-z": ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
}


def _normalize_vec3(values: list[float]) -> list[float] | None:
    norm = sum(component * component for component in values) ** 0.5
    if norm <= 1e-9:
        return None
    return [component / norm for component in values]


@dataclass
class LlmSlotBuildResult:
    ok: bool
    frame: SlotFrame | None
    normalized_text: str
    confidence: float
    llm_raw: str
    fallback_reason: str | None
    schema_errors: list[str]
    stage_trace: dict[str, Any] = field(default_factory=dict)


def _clean_response(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    return text


def _build_prompt(user_text: str, context_summary: str) -> str:
    return build_strict_slot_prompt(user_text, context_summary)


def _intent_from_any(value: Any) -> Intent:
    return _INTENT_MAP.get(str(value or "").strip().upper(), Intent.OTHER)


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in _NULL_LITERALS:
        return None
    return text


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in _NULL_LITERALS:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.match(r"^\s*([-+]?\d*\.?\d+)\s*(mev|gev|kev)\s*$", text)
        if not m:
            return None
        scalar = float(m.group(1))
        unit = m.group(2)
        if unit == "gev":
            return scalar * 1000.0
        if unit == "kev":
            return scalar * 0.001
        return scalar


def _to_mm(value: float, unit: str) -> float:
    u = unit.lower()
    if u in {"m", "\u7c73"}:
        return value * 1000.0
    if u in {"cm", "\u5398\u7c73"}:
        return value * 10.0
    return value


def _coerce_length_mm(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in _NULL_LITERALS:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.fullmatch(r"([-+]?\d*\.?\d+)\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)", text)
        if not m:
            return None
        return _to_mm(float(m.group(1)), m.group(2))


def _coerce_triplet_mm(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            pass
    text = str(value or "").strip().lower().replace("\u00d7", "x")
    if not text or text in _NULL_LITERALS:
        return None
    num = r"[-+]?\d*\.?\d+"
    unit = r"(mm|cm|m)"
    m1 = re.fullmatch(rf"({num})\s*{unit}\s*[,x]\s*({num})\s*{unit}\s*[,x]\s*({num})\s*{unit}", text)
    if m1:
        return [
            _to_mm(float(m1.group(1)), m1.group(2)),
            _to_mm(float(m1.group(3)), m1.group(4)),
            _to_mm(float(m1.group(5)), m1.group(6)),
        ]
    m2 = re.fullmatch(rf"({num})\s*[,x]\s*({num})\s*[,x]\s*({num})\s*{unit}", text)
    if m2:
        return [
            _to_mm(float(m2.group(1)), m2.group(4)),
            _to_mm(float(m2.group(2)), m2.group(4)),
            _to_mm(float(m2.group(3)), m2.group(4)),
        ]
    return None


def _coerce_vec3(value: Any, *, metric: bool) -> list[float] | None:
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, list):
            value = nested
    if isinstance(value, list) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            pass
    text = str(value or "").strip().lower()
    if not text or text in _NULL_LITERALS:
        return None
    compact = text.replace(" ", "")
    if not metric and compact in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        mapping = {
            "+x": [1.0, 0.0, 0.0],
            "-x": [-1.0, 0.0, 0.0],
            "+y": [0.0, 1.0, 0.0],
            "-y": [0.0, -1.0, 0.0],
            "+z": [0.0, 0.0, 1.0],
            "-z": [0.0, 0.0, -1.0],
        }
        return mapping[compact]
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\b(mm|cm|m)\b", "", text)
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if len(parts) != 3:
        return None
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError:
        return None


def _coerce_pair_mm(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except Exception:
            return None
    text = str(value or "").strip().lower().replace("\u00d7", "x")
    if not text or text in _NULL_LITERALS:
        return None
    num = r"[-+]?\d*\.?\d+"
    unit = r"(mm|cm|m)"
    m1 = re.fullmatch(rf"({num})\s*{unit}\s*[,x]\s*({num})\s*{unit}", text)
    if m1:
        return [
            _to_mm(float(m1.group(1)), m1.group(2)),
            _to_mm(float(m1.group(3)), m1.group(4)),
        ]
    m2 = re.fullmatch(rf"({num})\s*[,x]\s*({num})\s*{unit}", text)
    if m2:
        return [
            _to_mm(float(m2.group(1)), m2.group(3)),
            _to_mm(float(m2.group(2)), m2.group(3)),
        ]
    return None


def _canonical_material(value: Any) -> str | None:
    text = _clean_scalar(value)
    if not text:
        return None
    return _MATERIAL_ALIASES.get(text.lower(), text)


def _canonical_geometry_kind(value: Any) -> str | None:
    text = (_clean_scalar(value) or "").lower()
    if not text:
        return None
    return _GEOMETRY_ALIASES.get(text, text if text in SUPPORTED_GEOMETRY_KINDS else None)


def _canonical_source_kind(value: Any) -> str | None:
    text = (_clean_scalar(value) or "").lower()
    if not text:
        return None
    return _SOURCE_ALIASES.get(text, text if text in _SOURCE_ALIASES.values() else None)


def _canonical_particle(value: Any) -> str | None:
    text = (_clean_scalar(value) or "").lower()
    if not text:
        return None
    return _PARTICLE_ALIASES.get(text, text)


def _canonical_output_format(value: Any) -> str | None:
    return canonical_output_format(_clean_scalar(value))


def _extract_explicit_material(text: str) -> str | None:
    low = text.lower()
    hits: list[tuple[int, str]] = []
    for alias, canonical in _MATERIAL_ALIASES.items():
        alias_low = alias.lower()
        if re.search(r"[A-Za-z0-9_]", alias_low):
            for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(alias_low)}(?![A-Za-z0-9_])", low):
                hits.append((match.start(), canonical))
        elif alias in text:
            hits.append((text.index(alias), canonical))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, canonical in hits:
        if canonical not in ordered:
            ordered.append(canonical)
    if any(
        token in low
        for token in (
            "boolean",
            "subtract",
            "subtraction",
            "difference",
            "minus",
            "hole",
            "cut out",
            "cutout",
        )
    ):
        for material in ordered:
            if material != "G4_AIR":
                return material
    return ordered[0]


def _extract_explicit_source_kind(text: str) -> str | None:
    low = text.lower()
    if "isotropic" in low or "\u5404\u5411\u540c\u6027" in text:
        return "isotropic"
    if re.search(r"(?<![A-Za-z0-9_])beam(?![A-Za-z0-9_])", low) or any(
        token in text for token in ("\u675f\u6d41", "\u7c92\u5b50\u675f", "\u51c6\u76f4", "\u5e73\u884c\u675f")
    ):
        return "beam"
    if re.search(r"(?<![A-Za-z0-9_])point source(?![A-Za-z0-9_])", low) or re.search(
        r"(?<![A-Za-z0-9_])point(?![A-Za-z0-9_])", low
    ) or any(token in text for token in ("\u70b9\u6e90", "\u70b9\u72b6\u6e90")):
        return "point"
    if re.search(r"(?<![A-Za-z0-9_])plane(?: source)?(?![A-Za-z0-9_])", low) or "\u9762\u6e90" in text:
        return "plane"
    return None


def _has_unknown_material_marker(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search("(?:\u6750\u6599|material)\\s*[:\uff1a]?\\s*[?\uff1f]+", text)
        or re.search("(?:\u6750\u6599|material)\\s*(?:unknown|tbd|unspecified)\\b", low)
    )


def _has_graph_family_cue(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            "ring",
            "annulus",
            "circular array",
            "grid",
            "array",
            "matrix",
            "stack",
            "layers",
            "shell",
            "concentric",
            "coaxial",
            "nest",
            "inside",
            "contains",
            "boolean",
            "union",
            "subtraction",
            "intersection",
            "\u9635\u5217",
            "\u4e8c\u7ef4\u9635\u5217",
            "\u63a2\u6d4b\u677f",
            "\u7f51\u683c",
            "\u5806\u53e0",
            "\u540c\u5fc3",
            "\u5d4c\u5957",
            "\u5e03\u5c14",
            "\u5e76\u96c6",
            "\u51cf\u6cd5",
            "\u76f8\u4ea4",
        )
    )


def _present_slot_paths(frame: SlotFrame) -> set[str]:
    paths: set[str] = set()
    if frame.geometry.kind:
        paths.add("geometry.kind")
    if frame.geometry.size_triplet_mm:
        paths.add("geometry.size_triplet_mm")
    if frame.geometry.radius_mm is not None:
        paths.add("geometry.radius_mm")
    if frame.geometry.half_length_mm is not None:
        paths.add("geometry.half_length_mm")
    if frame.geometry.radius1_mm is not None:
        paths.add("geometry.radius1_mm")
    if frame.geometry.radius2_mm is not None:
        paths.add("geometry.radius2_mm")
    if frame.geometry.x1_mm is not None:
        paths.add("geometry.x1_mm")
    if frame.geometry.x2_mm is not None:
        paths.add("geometry.x2_mm")
    if frame.geometry.y1_mm is not None:
        paths.add("geometry.y1_mm")
    if frame.geometry.y2_mm is not None:
        paths.add("geometry.y2_mm")
    if frame.geometry.z_mm is not None:
        paths.add("geometry.z_mm")
    if frame.geometry.z_planes_mm:
        paths.add("geometry.z_planes_mm")
    if frame.geometry.radii_mm:
        paths.add("geometry.radii_mm")
    if frame.geometry.trap_x1_mm is not None:
        paths.add("geometry.trap_x1_mm")
    if frame.geometry.trap_x2_mm is not None:
        paths.add("geometry.trap_x2_mm")
    if frame.geometry.trap_x3_mm is not None:
        paths.add("geometry.trap_x3_mm")
    if frame.geometry.trap_x4_mm is not None:
        paths.add("geometry.trap_x4_mm")
    if frame.geometry.trap_y1_mm is not None:
        paths.add("geometry.trap_y1_mm")
    if frame.geometry.trap_y2_mm is not None:
        paths.add("geometry.trap_y2_mm")
    if frame.geometry.trap_z_mm is not None:
        paths.add("geometry.trap_z_mm")
    if frame.geometry.para_x_mm is not None:
        paths.add("geometry.para_x_mm")
    if frame.geometry.para_y_mm is not None:
        paths.add("geometry.para_y_mm")
    if frame.geometry.para_z_mm is not None:
        paths.add("geometry.para_z_mm")
    if frame.geometry.para_alpha_deg is not None:
        paths.add("geometry.para_alpha_deg")
    if frame.geometry.para_theta_deg is not None:
        paths.add("geometry.para_theta_deg")
    if frame.geometry.para_phi_deg is not None:
        paths.add("geometry.para_phi_deg")
    if frame.geometry.torus_major_radius_mm is not None:
        paths.add("geometry.torus_major_radius_mm")
    if frame.geometry.torus_minor_radius_mm is not None:
        paths.add("geometry.torus_minor_radius_mm")
    if frame.geometry.ellipsoid_ax_mm is not None:
        paths.add("geometry.ellipsoid_ax_mm")
    if frame.geometry.ellipsoid_by_mm is not None:
        paths.add("geometry.ellipsoid_by_mm")
    if frame.geometry.ellipsoid_cz_mm is not None:
        paths.add("geometry.ellipsoid_cz_mm")
    if frame.geometry.elltube_ax_mm is not None:
        paths.add("geometry.elltube_ax_mm")
    if frame.geometry.elltube_by_mm is not None:
        paths.add("geometry.elltube_by_mm")
    if frame.geometry.elltube_hz_mm is not None:
        paths.add("geometry.elltube_hz_mm")
    if frame.geometry.polyhedra_sides is not None:
        paths.add("geometry.polyhedra_sides")
    if frame.geometry.tilt_x_deg is not None:
        paths.add("geometry.tilt_x_deg")
    if frame.geometry.tilt_y_deg is not None:
        paths.add("geometry.tilt_y_deg")
    if frame.materials.primary:
        paths.add("materials.primary")
    if frame.source.kind:
        paths.add("source.kind")
    if frame.source.particle:
        paths.add("source.particle")
    if frame.source.energy_mev is not None:
        paths.add("source.energy_mev")
    if frame.source.position_mm:
        paths.add("source.position_mm")
    if frame.source.direction_vec:
        paths.add("source.direction_vec")
    if frame.physics.explicit_list:
        paths.add("physics.explicit_list")
    if frame.physics.recommendation_intent:
        paths.add("physics.recommendation_intent")
    if frame.output.format:
        paths.add("output.format")
    if frame.output.path:
        paths.add("output.path")
    return paths


def _clear_geometry_slots(frame: SlotFrame) -> None:
    frame.geometry.kind = None
    frame.geometry.size_triplet_mm = None
    frame.geometry.radius_mm = None
    frame.geometry.half_length_mm = None
    frame.geometry.radius1_mm = None
    frame.geometry.radius2_mm = None
    frame.geometry.x1_mm = None
    frame.geometry.x2_mm = None
    frame.geometry.y1_mm = None
    frame.geometry.y2_mm = None
    frame.geometry.z_mm = None
    frame.geometry.z_planes_mm = None
    frame.geometry.radii_mm = None
    frame.geometry.trap_x1_mm = None
    frame.geometry.trap_x2_mm = None
    frame.geometry.trap_x3_mm = None
    frame.geometry.trap_x4_mm = None
    frame.geometry.trap_y1_mm = None
    frame.geometry.trap_y2_mm = None
    frame.geometry.trap_z_mm = None
    frame.geometry.para_x_mm = None
    frame.geometry.para_y_mm = None
    frame.geometry.para_z_mm = None
    frame.geometry.para_alpha_deg = None
    frame.geometry.para_theta_deg = None
    frame.geometry.para_phi_deg = None
    frame.geometry.torus_major_radius_mm = None
    frame.geometry.torus_minor_radius_mm = None
    frame.geometry.ellipsoid_ax_mm = None
    frame.geometry.ellipsoid_by_mm = None
    frame.geometry.ellipsoid_cz_mm = None
    frame.geometry.elltube_ax_mm = None
    frame.geometry.elltube_by_mm = None
    frame.geometry.elltube_hz_mm = None
    frame.geometry.polyhedra_sides = None
    frame.geometry.tilt_x_deg = None
    frame.geometry.tilt_y_deg = None


def _clear_target_slot(frame: SlotFrame, target: str) -> None:
    if target == "geometry.kind":
        _clear_geometry_slots(frame)
        return
    if target == "materials.primary":
        frame.materials.primary = None
        return
    if target == "source.kind":
        frame.source.kind = None
        return
    if target == "source.particle":
        frame.source.particle = None
        return
    if target == "source.energy_mev":
        frame.source.energy_mev = None
        return
    if target == "source.position_mm":
        frame.source.position_mm = None
        return
    if target == "source.direction_vec":
        frame.source.direction_vec = None
        return
    if target == "physics.explicit_list":
        frame.physics.explicit_list = None
        return
    if target == "output.format":
        frame.output.format = None
        frame.output.path = None


def _prune_unresolved_targets(frame: SlotFrame, user_text: str) -> list[str]:
    unresolved_targets = infer_unresolved_targets(user_text)
    uncertainty_signal = has_uncertainty_signal(user_text)
    conservative_targets = {
        "geometry.kind",
        "materials.primary",
        "source.kind",
        "source.particle",
        "source.energy_mev",
        "source.position_mm",
        "source.direction_vec",
        "physics.explicit_list",
        "output.format",
    }
    targets_to_check = set(unresolved_targets)
    if uncertainty_signal:
        targets_to_check.update(conservative_targets)
    if not targets_to_check:
        return []
    frame.target_slots = sorted(set(frame.target_slots) | set(unresolved_targets))
    cleared: list[str] = []
    for target in sorted(targets_to_check):
        if has_grounded_payload_for_target(user_text, target):
            continue
        _clear_target_slot(frame, target)
        cleared.append(target)
    if cleared and frame.intent == Intent.SET and uncertainty_signal:
        frame.intent = Intent.QUESTION
    return cleared


def _normalize_inferred_slots(frame: SlotFrame) -> None:
    if frame.geometry.kind is None:
        if frame.geometry.size_triplet_mm:
            frame.geometry.kind = "box"
        elif frame.geometry.radius1_mm is not None or frame.geometry.radius2_mm is not None:
            frame.geometry.kind = "cons"
        elif (
            frame.geometry.x1_mm is not None
            or frame.geometry.x2_mm is not None
            or frame.geometry.y1_mm is not None
            or frame.geometry.y2_mm is not None
            or frame.geometry.z_mm is not None
        ):
            frame.geometry.kind = "trd"
        elif frame.geometry.z_planes_mm or frame.geometry.radii_mm:
            frame.geometry.kind = "polycone"
        elif frame.geometry.polyhedra_sides is not None:
            frame.geometry.kind = "polyhedra"
        elif any(
            value is not None
            for value in (
                frame.geometry.trap_x1_mm,
                frame.geometry.trap_x2_mm,
                frame.geometry.trap_x3_mm,
                frame.geometry.trap_x4_mm,
                frame.geometry.trap_y1_mm,
                frame.geometry.trap_y2_mm,
                frame.geometry.trap_z_mm,
            )
        ):
            frame.geometry.kind = "trap"
        elif any(
            value is not None
            for value in (
                frame.geometry.para_x_mm,
                frame.geometry.para_y_mm,
                frame.geometry.para_z_mm,
                frame.geometry.para_alpha_deg,
                frame.geometry.para_theta_deg,
                frame.geometry.para_phi_deg,
            )
        ):
            frame.geometry.kind = "para"
        elif frame.geometry.torus_major_radius_mm is not None or frame.geometry.torus_minor_radius_mm is not None:
            frame.geometry.kind = "torus"
        elif any(
            value is not None
            for value in (
                frame.geometry.ellipsoid_ax_mm,
                frame.geometry.ellipsoid_by_mm,
                frame.geometry.ellipsoid_cz_mm,
            )
        ):
            frame.geometry.kind = "ellipsoid"
        elif any(
            value is not None
            for value in (
                frame.geometry.elltube_ax_mm,
                frame.geometry.elltube_by_mm,
                frame.geometry.elltube_hz_mm,
            )
        ):
            frame.geometry.kind = "elltube"
        elif frame.geometry.tilt_x_deg is not None or frame.geometry.tilt_y_deg is not None:
            frame.geometry.kind = "cuttubs"
        elif frame.geometry.radius_mm is not None or frame.geometry.half_length_mm is not None:
            frame.geometry.kind = "cylinder"


def _geometry_box_from_phrase(text: str) -> list[float] | None:
    low = text.lower().replace("\u8133", "x")
    m = re.search(
        "([-+]?\\d*\\.?\\d+)\\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)\\s*(?:x|by)\\s*"
        "([-+]?\\d*\\.?\\d+)\\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)\\s*(?:x|by)\\s*"
        "([-+]?\\d*\\.?\\d+)\\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)",
        low,
    )
    if m:
        return [
            _to_mm(float(m.group(1)), m.group(2)),
            _to_mm(float(m.group(3)), m.group(4)),
            _to_mm(float(m.group(5)), m.group(6)),
        ]

    side_patterns = [
        r"(?:side(?:\s+length)?|edge(?:\s+length)?)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "([-+]?\\d*\\.?\\d+)\\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)\\s*\u89c1\u65b9",
        "\u8fb9\u957f\\s*[:=]?\\s*([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
        "([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)\\s*(?:\u7684)?(?:\u7acb\u65b9\u4f53|\u7acb\u65b9\u5757)",
    ]
    for pattern in side_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        side = _coerce_length_mm(f"{m.group(1)} {m.group(2)}")
        if side is not None:
            return [side, side, side]

    side_after_patterns = [
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m)\s*(?:each|per)\s*side",
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m)\s*on\s*each\s*side",
        r"(?:cube|box|cuboid)\s*(?:with\s*)?([-+]?\d*\.?\d+)\s*(mm|cm|m)\s*(?:side|edge)",
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m)\s*(?:cube|box|cuboid)\b",
        r"([-+]?\d*\.?\d+)\s*mm\^?3\s*(?:each|per)\s*side",
    ]
    for pattern in side_after_patterns:
        m = re.search(pattern, low, flags=re.IGNORECASE)
        if not m:
            continue
        side = _coerce_length_mm(f"{m.group(1)} mm" if "mm\\^?3" in pattern else f"{m.group(1)} {m.group(2)}")
        if side is not None:
            return [side, side, side]

    if re.search(r"\b(one|1)\s*(meter|m)\s*(cube|box|cuboid)\b", low):
        side = _coerce_length_mm("1 m")
        if side is not None:
            return [side, side, side]
    return None

def _geometry_cylinder_from_phrase(text: str) -> tuple[float | None, float | None]:
    low = text.lower()

    def _match_length(patterns: list[str], *, halve: bool = False) -> float | None:
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            value = _coerce_length_mm(f"{m.group(1)} {m.group(2)}")
            if value is None:
                continue
            return value / 2.0 if halve else value
        return None

    radius = _match_length(
        [
            r"radius\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            "\u534a\u5f84\\s*[:=]?\\s*([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
            r"r\s*[:=]\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        ]
    )
    if radius is None:
        diameter = _match_length(
            [
                r"diameter\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
                "\u76f4\u5f84\\s*[:=]?\\s*([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
            ]
        )
        if diameter is not None:
            radius = diameter / 2.0

    half_length = _match_length(
        [
            r"half[-\s]*length\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            r"half[-\s]*z\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            r"hz\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            "\u534a(?:\u957f|\u957f\u5ea6)\\s*[:=]?\\s*([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
        ]
    )
    if half_length is None:
        half_length = _match_length(
            [
                r"(?:height|length)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
                "\u9ad8(?:\u5ea6)?\\s*[:=]?\\s*([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
                "\u957f(?:\u5ea6)?\\s*[:=]?\\s*([-+]?\\d*\\.?\\d+)\\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
            ],
            halve=True,
        )

    if (radius is None or half_length is None) and "radius" in low and "half-length" in low:
        pair = re.search(
            r"radius\s*([-+]?\d*\.?\d+)\s*(mm|cm|m).*?half[-\s]*length\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            low,
        )
        if pair:
            radius = radius if radius is not None else _to_mm(float(pair.group(1)), pair.group(2))
            half_length = half_length if half_length is not None else _to_mm(float(pair.group(3)), pair.group(4))

    return radius, half_length

def _source_position_from_phrase(text: str) -> list[float] | None:
    patterns = [
        r"(?:position(?:ed)?|located)\s*(?:at|=|:)?\s*(\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)\s*(?:mm|cm|m)?)",
        r"(?:source\s+at|at)\s*(\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)\s*(?:mm|cm|m)?)",
        r"(?:from)\s*(\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)\s*(?:mm|cm|m)?)",
        r"(?:position|located)\s*(?:at|=|:)?\s*([-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*(?:mm|cm|m))",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        vec = _coerce_vec3(m.group(1), metric=True)
        if vec is not None:
            return vec
    return None


def _source_relative_to_center_phrase(text: str) -> tuple[list[float], list[float]] | None:
    patterns = [
        r"(?:from\s+)?(?:the\s+)?center(?:\s+point)?\s*(?:by|at|from)?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)\s*(?:outside|away)(?:\s+along\s*([+-][xyz]))?",
        r"(?:from\s+)?(?:the\s+)?center(?:\s+point)?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)\s*(?:outside|away)(?:\s+along\s*([+-][xyz]))?",
        r"(?:中心(?:点)?)\s*([-+]?\d*\.?\d+)\s*(毫米|厘米|米|mm|cm|m)\s*外(?:\s*(?:沿|朝)\s*([+-][xyz]))?",
        r"(?:距(?:离)?中心(?:点)?)\s*([-+]?\d*\.?\d+)\s*(毫米|厘米|米|mm|cm|m)(?:\s*外)?(?:\s*(?:沿|朝)\s*([+-][xyz]))?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        distance_mm = _to_mm(float(match.group(1)), match.group(2))
        axis = (match.group(3) or "-z").lower()
        axis_map = {
            "+x": ([distance_mm, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            "-x": ([-distance_mm, 0.0, 0.0], [1.0, 0.0, 0.0]),
            "+y": ([0.0, distance_mm, 0.0], [0.0, -1.0, 0.0]),
            "-y": ([0.0, -distance_mm, 0.0], [0.0, 1.0, 0.0]),
            "+z": ([0.0, 0.0, distance_mm], [0.0, 0.0, -1.0]),
            "-z": ([0.0, 0.0, -distance_mm], [0.0, 0.0, 1.0]),
        }
        return axis_map.get(axis, axis_map["-z"])
    return None


def _source_direction_from_phrase(text: str) -> list[float] | None:
    patterns = [
        r"(?:direction|pointing)\s*(?:=|:)?\s*(\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\))",
        r"(?:direction|pointing)\s*(?:=|:)?\s*([+-][xyz])",
        r"(?:toward|towards)\s*(\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\))",
        r"(?:toward|towards)\s*([+-][xyz])",
        r"along\s*(?:the\s*)?([+-][xyz])(?:\s+direction)?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        vec = _coerce_vec3(m.group(1), metric=False)
        if vec is not None:
            return vec
    return None


def _apply_clause(frame: SlotFrame, key: str, raw_value: str) -> None:
    k = key.strip().lower()
    v = raw_value.strip()
    if not k:
        return
    if k in {"geometry.kind", "geometry.type", "geometry.shape"}:
        frame.geometry.kind = _canonical_geometry_kind(v) or frame.geometry.kind
        return
    if k in {"geometry.size", "geometry.size_triplet_mm", "geometry.box", "geometry.dimensions"}:
        triplet = _coerce_triplet_mm(v)
        if triplet is not None:
            frame.geometry.size_triplet_mm = triplet
        if frame.geometry.kind is None:
            frame.geometry.kind = "box"
        return
    if k in {"geometry.radius", "geometry.radius_mm"}:
        radius = _coerce_length_mm(v)
        if radius is not None:
            frame.geometry.radius_mm = radius
        if frame.geometry.kind is None:
            frame.geometry.kind = "cylinder"
        return
    if k in {"geometry.radius1", "geometry.radius1_mm", "geometry.rmax1"}:
        radius1 = _coerce_length_mm(v)
        if radius1 is not None:
            frame.geometry.radius1_mm = radius1
        if frame.geometry.kind is None:
            frame.geometry.kind = "cons"
        return
    if k in {"geometry.radius2", "geometry.radius2_mm", "geometry.rmax2"}:
        radius2 = _coerce_length_mm(v)
        if radius2 is not None:
            frame.geometry.radius2_mm = radius2
        if frame.geometry.kind is None:
            frame.geometry.kind = "cons"
        return
    if k in {"geometry.half_length", "geometry.half_length_mm", "geometry.height", "geometry.height_mm", "geometry.hz"}:
        half_length = _coerce_length_mm(v)
        if half_length is not None:
            frame.geometry.half_length_mm = half_length
        if frame.geometry.kind is None:
            frame.geometry.kind = "cylinder"
        return
    if k in {"geometry.x1", "geometry.x1_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.x1_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trd"
        return
    if k in {"geometry.x2", "geometry.x2_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.x2_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trd"
        return
    if k in {"geometry.y1", "geometry.y1_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.y1_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trd"
        return
    if k in {"geometry.y2", "geometry.y2_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.y2_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trd"
        return
    if k in {"geometry.z", "geometry.z_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.z_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trd"
        return
    if k in {"geometry.z_planes", "geometry.z_planes_mm"}:
        vec = _coerce_vec3(v, metric=True)
        if vec is not None:
            frame.geometry.z_planes_mm = vec
        if frame.geometry.kind is None:
            frame.geometry.kind = "polycone"
        return
    if k in {"geometry.radii", "geometry.radii_mm", "geometry.rmax_list"}:
        vec = _coerce_vec3(v, metric=True)
        if vec is not None:
            frame.geometry.radii_mm = vec
        if frame.geometry.kind is None:
            frame.geometry.kind = "polycone"
        return
    if k in {"geometry.polyhedra_sides", "geometry.polyhedra_nsides", "geometry.nsides"}:
        value = _coerce_float(v)
        if value is not None:
            frame.geometry.polyhedra_sides = int(value)
            frame.geometry.kind = "polyhedra"
        return
    if k in {"geometry.trap_x1", "geometry.trap_x1_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_x1_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.trap_x2", "geometry.trap_x2_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_x2_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.trap_x3", "geometry.trap_x3_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_x3_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.trap_x4", "geometry.trap_x4_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_x4_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.trap_y1", "geometry.trap_y1_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_y1_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.trap_y2", "geometry.trap_y2_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_y2_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.trap_z", "geometry.trap_z_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.trap_z_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "trap"
        return
    if k in {"geometry.para_x", "geometry.para_x_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.para_x_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "para"
        return
    if k in {"geometry.para_y", "geometry.para_y_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.para_y_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "para"
        return
    if k in {"geometry.para_z", "geometry.para_z_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.para_z_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "para"
        return
    if k in {"geometry.para_alpha", "geometry.para_alpha_deg"}:
        value = _coerce_float(v)
        if value is not None:
            frame.geometry.para_alpha_deg = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "para"
        return
    if k in {"geometry.para_theta", "geometry.para_theta_deg"}:
        value = _coerce_float(v)
        if value is not None:
            frame.geometry.para_theta_deg = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "para"
        return
    if k in {"geometry.para_phi", "geometry.para_phi_deg"}:
        value = _coerce_float(v)
        if value is not None:
            frame.geometry.para_phi_deg = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "para"
        return
    if k in {"geometry.torus_major_radius", "geometry.torus_major_radius_mm", "geometry.torus_rtor"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.torus_major_radius_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "torus"
        return
    if k in {"geometry.torus_minor_radius", "geometry.torus_minor_radius_mm", "geometry.torus_rmax"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.torus_minor_radius_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "torus"
        return
    if k in {"geometry.ellipsoid_ax", "geometry.ellipsoid_ax_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.ellipsoid_ax_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "ellipsoid"
        return
    if k in {"geometry.ellipsoid_by", "geometry.ellipsoid_by_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.ellipsoid_by_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "ellipsoid"
        return
    if k in {"geometry.ellipsoid_cz", "geometry.ellipsoid_cz_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.ellipsoid_cz_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "ellipsoid"
        return
    if k in {"geometry.elltube_ax", "geometry.elltube_ax_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.elltube_ax_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "elltube"
        return
    if k in {"geometry.elltube_by", "geometry.elltube_by_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.elltube_by_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "elltube"
        return
    if k in {"geometry.elltube_hz", "geometry.elltube_hz_mm"}:
        value = _coerce_length_mm(v)
        if value is not None:
            frame.geometry.elltube_hz_mm = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "elltube"
        return
    if k in {"geometry.tilt_x", "geometry.tilt_x_deg"}:
        value = _coerce_float(v)
        if value is not None:
            frame.geometry.tilt_x_deg = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "cuttubs"
        return
    if k in {"geometry.tilt_y", "geometry.tilt_y_deg"}:
        value = _coerce_float(v)
        if value is not None:
            frame.geometry.tilt_y_deg = value
        if frame.geometry.kind is None:
            frame.geometry.kind = "cuttubs"
        return
    if k in {"materials.primary", "materials.material", "materials.target", "materials.copper"}:
        mat = _canonical_material(v if k != "materials.copper" else "copper")
        if mat:
            frame.materials.primary = mat
        return
    if k in {"source.kind", "source.type", "source.source_type"}:
        sk = _canonical_source_kind(v)
        if sk:
            frame.source.kind = sk
            return
        particle = _canonical_particle(v)
        if particle:
            frame.source.particle = particle
        return
    if k in {"source.particle", "source.particle_type"}:
        particle = _canonical_particle(v)
        if particle:
            frame.source.particle = particle
        return
    if k in {"source.energy", "source.energy_mev"}:
        energy = _coerce_float(v)
        if energy is not None:
            frame.source.energy_mev = energy
        return
    if k in {"source.position", "source.position_mm"}:
        vec = _coerce_vec3(v, metric=True)
        if vec is not None:
            frame.source.position_mm = vec
        return
    if k in {"source.direction", "source.direction_vec"}:
        vec = _coerce_vec3(v, metric=False)
        if vec is not None:
            frame.source.direction_vec = vec
        return
    if k in {"physics.explicit_list", "physics.list", "physics.physics_list"}:
        explicit = _clean_scalar(v)
        if explicit:
            frame.physics.explicit_list = explicit
        return
    if k in {"physics.recommendation_intent", "physics.intent"}:
        recommendation = _clean_scalar(v)
        if recommendation:
            frame.physics.recommendation_intent = recommendation
        return
    if k in {"output.format", "output.type"}:
        fmt = _canonical_output_format(v)
        if fmt:
            frame.output.format = fmt
        return
    if k == "output.path":
        path = _clean_scalar(v)
        if path:
            frame.output.path = path


def _backfill_from_normalized_text(frame: SlotFrame) -> None:
    text = frame.normalized_text or ""
    for clause in text.split(";"):
        if ":" not in clause:
            continue
        key, raw_value = clause.split(":", 1)
        _apply_clause(frame, key, raw_value)


def _backfill_from_user_text(frame: SlotFrame, user_text: str) -> None:
    text = user_text or ""
    low = text.lower()

    if frame.geometry.kind is None:
        if any(token in low for token in ("box", "cube", "cuboid")) or any(token in text for token in ("\u7acb\u65b9\u4f53", "\u7acb\u65b9\u5757")):
            frame.geometry.kind = "box"
        elif any(token in low for token in ("cylinder", "cylindrical", "tubs")) or "\u5706\u67f1" in text:
            frame.geometry.kind = "cylinder"
        elif re.search(r"(?<![a-z0-9_])orb(?![a-z0-9_])", low):
            frame.geometry.kind = "orb"
        elif "sphere" in low or "\u7403" in text:
            frame.geometry.kind = "sphere"
        elif any(token in low for token in ("cons", "cone", "frustum")):
            frame.geometry.kind = "cons"
        elif any(token in low for token in ("trd", "trapezoid", "trapezoidal")):
            frame.geometry.kind = "trd"
        elif "trap" in low:
            frame.geometry.kind = "trap"
        elif any(token in low for token in ("para", "parallelepiped", "skewed box")):
            frame.geometry.kind = "para"
        elif any(token in low for token in ("torus", "donut", "ring tube")):
            frame.geometry.kind = "torus"
        elif "ellipsoid" in low:
            frame.geometry.kind = "ellipsoid"
        elif any(token in low for token in ("elltube", "elliptical tube", "ellipse tube", "elliptic tube")):
            frame.geometry.kind = "elltube"
        elif "polycone" in low:
            frame.geometry.kind = "polycone"
        elif any(token in low for token in ("polyhedra", "polyhedron")):
            frame.geometry.kind = "polyhedra"
        elif any(token in low for token in ("cuttubs", "cut tubs")):
            frame.geometry.kind = "cuttubs"

    if frame.geometry.size_triplet_mm is None:
        triplet = _coerce_triplet_mm(text)
        if triplet is None:
            triplet = _geometry_box_from_phrase(text)
        if triplet is not None:
            frame.geometry.size_triplet_mm = triplet
            frame.geometry.kind = frame.geometry.kind or "box"

    if frame.geometry.radius_mm is None or frame.geometry.half_length_mm is None:
        radius, half_length = _geometry_cylinder_from_phrase(text)
        if frame.geometry.radius_mm is None and radius is not None:
            frame.geometry.radius_mm = radius
            frame.geometry.kind = frame.geometry.kind or "cylinder"
        if frame.geometry.half_length_mm is None and half_length is not None:
            frame.geometry.half_length_mm = half_length
            frame.geometry.kind = frame.geometry.kind or "cylinder"

    if frame.geometry.radius1_mm is None:
        m = re.search(r"(?:rmax1|radius1|top radius)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)", low)
        if m:
            frame.geometry.radius1_mm = _to_mm(float(m.group(1)), m.group(2))
            frame.geometry.kind = frame.geometry.kind or "cons"
    if frame.geometry.radius2_mm is None:
        m = re.search(r"(?:rmax2|radius2|bottom radius)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)", low)
        if m:
            frame.geometry.radius2_mm = _to_mm(float(m.group(1)), m.group(2))
            frame.geometry.kind = frame.geometry.kind or "cons"

    trd_patterns = {
        "x1_mm": r"x1\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "x2_mm": r"x2\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "y1_mm": r"y1\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "y2_mm": r"y2\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "z_mm": r"(?:z|depth)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
    }
    for field_name, pattern in trd_patterns.items():
        if getattr(frame.geometry, field_name) is not None:
            continue
        m = re.search(pattern, low)
        if not m:
            continue
        setattr(frame.geometry, field_name, _to_mm(float(m.group(1)), m.group(2)))
        frame.geometry.kind = frame.geometry.kind or "trd"

    named_patterns = {
        "trap_x1_mm": r"trap[_\s-]*x1\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "trap_x2_mm": r"trap[_\s-]*x2\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "trap_x3_mm": r"trap[_\s-]*x3\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "trap_x4_mm": r"trap[_\s-]*x4\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "trap_y1_mm": r"trap[_\s-]*y1\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "trap_y2_mm": r"trap[_\s-]*y2\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "trap_z_mm": r"trap[_\s-]*z\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "para_x_mm": r"para[_\s-]*x\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "para_y_mm": r"para[_\s-]*y\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "para_z_mm": r"para[_\s-]*z\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "torus_major_radius_mm": r"(?:torus[_\s-]*)?(?:major radius|rtor)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "torus_minor_radius_mm": r"(?:torus[_\s-]*)?(?:tube radius|minor radius|rmax)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "ellipsoid_ax_mm": r"(?:ellipsoid[_\s-]*)?ax\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "ellipsoid_by_mm": r"(?:ellipsoid[_\s-]*)?by\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "ellipsoid_cz_mm": r"(?:ellipsoid[_\s-]*)?cz\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "elltube_ax_mm": r"(?:elltube|elliptical tube|ellipse tube|elliptic tube)[_\s-]*ax\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "elltube_by_mm": r"(?:elltube|elliptical tube|ellipse tube|elliptic tube)[_\s-]*by\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        "elltube_hz_mm": r"(?:elltube|elliptical tube|ellipse tube|elliptic tube)[_\s-]*hz\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
    }
    for field_name, pattern in named_patterns.items():
        if getattr(frame.geometry, field_name) is not None:
            continue
        m = re.search(pattern, low)
        if not m:
            continue
        setattr(frame.geometry, field_name, _to_mm(float(m.group(1)), m.group(2)))

    for field_name, pattern in {
        "para_alpha_deg": r"para[_\s-]*alpha\s*[:=]?\s*([-+]?\d*\.?\d+)",
        "para_theta_deg": r"para[_\s-]*theta\s*[:=]?\s*([-+]?\d*\.?\d+)",
        "para_phi_deg": r"para[_\s-]*phi\s*[:=]?\s*([-+]?\d*\.?\d+)",
    }.items():
        if getattr(frame.geometry, field_name) is not None:
            continue
        m = re.search(pattern, low)
        if m:
            setattr(frame.geometry, field_name, float(m.group(1)))

    if frame.geometry.polyhedra_sides is None:
        m = re.search(r"(?:polyhedra|polyhedron)[_\s-]*(?:nsides|sides|n)\s*[:=]?\s*(\d+)", low)
        if m:
            frame.geometry.polyhedra_sides = int(m.group(1))
            frame.geometry.kind = frame.geometry.kind or "polyhedra"

    if frame.geometry.tilt_x_deg is None:
        m = re.search(r"(?:tilt_x|tilt x)\s*[:=]?\s*([-+]?\d*\.?\d+)", low)
        if m:
            frame.geometry.tilt_x_deg = float(m.group(1))
            frame.geometry.kind = frame.geometry.kind or "cuttubs"
    if frame.geometry.tilt_y_deg is None:
        m = re.search(r"(?:tilt_y|tilt y)\s*[:=]?\s*([-+]?\d*\.?\d+)", low)
        if m:
            frame.geometry.tilt_y_deg = float(m.group(1))
            frame.geometry.kind = frame.geometry.kind or "cuttubs"

    explicit_material = _extract_explicit_material(text)
    if explicit_material is not None:
        frame.materials.primary = explicit_material
    elif _has_unknown_material_marker(text):
        frame.materials.primary = None

    explicit_source_kind = _extract_explicit_source_kind(text)
    if explicit_source_kind is not None:
        frame.source.kind = explicit_source_kind

    if _has_graph_family_cue(text):
        frame.geometry.kind = None

    if frame.source.particle is None:
        for token, canonical in _PARTICLE_ALIASES.items():
            if token in low:
                frame.source.particle = canonical
                break

    if frame.source.energy_mev is None:
        m_energy = re.search(r"([-+]?\d*\.?\d+)\s*(mev|gev|kev)", low, flags=re.IGNORECASE)
        if m_energy:
            frame.source.energy_mev = _coerce_float(f"{m_energy.group(1)} {m_energy.group(2)}")

    if frame.source.position_mm is None and any(token in low for token in ("center", "centre", "涓績", "origin", "鍘熺偣")):
        frame.source.position_mm = [0.0, 0.0, 0.0]

    if frame.source.position_mm is None:
        position = _source_position_from_phrase(text)
        if position is not None:
            frame.source.position_mm = position

    if frame.source.position_mm is None or frame.source.direction_vec is None:
        relative = _source_relative_to_center_phrase(text)
        if relative is not None:
            if frame.source.position_mm is None:
                frame.source.position_mm = relative[0]
            if frame.source.direction_vec is None:
                frame.source.direction_vec = relative[1]

    if frame.source.direction_vec is None:
        direction = _source_direction_from_phrase(text)
        if direction is not None:
            frame.source.direction_vec = direction

    _normalize_inferred_slots(frame)


def _user_text_supports_source_direction(user_text: str) -> bool:
    text = user_text or ""
    if _source_direction_from_phrase(text) is not None:
        return True
    if _source_relative_to_center_phrase(text) is not None:
        return True
    return False


def _user_text_supports_source_energy(user_text: str) -> bool:
    text = (user_text or "").lower()
    return re.search(r"([-+]?\d*\.?\d+)\s*(mev|gev|kev)", text, flags=re.IGNORECASE) is not None


def _strip_unsupported_source_direction(frame: SlotFrame, user_text: str) -> list[str]:
    cleared: list[str] = []
    if frame.source.kind not in {"beam", "plane"}:
        return cleared
    if frame.source.direction_vec is None:
        return cleared
    if _user_text_supports_source_direction(user_text):
        return cleared
    frame.source.direction_vec = None
    cleared.append("source.direction_vec")
    return cleared


def _strip_unsupported_source_energy(frame: SlotFrame, user_text: str) -> list[str]:
    cleared: list[str] = []
    if frame.source.energy_mev is None:
        return cleared
    if _user_text_supports_source_energy(user_text):
        return cleared
    frame.source.energy_mev = None
    cleared.append("source.energy_mev")
    return cleared


def _coerce_slot_payload(payload: dict[str, Any]) -> tuple[SlotFrame, dict[str, Any]]:
    errors: list[str] = []
    frame = SlotFrame()
    frame.intent = _intent_from_any(payload.get("intent"))

    confidence = payload.get("confidence", 0.75)
    if not isinstance(confidence, (int, float)):
        errors.append("confidence_not_number")
        confidence = 0.75
    frame.confidence = max(0.0, min(1.0, float(confidence)))
    frame.normalized_text = str(payload.get("normalized_text", "") or "").strip()

    target_slots_raw = payload.get("target_slots", [])
    if isinstance(target_slots_raw, list):
        frame.target_slots = [str(x).strip() for x in target_slots_raw if str(x).strip()]
    elif target_slots_raw is not None:
        errors.append("target_slots_not_list")

    slots = payload.get("slots", {})
    if not isinstance(slots, dict):
        errors.append("slots_not_object")
        slots = {}

    geometry = slots.get("geometry", {})
    if isinstance(geometry, dict):
        frame.geometry.kind = _canonical_geometry_kind(geometry.get("kind"))
        frame.geometry.size_triplet_mm = _coerce_triplet_mm(geometry.get("size_triplet_mm"))
        frame.geometry.radius_mm = _coerce_length_mm(geometry.get("radius_mm"))
        frame.geometry.half_length_mm = _coerce_length_mm(geometry.get("half_length_mm"))
        frame.geometry.radius1_mm = _coerce_length_mm(geometry.get("radius1_mm"))
        frame.geometry.radius2_mm = _coerce_length_mm(geometry.get("radius2_mm"))
        frame.geometry.x1_mm = _coerce_length_mm(geometry.get("x1_mm"))
        frame.geometry.x2_mm = _coerce_length_mm(geometry.get("x2_mm"))
        frame.geometry.y1_mm = _coerce_length_mm(geometry.get("y1_mm"))
        frame.geometry.y2_mm = _coerce_length_mm(geometry.get("y2_mm"))
        frame.geometry.z_mm = _coerce_length_mm(geometry.get("z_mm"))
        frame.geometry.z_planes_mm = _coerce_vec3(geometry.get("z_planes_mm"), metric=True)
        frame.geometry.radii_mm = _coerce_vec3(geometry.get("radii_mm"), metric=True)
        frame.geometry.trap_x1_mm = _coerce_length_mm(geometry.get("trap_x1_mm"))
        frame.geometry.trap_x2_mm = _coerce_length_mm(geometry.get("trap_x2_mm"))
        frame.geometry.trap_x3_mm = _coerce_length_mm(geometry.get("trap_x3_mm"))
        frame.geometry.trap_x4_mm = _coerce_length_mm(geometry.get("trap_x4_mm"))
        frame.geometry.trap_y1_mm = _coerce_length_mm(geometry.get("trap_y1_mm"))
        frame.geometry.trap_y2_mm = _coerce_length_mm(geometry.get("trap_y2_mm"))
        frame.geometry.trap_z_mm = _coerce_length_mm(geometry.get("trap_z_mm"))
        frame.geometry.para_x_mm = _coerce_length_mm(geometry.get("para_x_mm"))
        frame.geometry.para_y_mm = _coerce_length_mm(geometry.get("para_y_mm"))
        frame.geometry.para_z_mm = _coerce_length_mm(geometry.get("para_z_mm"))
        frame.geometry.para_alpha_deg = _coerce_float(geometry.get("para_alpha_deg"))
        frame.geometry.para_theta_deg = _coerce_float(geometry.get("para_theta_deg"))
        frame.geometry.para_phi_deg = _coerce_float(geometry.get("para_phi_deg"))
        frame.geometry.torus_major_radius_mm = _coerce_length_mm(geometry.get("torus_major_radius_mm"))
        frame.geometry.torus_minor_radius_mm = _coerce_length_mm(geometry.get("torus_minor_radius_mm"))
        frame.geometry.ellipsoid_ax_mm = _coerce_length_mm(geometry.get("ellipsoid_ax_mm"))
        frame.geometry.ellipsoid_by_mm = _coerce_length_mm(geometry.get("ellipsoid_by_mm"))
        frame.geometry.ellipsoid_cz_mm = _coerce_length_mm(geometry.get("ellipsoid_cz_mm"))
        frame.geometry.elltube_ax_mm = _coerce_length_mm(geometry.get("elltube_ax_mm"))
        frame.geometry.elltube_by_mm = _coerce_length_mm(geometry.get("elltube_by_mm"))
        frame.geometry.elltube_hz_mm = _coerce_length_mm(geometry.get("elltube_hz_mm"))
        polyhedra_sides = _coerce_float(geometry.get("polyhedra_sides"))
        frame.geometry.polyhedra_sides = int(polyhedra_sides) if polyhedra_sides is not None else None
        frame.geometry.tilt_x_deg = _coerce_float(geometry.get("tilt_x_deg"))
        frame.geometry.tilt_y_deg = _coerce_float(geometry.get("tilt_y_deg"))
        if frame.geometry.half_length_mm is None:
            frame.geometry.half_length_mm = _coerce_length_mm(geometry.get("height_mm"))
    elif "geometry" in slots:
        errors.append("geometry_not_object")

    materials = slots.get("materials", {})
    if isinstance(materials, dict):
        frame.materials.primary = _canonical_material(materials.get("primary"))
    elif "materials" in slots:
        errors.append("materials_not_object")

    source = slots.get("source", {})
    if isinstance(source, dict):
        frame.source.kind = _canonical_source_kind(source.get("kind"))
        frame.source.particle = _canonical_particle(source.get("particle"))
        frame.source.energy_mev = _coerce_float(source.get("energy_mev"))
        frame.source.position_mm = _coerce_vec3(source.get("position_mm"), metric=True)
        frame.source.direction_vec = _coerce_vec3(source.get("direction_vec"), metric=False)
    elif "source" in slots:
        errors.append("source_not_object")

    physics = slots.get("physics", {})
    if isinstance(physics, dict):
        frame.physics.explicit_list = _clean_scalar(physics.get("explicit_list"))
        frame.physics.recommendation_intent = _clean_scalar(physics.get("recommendation_intent"))
    elif "physics" in slots:
        errors.append("physics_not_object")

    output = slots.get("output", {})
    if isinstance(output, dict):
        frame.output.format = _canonical_output_format(output.get("format"))
        frame.output.path = _clean_scalar(output.get("path"))
    elif "output" in slots:
        errors.append("output_not_object")

    candidates = payload.get("candidates", {})
    if isinstance(candidates, dict):
        _apply_controlled_candidates(frame, candidates, errors)
    elif "candidates" in payload:
        errors.append("candidates_not_object")

    meta = {
        "confidence": frame.confidence,
        "normalized_text": frame.normalized_text,
        "schema_errors": errors,
    }
    return frame, meta


def _apply_controlled_candidates(frame: SlotFrame, candidates: dict[str, Any], errors: list[str]) -> None:
    geometry = candidates.get("geometry", {})
    if isinstance(geometry, dict):
        kind_candidate = _canonical_geometry_kind(geometry.get("kind_candidate"))
        side_length = _coerce_length_mm(geometry.get("side_length_mm"))
        radius_mm = _coerce_length_mm(geometry.get("radius_mm"))
        diameter_mm = _coerce_length_mm(geometry.get("diameter_mm"))
        half_length_mm = _coerce_length_mm(geometry.get("half_length_mm"))
        full_length_mm = _coerce_length_mm(geometry.get("full_length_mm"))
        thickness_mm = _coerce_length_mm(geometry.get("thickness_mm"))
        plate_size_xy_mm = _coerce_pair_mm(geometry.get("plate_size_xy_mm"))
        if frame.geometry.kind is None and kind_candidate is not None:
            frame.geometry.kind = kind_candidate
            frame.notes.append(f"candidate.geometry.kind:{kind_candidate}")
        if (
            frame.geometry.size_triplet_mm is None
            and frame.geometry.kind == "box"
            and side_length is not None
        ):
            frame.geometry.size_triplet_mm = [side_length, side_length, side_length]
            frame.notes.append(f"candidate.geometry.side_length_mm:{side_length}")
        if frame.geometry.kind == "box" and frame.geometry.size_triplet_mm is None:
            if thickness_mm is not None:
                frame.geometry.size_triplet_mm = [10.0, 10.0, thickness_mm]
                frame.notes.append(f"candidate.geometry.thickness_mm:{thickness_mm}")
            elif isinstance(plate_size_xy_mm, list) and len(plate_size_xy_mm) >= 2:
                frame.geometry.size_triplet_mm = [plate_size_xy_mm[0], plate_size_xy_mm[1], 1.0]
                frame.notes.append(
                    f"candidate.geometry.plate_size_xy_mm:{plate_size_xy_mm[0]}:{plate_size_xy_mm[1]}"
                )
        if frame.geometry.kind == "cylinder":
            resolved_radius = radius_mm
            if resolved_radius is None and diameter_mm is not None:
                resolved_radius = diameter_mm / 2.0
                frame.notes.append(f"candidate.geometry.diameter_mm:{diameter_mm}")
            if frame.geometry.radius_mm is None and resolved_radius is not None:
                frame.geometry.radius_mm = resolved_radius
                frame.notes.append(f"candidate.geometry.radius_mm:{resolved_radius}")
            resolved_half_length = half_length_mm
            if resolved_half_length is None and full_length_mm is not None:
                resolved_half_length = full_length_mm / 2.0
                frame.notes.append(
                    f"candidate.geometry.full_length_mm:{full_length_mm}"
                )
            if frame.geometry.half_length_mm is None and resolved_half_length is not None:
                frame.geometry.half_length_mm = resolved_half_length
                frame.notes.append(
                    f"candidate.geometry.half_length_mm:{resolved_half_length}"
                )
        if frame.geometry.kind in {"sphere", "orb"} and frame.geometry.radius_mm is None:
            resolved_radius = radius_mm
            if resolved_radius is None and diameter_mm is not None:
                resolved_radius = diameter_mm / 2.0
                frame.notes.append(f"candidate.geometry.diameter_mm:{diameter_mm}")
            if resolved_radius is not None:
                frame.geometry.radius_mm = resolved_radius
                frame.notes.append(f"candidate.geometry.radius_mm:{resolved_radius}")
    elif "geometry" in candidates:
        errors.append("candidate_geometry_not_object")

    source = candidates.get("source", {})
    if isinstance(source, dict):
        relation = _clean_scalar(source.get("relation"))
        offset_mm = _coerce_length_mm(source.get("offset_mm"))
        axis = (_clean_scalar(source.get("axis")) or "").lower()
        direction_mode = (_clean_scalar(source.get("direction_mode")) or "toward_target_center").lower()
        direction_relation = (_clean_scalar(source.get("direction_relation")) or "").lower()
        if relation in {"outside_target_center", "in_front_of_target", "upstream_of_target"} and offset_mm is not None and axis in _AXIS_VECTORS:
            axis_position, axis_toward_center = _AXIS_VECTORS[axis]
            if frame.source.position_mm is None:
                frame.source.position_mm = [component * offset_mm for component in axis_position]
                frame.notes.append(f"candidate.source.relative_position:{offset_mm}:{axis}")
            if frame.source.direction_vec is None:
                if direction_mode == "along_axis":
                    frame.source.direction_vec = list(axis_position)
                elif direction_mode == "against_axis":
                    frame.source.direction_vec = [-component for component in axis_position]
                else:
                    frame.source.direction_vec = list(axis_toward_center)
                frame.notes.append(f"candidate.source.direction_mode:{direction_mode}")
        if frame.source.direction_vec is None and direction_relation:
            if direction_relation in {"normal_to_target_face", "toward_target_face", "toward_target_surface_normal"} and axis in _AXIS_VECTORS:
                _, axis_toward_center = _AXIS_VECTORS[axis]
                frame.source.direction_vec = list(axis_toward_center)
                frame.notes.append(f"candidate.source.direction_relation:{direction_relation}")
            elif direction_relation == "toward_target_center" and frame.source.position_mm is not None:
                normalized = _normalize_vec3([-component for component in frame.source.position_mm])
                if normalized is not None:
                    frame.source.direction_vec = normalized
                    frame.notes.append(f"candidate.source.direction_relation:{direction_relation}")
    elif "source" in candidates:
        errors.append("candidate_source_not_object")


def parse_slot_payload(payload: dict[str, Any]) -> tuple[SlotFrame | None, dict[str, Any]]:
    frame, meta = _coerce_slot_payload(payload)
    _backfill_from_normalized_text(frame)
    _normalize_inferred_slots(frame)
    validation = validate_slot_frame(frame)
    meta = {
        "confidence": frame.confidence,
        "normalized_text": frame.normalized_text,
        "schema_errors": list(meta.get("schema_errors", [])) + list(validation.errors),
    }
    if meta.get("schema_errors"):
        return None, meta
    return frame if frame.has_content() else None, meta


def build_llm_slot_frame(
    user_text: str,
    *,
    context_summary: str,
    config_path: str,
) -> LlmSlotBuildResult:
    prompt = _build_prompt(user_text, context_summary)
    llm_raw = ""
    stage_trace: dict[str, Any] = {
        "mode": "slot_first",
        "prompt_profile": STRICT_SLOT_PROMPT_PROFILE,
        "llm_json_parsed": False,
        "initial_schema_errors": [],
        "normalized_backfill_fields": [],
        "raw_text_backfill_fields": [],
        "content_after_payload": [],
        "content_after_refine": [],
        "repair_used": False,
        "uncertainty_signal": has_uncertainty_signal(user_text),
        "unresolved_targets": sorted(infer_unresolved_targets(user_text)),
        "pruned_unresolved_targets": [],
    }
    try:
        resp = chat(prompt, config_path=config_path, temperature=0.0)
        llm_raw = str(resp.get("response", ""))
    except Exception:
        logger.warning("LLM slot-frame call failed; returning structured failure.", exc_info=True)
        return LlmSlotBuildResult(
            ok=False,
            frame=None,
            normalized_text=user_text,
            confidence=0.0,
            llm_raw=llm_raw,
            fallback_reason=E_LLM_SLOT_CALL_FAILED,
            schema_errors=["llm_call_failed"],
            stage_trace=stage_trace | {"final_status": "llm_call_failed"},
        )

    payload = extract_json(_clean_response(llm_raw))
    if not isinstance(payload, dict):
        return LlmSlotBuildResult(
            ok=False,
            frame=None,
            normalized_text=user_text,
            confidence=0.0,
            llm_raw=llm_raw,
            fallback_reason=E_LLM_SLOT_PARSE_FAILED,
            schema_errors=["json_parse_failed"],
            stage_trace=stage_trace | {"final_status": "json_parse_failed"},
        )

    stage_trace["llm_json_parsed"] = True
    frame, meta = _coerce_slot_payload(payload)
    stage_trace["initial_schema_errors"] = list(meta.get("schema_errors", []))
    stage_trace["content_after_payload"] = sorted(_present_slot_paths(frame))

    normalized_before = _present_slot_paths(frame)
    if not frame.normalized_text:
        frame.normalized_text = str(meta.get("normalized_text") or user_text)
    _backfill_from_normalized_text(frame)
    normalized_after = _present_slot_paths(frame)
    stage_trace["normalized_backfill_fields"] = sorted(normalized_after - normalized_before)

    raw_before = set(normalized_after)
    _backfill_from_user_text(frame, user_text)
    _normalize_inferred_slots(frame)
    stage_trace["pruned_unresolved_targets"] = _prune_unresolved_targets(frame, user_text)
    _normalize_inferred_slots(frame)
    if _has_graph_family_cue(user_text):
        frame.geometry.kind = None
    unsupported_fields = _strip_unsupported_source_direction(frame, user_text)
    unsupported_fields.extend(_strip_unsupported_source_energy(frame, user_text))
    stage_trace["unsupported_llm_fields"] = unsupported_fields
    raw_after = _present_slot_paths(frame)
    stage_trace["raw_text_backfill_fields"] = sorted(raw_after - raw_before)
    stage_trace["repair_used"] = bool(stage_trace["normalized_backfill_fields"] or stage_trace["raw_text_backfill_fields"])
    stage_trace["content_after_refine"] = sorted(raw_after)

    final_validation = validate_slot_frame(frame)
    if final_validation.errors:
        meta = {
            "confidence": float(meta.get("confidence", 0.0)),
            "normalized_text": str(meta.get("normalized_text") or user_text),
            "schema_errors": list(final_validation.errors),
        }
        frame = None
    elif not frame.has_content():
        meta = {
            "confidence": float(meta.get("confidence", 0.0)),
            "normalized_text": str(meta.get("normalized_text") or user_text),
            "schema_errors": [],
        }
        frame = None

    if frame is None:
        fallback = E_LLM_SLOT_EMPTY if not meta.get("schema_errors") else E_LLM_SLOT_SCHEMA_INVALID
        stage_trace["final_status"] = fallback
        return LlmSlotBuildResult(
            ok=False,
            frame=None,
            normalized_text=str(meta.get("normalized_text") or user_text),
            confidence=float(meta.get("confidence", 0.0)),
            llm_raw=llm_raw,
            fallback_reason=fallback,
            schema_errors=list(meta.get("schema_errors", [])),
            stage_trace=stage_trace,
        )

    stage_trace["final_status"] = "ok"
    return LlmSlotBuildResult(
        ok=True,
        frame=frame,
        normalized_text=frame.normalized_text or user_text,
        confidence=frame.confidence,
        llm_raw=llm_raw,
        fallback_reason=None,
        schema_errors=[],
        stage_trace=stage_trace,
    )


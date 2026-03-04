from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.config.llm_prompt_registry import (
    STRICT_SLOT_PROMPT_PROFILE,
    build_strict_slot_prompt,
)
from core.config.output_format_registry import canonical_output_format
from core.orchestrator.types import Intent
from core.slots.slot_frame import SlotFrame
from core.slots.slot_validator import validate_slot_frame
from core.validation.error_codes import (
    E_LLM_SLOT_CALL_FAILED,
    E_LLM_SLOT_EMPTY,
    E_LLM_SLOT_PARSE_FAILED,
    E_LLM_SLOT_SCHEMA_INVALID,
)
from nlu.bert_lab.ollama_client import chat, extract_json


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
_MATERIAL_ALIASES = {
    "air": "G4_AIR",
    "\u7a7a\u6c14": "G4_AIR",
    "water": "G4_WATER",
    "\u6c34": "G4_WATER",
    "silicon": "G4_Si",
    "si": "G4_Si",
    "\u7845": "G4_Si",
    "copper": "G4_Cu",
    "cu": "G4_Cu",
    "g4_cu": "G4_Cu",
    "\u94dc": "G4_Cu",
    "aluminum": "G4_Al",
    "aluminium": "G4_Al",
    "al": "G4_Al",
    "g4_al": "G4_Al",
    "\u94dd": "G4_Al",
    "iron": "G4_Fe",
    "fe": "G4_Fe",
    "\u94c1": "G4_Fe",
    "lead": "G4_Pb",
    "pb": "G4_Pb",
    "\u94c5": "G4_Pb",
    "tungsten": "G4_W",
    "w": "G4_W",
    "\u94a8": "G4_W",
}
_NULL_LITERALS = {"null", "none", "n/a", "na", "unknown", "\u65e0", "\u672a\u77e5", "\u7a7a"}


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
    text = str(value or "").strip().lower().replace("×", "x")
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


def _canonical_material(value: Any) -> str | None:
    text = _clean_scalar(value)
    if not text:
        return None
    return _MATERIAL_ALIASES.get(text.lower(), text)


def _canonical_geometry_kind(value: Any) -> str | None:
    text = (_clean_scalar(value) or "").lower()
    if not text:
        return None
    return _GEOMETRY_ALIASES.get(text, text if text in _GEOMETRY_ALIASES.values() else None)


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


def _normalize_inferred_slots(frame: SlotFrame) -> None:
    if frame.geometry.kind is None:
        if frame.geometry.size_triplet_mm:
            frame.geometry.kind = "box"
        elif frame.geometry.radius_mm is not None or frame.geometry.half_length_mm is not None:
            frame.geometry.kind = "cylinder"


def _geometry_box_from_phrase(text: str) -> list[float] | None:
    low = text.lower().replace("×", "x")
    m = re.search(
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)\s*(?:x|by)\s*"
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)\s*(?:x|by)\s*"
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)",
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
        r"([-+]?\d*\.?\d+)\s*(mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)\s*\u89c1\u65b9",
        r"\u8fb9\u957f\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
        r"([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)\s*(?:\u7684)?(?:\u7acb\u65b9\u4f53|\u7acb\u65b9\u5757)",
    ]
    for pattern in side_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        side = _coerce_length_mm(f"{m.group(1)} {m.group(2)}")
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
            r"\u534a\u5f84\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
            r"r\s*[:=]\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
        ]
    )
    if radius is None:
        diameter = _match_length(
            [
                r"diameter\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
                r"\u76f4\u5f84\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
            ]
        )
        if diameter is not None:
            radius = diameter / 2.0

    half_length = _match_length(
        [
            r"half[-\s]*length\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            r"half[-\s]*z\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            r"hz\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
            r"\u534a(?:\u957f|\u957f\u5ea6)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
        ]
    )
    if half_length is None:
        half_length = _match_length(
            [
                r"(?:height|length)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(mm|cm|m)",
                r"\u9ad8(?:\u5ea6)?\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
                r"\u957f(?:\u5ea6)?\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(\u6beb\u7c73|\u5398\u7c73|\u7c73|mm|cm|m)",
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


def _source_direction_from_phrase(text: str) -> list[float] | None:
    patterns = [
        r"(?:direction|pointing)\s*(?:=|:)?\s*(\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\))",
        r"(?:direction|pointing)\s*(?:=|:)?\s*([+-][xyz])",
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
    if k in {"geometry.half_length", "geometry.half_length_mm", "geometry.height", "geometry.height_mm", "geometry.hz"}:
        half_length = _coerce_length_mm(v)
        if half_length is not None:
            frame.geometry.half_length_mm = half_length
        if frame.geometry.kind is None:
            frame.geometry.kind = "cylinder"
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
        elif "sphere" in low or "\u7403" in text:
            frame.geometry.kind = "sphere"

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

    if frame.materials.primary is None:
        for alias, canonical in _MATERIAL_ALIASES.items():
            if alias in low or alias in text:
                frame.materials.primary = canonical
                break

    if frame.source.position_mm is None:
        position = _source_position_from_phrase(text)
        if position is not None:
            frame.source.position_mm = position

    if frame.source.direction_vec is None:
        direction = _source_direction_from_phrase(text)
        if direction is not None:
            frame.source.direction_vec = direction

    _normalize_inferred_slots(frame)


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

    meta = {
        "confidence": frame.confidence,
        "normalized_text": frame.normalized_text,
        "schema_errors": errors,
    }
    return frame, meta


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
    }
    try:
        resp = chat(prompt, config_path=config_path, temperature=0.0)
        llm_raw = str(resp.get("response", ""))
    except Exception:
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

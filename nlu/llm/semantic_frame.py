from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.config.output_format_registry import canonical_output_format
from core.config.prompt_profiles import PromptBuildResult, PromptTask, build_prompt, validate_prompt_output
from core.domain.lexicon import BASE_MATERIAL_ALIASES
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.validation.error_codes import (
    E_LLM_FRAME_CALL_FAILED,
    E_LLM_FRAME_EMPTY_UPDATES,
    E_LLM_FRAME_PARSE_FAILED,
    E_LLM_FRAME_SCHEMA_INVALID,
)
from nlu.llm_support.ollama_client import chat, extract_json


logger = logging.getLogger(__name__)


_ALLOWED_PREFIXES = ("geometry.", "materials.", "source.", "physics.", "output.")
_CANONICAL_PATH_ALIASES = {
    "geometry.type": "geometry.structure",
    "materials.target": "materials.selected_materials",
    "materials.material": "materials.selected_materials",
    "materials.primary_material": "materials.selected_materials",
    "source.kind": "source.type",
    "source.source_type": "source.type",
    "source.particle_type": "source.particle",
    "physics.list": "physics.physics_list",
    "physics.list.name": "physics.physics_list",
    "output.type": "output.format",
}
_SPECIAL_GEOMETRY_SIZE_PATHS = {
    "geometry.size",
    "geometry.box",
    "geometry.dimensions",
    "geometry.box_size",
}
_SOURCE_TYPES = {"point", "beam", "plane", "isotropic"}
_PARTICLE_ALIASES = {
    "gamma": "gamma",
    "photon": "gamma",
    "electron": "e-",
    "e-": "e-",
    "proton": "proton",
    "neutron": "neutron",
}
_MATERIAL_ALIASES = dict(BASE_MATERIAL_ALIASES)
_STRUCTURE_ALIASES = {
    "box": "single_box",
    "cube": "single_box",
    "cuboid": "single_box",
    "single_box": "single_box",
    "cylinder": "single_tubs",
    "tubs": "single_tubs",
    "single_tubs": "single_tubs",
    "sphere": "single_sphere",
    "single_sphere": "single_sphere",
    "orb": "single_orb",
    "single_orb": "single_orb",
}
_ALLOWED_HINTS = {
    "",
    "ring",
    "grid",
    "nest",
    "stack",
    "shell",
    "single_box",
    "single_tubs",
    "single_sphere",
    "single_orb",
    "single_cons",
    "single_trd",
    "single_polycone",
    "single_cuttubs",
    "boolean",
    "unknown",
}
_INTENT_MAP = {x.value: x for x in Intent}


@dataclass
class LlmFrameBuildResult:
    ok: bool
    candidate: CandidateUpdate | None
    user_candidate: CandidateUpdate | None
    normalized_text: str
    structure_hint: str
    confidence: float
    llm_raw: str
    fallback_reason: str | None
    schema_errors: list[str]


def _clean_response(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    return text


def _build_prompt(user_text: str, context_summary: str) -> PromptBuildResult:
    return build_prompt(
        PromptTask.SEMANTIC_EXTRACT,
        "en",
        {"user_text": user_text, "context_summary": context_summary},
    )


def _intent_from_any(value: Any) -> Intent | None:
    key = str(value or "").strip().upper()
    return _INTENT_MAP.get(key)


def _coerce_number(value: Any) -> float | int | Any:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    v = value.strip()
    if not v:
        return value
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        m = re.match(r"^\s*([-+]?\d*\.?\d+)\s*(mev|gev|kev)\s*$", v.lower())
        if not m:
            return value
        scalar = float(m.group(1))
        unit = m.group(2)
        if unit == "gev":
            return scalar * 1000.0
        if unit == "kev":
            return scalar * 0.001
        return scalar


def _value_to_mm(value: float, unit: str) -> float:
    u = unit.strip().lower()
    if u == "m":
        return value * 1000.0
    if u == "cm":
        return value * 10.0
    return value


def _parse_triplet_mm(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            text = ",".join(str(x) for x in value)
    else:
        text = str(value or "")
    text = text.strip().lower().replace("脳", "x")
    if not text:
        return None

    num = r"[-+]?\d*\.?\d+"
    unit = r"(mm|cm|m)"
    pat1 = rf"({num})\s*{unit}\s*[,x]\s*({num})\s*{unit}\s*[,x]\s*({num})\s*{unit}"
    m1 = re.fullmatch(pat1, text)
    if m1:
        return (
            _value_to_mm(float(m1.group(1)), m1.group(2)),
            _value_to_mm(float(m1.group(3)), m1.group(4)),
            _value_to_mm(float(m1.group(5)), m1.group(6)),
        )

    pat2 = rf"({num})\s*[,x]\s*({num})\s*[,x]\s*({num})\s*{unit}"
    m2 = re.fullmatch(pat2, text)
    if m2:
        return (
            _value_to_mm(float(m2.group(1)), m2.group(4)),
            _value_to_mm(float(m2.group(2)), m2.group(4)),
            _value_to_mm(float(m2.group(3)), m2.group(4)),
        )
    return None


def _coerce_vector(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        vv = value.get("value")
        if isinstance(vv, list) and len(vv) == 3:
            try:
                return {"type": "vector", "value": [float(vv[0]), float(vv[1]), float(vv[2])]}
            except Exception:
                return None
        return None
    if isinstance(value, list) and len(value) == 3:
        try:
            return {"type": "vector", "value": [float(value[0]), float(value[1]), float(value[2])]}
        except Exception:
            return None
    if isinstance(value, str):
        low = value.strip().lower().replace(" ", "")
        if low in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            mapping = {
                "+x": [1.0, 0.0, 0.0],
                "-x": [-1.0, 0.0, 0.0],
                "+y": [0.0, 1.0, 0.0],
                "-y": [0.0, -1.0, 0.0],
                "+z": [0.0, 0.0, 1.0],
                "-z": [0.0, 0.0, -1.0],
            }
            return {"type": "vector", "value": mapping[low]}
        m = re.match(r"^\(?\s*([-+]?\d*\.?\d+)\s*[,锛宂\s*([-+]?\d*\.?\d+)\s*[,锛宂\s*([-+]?\d*\.?\d+)\s*\)?$", low)
        if m:
            return {"type": "vector", "value": [float(m.group(1)), float(m.group(2)), float(m.group(3))]}
    return None


def _sanitize_update_value(path: str, value: Any) -> Any:
    if path in {"source.position", "source.direction"}:
        vec = _coerce_vector(value)
        return vec if vec is not None else value
    if path == "source.energy":
        return _coerce_number(value)
    if path.startswith("geometry.params."):
        return _coerce_number(value)
    if path == "materials.selected_materials" and isinstance(value, str):
        return [value]
    return value


def _canonicalize_target_path(path: str, structure_hint: str) -> tuple[list[str], list[str], bool]:
    raw = str(path or "").strip()
    if not raw:
        return [], ["target_path_empty"], False
    if not raw.startswith(_ALLOWED_PREFIXES):
        return [], [f"target_path_out_of_scope:{raw}"], True
    canon = _CANONICAL_PATH_ALIASES.get(raw, raw)
    if canon in _SPECIAL_GEOMETRY_SIZE_PATHS:
        out = ["geometry.params.module_x", "geometry.params.module_y", "geometry.params.module_z"]
        if structure_hint in {"", "single_box"}:
            out.insert(0, "geometry.structure")
        return out, [], False
    if canon == "source.type":
        return ["source.type", "source.particle"], [], False
    return [canon], [], False


def _expand_update_paths(path: str, value: Any, structure_hint: str) -> tuple[list[tuple[str, Any]], list[str], bool]:
    raw = str(path or "").strip()
    if not raw:
        return [], ["update_missing_path"], False
    if not raw.startswith(_ALLOWED_PREFIXES):
        return [], [f"update_out_of_scope:{raw}"], True

    canon = _CANONICAL_PATH_ALIASES.get(raw, raw)
    if canon in _SPECIAL_GEOMETRY_SIZE_PATHS:
        triplet = _parse_triplet_mm(value)
        if triplet is None:
            return [], [f"update_bad_geometry_size:{value}"], False
        out: list[tuple[str, Any]] = []
        if structure_hint in {"", "single_box"}:
            out.append(("geometry.structure", "single_box"))
        out.extend(
            [
                ("geometry.params.module_x", float(triplet[0])),
                ("geometry.params.module_y", float(triplet[1])),
                ("geometry.params.module_z", float(triplet[2])),
            ]
        )
        return out, [], False

    if canon == "geometry.structure":
        mapped = _STRUCTURE_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip())
        return [(canon, mapped)], [], False

    if canon == "materials.selected_materials":
        if isinstance(value, list):
            selected = []
            for item in value:
                text = str(item).strip()
                if not text:
                    continue
                selected.append(_MATERIAL_ALIASES.get(text.lower(), text))
            if not selected:
                return [], ["update_empty_materials"], False
            return [(canon, selected)], [], False
        text = str(value or "").strip()
        if not text:
            return [], ["update_empty_materials"], False
        return [(canon, [_MATERIAL_ALIASES.get(text.lower(), text)])], [], False

    if canon == "source.type":
        low = str(value or "").strip().lower()
        if low in _SOURCE_TYPES:
            return [(canon, low)], [], False
        particle = _PARTICLE_ALIASES.get(low)
        if particle:
            return [("source.particle", particle)], [], False
        return [], [f"update_invalid_source_type:{value}"], False

    if canon == "source.particle":
        low = str(value or "").strip().lower()
        return [(canon, _PARTICLE_ALIASES.get(low, str(value or "").strip()))], [], False

    if canon == "physics.physics_list":
        text = str(value or "").strip()
        if not text:
            return [], ["update_empty_physics_list"], False
        return [(canon, text)], [], False

    if canon == "output.format":
        fmt = canonical_output_format(value)
        if fmt is None and not str(value or "").strip():
            return [], ["update_empty_output_format"], False
        if fmt is not None:
            return [(canon, fmt)], [], False
        return [(canon, str(value or "").strip().lower())], [], False

    if canon.startswith("geometry.params.") or canon in {
        "source.energy",
        "source.position",
        "source.direction",
        "materials.volume_material_map",
        "physics.backup_physics_list",
        "output.path",
    }:
        return [(canon, value)], [], False

    return [], [f"update_unknown_path:{canon}"], False


def parse_semantic_frame_payload(payload: dict[str, Any], *, turn_id: int) -> tuple[CandidateUpdate | None, CandidateUpdate | None, dict[str, Any]]:
    errors: list[str] = []
    fatal_errors: list[str] = []
    intent = _intent_from_any(payload.get("intent"))
    if intent is None:
        errors.append("intent_invalid")
        intent = Intent.OTHER

    normalized_text = str(payload.get("normalized_text", "")).strip()
    structure_hint = str(payload.get("structure_hint", "")).strip().lower()
    if structure_hint not in _ALLOWED_HINTS:
        errors.append(f"structure_hint_invalid:{structure_hint}")
        structure_hint = ""

    target_paths_raw = payload.get("target_paths", [])
    if not isinstance(target_paths_raw, list):
        errors.append("target_paths_not_list")
        target_paths_raw = []
    target_paths: list[str] = []
    for raw_path in target_paths_raw:
        paths, path_errors, fatal = _canonicalize_target_path(str(raw_path), structure_hint)
        target_paths.extend(paths)
        if fatal:
            fatal_errors.extend(path_errors)
        else:
            errors.extend(path_errors)

    updates_raw = payload.get("updates", [])
    if not isinstance(updates_raw, list):
        errors.append("updates_not_list")
        updates_raw = []

    updates: list[UpdateOp] = []
    for idx, item in enumerate(updates_raw):
        if not isinstance(item, dict):
            errors.append(f"update_{idx}_not_object")
            continue
        path = str(item.get("path", "")).strip()
        if not path:
            errors.append(f"update_{idx}_missing_path")
            continue
        op = str(item.get("op", "set")).strip().lower()
        if op not in {"set", "remove"}:
            errors.append(f"update_{idx}_bad_op:{op}")
            continue
        value = item.get("value")
        if op == "remove":
            canon_paths, path_errors, fatal = _canonicalize_target_path(path, structure_hint)
            if fatal:
                fatal_errors.extend(path_errors)
                continue
            errors.extend(path_errors)
            for canon_path in canon_paths:
                updates.append(
                    UpdateOp(
                        path=canon_path,
                        op="remove",
                        value=None,
                        producer=Producer.LLM_SEMANTIC_FRAME,
                        confidence=0.75,
                        turn_id=turn_id,
                    )
                )
            continue
        expanded, update_errors, fatal = _expand_update_paths(path, value, structure_hint)
        if fatal:
            fatal_errors.extend(update_errors)
            continue
        errors.extend(update_errors)
        for canon_path, canon_value in expanded:
            updates.append(
                UpdateOp(
                    path=canon_path,
                    op="set",
                    value=_sanitize_update_value(canon_path, canon_value),
                    producer=Producer.LLM_SEMANTIC_FRAME,
                    confidence=0.75,
                    turn_id=turn_id,
                )
            )

    deduped_updates: dict[str, UpdateOp] = {}
    for update in updates:
        deduped_updates[update.path] = update
    updates = list(deduped_updates.values())
    target_paths = sorted(set(target_paths) | {u.path for u in updates})

    confidence = payload.get("confidence", 0.75)
    if not isinstance(confidence, (int, float)):
        errors.append("confidence_not_number")
        confidence = 0.75
    confidence = max(0.0, min(1.0, float(confidence)))

    candidate = CandidateUpdate(
        producer=Producer.LLM_SEMANTIC_FRAME,
        intent=intent,
        target_paths=target_paths,
        updates=updates,
        confidence=confidence,
        rationale="llm_semantic_frame",
    )
    user_candidate = CandidateUpdate(
        producer=Producer.USER_EXPLICIT,
        intent=intent,
        target_paths=target_paths,
        updates=[],
        confidence=confidence,
        rationale="llm_semantic_frame_user",
    )
    meta = {
        "normalized_text": normalized_text,
        "structure_hint": structure_hint,
        "confidence": confidence,
        "schema_errors": fatal_errors + errors,
    }
    if fatal_errors:
        return None, None, meta
    return candidate, user_candidate, meta


def build_llm_semantic_frame(
    user_text: str,
    *,
    context_summary: str,
    config_path: str,
    turn_id: int,
) -> LlmFrameBuildResult:
    prompt_build = _build_prompt(user_text, context_summary)
    prompt = prompt_build.prompt
    llm_raw = ""
    try:
        resp = chat(prompt, config_path=config_path, temperature=0.0)
        llm_raw = str(resp.get("response", ""))
    except Exception:
        logger.warning("LLM semantic-frame call failed; returning structured failure.", exc_info=True)
        return LlmFrameBuildResult(
            ok=False,
            candidate=None,
            user_candidate=None,
            normalized_text=user_text,
            structure_hint="",
            confidence=0.0,
            llm_raw=llm_raw,
            fallback_reason=E_LLM_FRAME_CALL_FAILED,
            schema_errors=["llm_call_failed"],
        )

    cleaned = _clean_response(llm_raw)
    payload = extract_json(cleaned)
    if not isinstance(payload, dict):
        return LlmFrameBuildResult(
            ok=False,
            candidate=None,
            user_candidate=None,
            normalized_text=user_text,
            structure_hint="",
            confidence=0.0,
            llm_raw=llm_raw,
            fallback_reason=E_LLM_FRAME_PARSE_FAILED,
            schema_errors=["json_parse_failed"],
        )

    prompt_validation = validate_prompt_output(PromptTask.SEMANTIC_EXTRACT, "en", payload)
    if not prompt_validation.ok:
        return LlmFrameBuildResult(
            ok=False,
            candidate=None,
            user_candidate=None,
            normalized_text=user_text,
            structure_hint="",
            confidence=0.0,
            llm_raw=llm_raw,
            fallback_reason=E_LLM_FRAME_SCHEMA_INVALID,
            schema_errors=list(prompt_validation.errors),
        )

    candidate, user_candidate, meta = parse_semantic_frame_payload(payload, turn_id=turn_id)
    if candidate is None or user_candidate is None:
        return LlmFrameBuildResult(
            ok=False,
            candidate=None,
            user_candidate=None,
            normalized_text=str(meta.get("normalized_text") or user_text),
            structure_hint=str(meta.get("structure_hint", "")),
            confidence=float(meta.get("confidence", 0.0)),
            llm_raw=llm_raw,
            fallback_reason=E_LLM_FRAME_SCHEMA_INVALID,
            schema_errors=list(meta.get("schema_errors", [])),
        )
    if len(candidate.updates) == 0 and not (
        str(meta.get("normalized_text") or "").strip()
        or str(meta.get("structure_hint") or "").strip()
        or candidate.intent in {Intent.CONFIRM, Intent.REJECT, Intent.QUESTION, Intent.OTHER}
    ):
        return LlmFrameBuildResult(
            ok=False,
            candidate=None,
            user_candidate=None,
            normalized_text=str(meta.get("normalized_text") or user_text),
            structure_hint=str(meta.get("structure_hint", "")),
            confidence=float(meta.get("confidence", 0.0)),
            llm_raw=llm_raw,
            fallback_reason=E_LLM_FRAME_EMPTY_UPDATES,
            schema_errors=["empty_updates"],
        )
    return LlmFrameBuildResult(
        ok=True,
        candidate=candidate,
        user_candidate=user_candidate,
        normalized_text=str(meta.get("normalized_text") or user_text),
        structure_hint=str(meta.get("structure_hint", "")),
        confidence=float(meta.get("confidence", 0.0)),
        llm_raw=llm_raw,
        fallback_reason=None,
        schema_errors=[],
    )


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.domain.lexicon import BASE_MATERIAL_ALIASES, BASE_SOURCE_TYPE_ALIASES
from core.semantic_frame import SemanticFrame
from core.geometry.dialogue_registry import graph_dialogue_missing_paths
from builder.geometry.synthesize import synthesize_from_params
from nlu.runtime_components.graph_search import search_candidate_graphs
from nlu.runtime_components.infer import extract_params, predict_structure
from nlu.runtime_components.infer import _require_local_model_dir
from nlu.runtime_components.postprocess import merge_params


ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge" / "data"
MODELS_DIR = ROOT / "nlu" / "training" / "bert_lab" / "models"

_CACHE: dict[str, list[str]] | None = None

MATERIAL_ALIASES = dict(BASE_MATERIAL_ALIASES)

PARTICLE_ALIASES = {
    "gamma": "gamma",
    "photon": "gamma",
    "electron": "e-",
    "e-": "e-",
    "proton": "proton",
    "neutron": "neutron",
}

SOURCE_TYPE_ALIASES = dict(BASE_SOURCE_TYPE_ALIASES)

FALLBACK_GRAPH_STRUCTURE = {
    "ring_modules": "ring",
    "grid_modules": "grid",
    "nest_box_tubs": "nest",
    "stack_in_box": "stack",
    "shell_nested": "shell",
    "boolean_union_boxes": "boolean",
    "boolean_subtraction_boxes": "boolean",
    "boolean_intersection_boxes": "boolean",
}


def _load_knowledge() -> dict[str, list[str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    materials = json.loads((KNOWLEDGE_DIR / "materials_geant4_nist.json").read_text(encoding="utf-8")).get(
        "materials", []
    )
    physics_lists = json.loads((KNOWLEDGE_DIR / "physics_lists.json").read_text(encoding="utf-8")).get("items", [])
    particles = json.loads((KNOWLEDGE_DIR / "particles.json").read_text(encoding="utf-8")).get("items", [])
    output_formats = json.loads((KNOWLEDGE_DIR / "output_formats.json").read_text(encoding="utf-8")).get("items", [])
    source_types = json.loads((KNOWLEDGE_DIR / "source_constraints.json").read_text(encoding="utf-8")).get("types", [])
    if not source_types:
        source_types = ["point", "beam", "plane", "isotropic"]
    _CACHE = {
        "materials": materials,
        "physics_lists": physics_lists,
        "particles": particles,
        "output_formats": output_formats,
        "source_types": source_types,
    }
    return _CACHE


def _match_any(text: str, items: list[str]) -> str | None:
    if not text:
        return None
    ordered = sorted((it for it in items if it), key=len, reverse=True)
    for item in ordered:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])", re.IGNORECASE)
        if pat.search(text):
            return item
    return None


def _alias_match(text: str, mapping: dict[str, str], allowed: list[str]) -> str | None:
    low = text.lower()
    for alias, canonical in mapping.items():
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", low):
            if canonical in allowed:
                return canonical
    return None


def _material_mentions(text: str, allowed: list[str]) -> list[str]:
    low = text.lower()
    hits: list[tuple[int, str]] = []
    for alias, canonical in MATERIAL_ALIASES.items():
        for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", low):
            if canonical in allowed:
                hits.append((match.start(), canonical))
    hits.sort(key=lambda item: item[0])
    ordered: list[str] = []
    seen: set[str] = set()
    for _, canonical in hits:
        if canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    return ordered


def _select_primary_material(text: str, allowed: list[str]) -> str | None:
    mentions = _material_mentions(text, allowed)
    if not mentions:
        return None
    low = text.lower()
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
            "减去",
            "差集",
            "挖空",
            "开孔",
            "打孔",
        )
    ):
        for material in mentions:
            if material != "G4_AIR":
                return material
    if any(token in low for token in ("parent", "outer", "container", "外盒", "容器")):
        for material in mentions:
            if material != "G4_AIR":
                return material
    return mentions[0]


def _has_unknown_material_marker(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search("(?:\u6750\u6599|material)\\s*[:\uff1a]?\\s*[?\uff1f]+", text)
        or re.search("(?:\u6750\u6599|material)\\s*(?:unknown|tbd|unspecified)\\b", low)
    )

def _pick_ner_model() -> str:
    p = MODELS_DIR / "ner"
    return _require_local_model_dir(p, label="NER")


def _pick_structure_model() -> str | None:
    candidates = [
        MODELS_DIR / "structure_router_v3",
        MODELS_DIR / "structure_controlled_v5_e2",
        MODELS_DIR / "structure_controlled_v4c_e1",
        MODELS_DIR / "structure_controlled_v3_e1",
        MODELS_DIR / "structure",
    ]
    for p in candidates:
        if (p / "config.json").exists():
            return str(p)
    return None


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "structure": candidate.summary,
        "chosen_skeleton": candidate.structure,
        "feasible": candidate.feasible,
        "missing_params": list(candidate.missing_params),
        "errors": list(candidate.errors),
        "warnings": list(candidate.warnings),
        "score": float(candidate.score),
        "score_breakdown": dict(candidate.score_breakdown),
    }


def _normalize_graph_program_root(structure: str | None, graph_program: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(graph_program, dict):
        return graph_program
    normalized = dict(graph_program)
    if structure == "stack":
        normalized["root"] = "stack"
    elif structure == "shell":
        normalized["root"] = "shell"
    elif structure == "boolean":
        normalized["root"] = "boolean"
    return normalized


def _fallback_structure_from_skeleton(chosen_skeleton: str) -> str:
    return FALLBACK_GRAPH_STRUCTURE.get(str(chosen_skeleton or "").strip(), "")


def _committed_geometry_params(
    explicit_params: dict[str, float],
    candidate_params: dict[str, float],
    missing_params: list[str],
) -> dict[str, float]:
    if not candidate_params:
        return dict(explicit_params)
    if not missing_params:
        return dict(candidate_params)
    committed = dict(explicit_params)
    for key in explicit_params:
        if key in candidate_params:
            committed[key] = candidate_params[key]
    return committed


def _fallback_dialogue_skeleton(text: str) -> str:
    low = text.lower()
    minus_is_boolean = bool(
        re.search(r"\b(?:box|cube|cuboid|cylinder|sphere|solid|target)\b.{0,80}\bminus\b.{0,80}\b(?:box|cube|cuboid|cylinder|sphere|solid|target)\b", low)
    )
    if any(token in low for token in ("intersection", "intersect", "相交", "交集")):
        return "boolean_intersection_boxes"
    if minus_is_boolean or any(
        token in low
        for token in (
            "subtraction",
            "subtract",
            "difference",
            "hole",
            "cut out",
            "cutout",
            "减去",
            "差集",
            "挖空",
            "开孔",
            "打孔",
        )
    ):
        return "boolean_subtraction_boxes"
    if any(token in low for token in ("union", "boolean", "combine", "merge", "并", "合并", "并集")):
        return "boolean_union_boxes"
    if any(token in low for token in ("ring", "annulus", "circular", "环", "环形", "圆环", "围成一圈", "一圈")):
        return "ring_modules"
    if any(token in low for token in ("grid", "array", "matrix", "阵列", "二维阵列", "探测板", "网格")):
        return "grid_modules"
    if any(token in low for token in ("stack", "layers", "layer", "stacked", "sandwich", "堆叠", "夹层", "沿 z 方向", "沿z方向")):
        return "stack_in_box"
    if any(token in low for token in ("shell", "concentric", "coaxial", "壳", "同心", "屏蔽壳", "屏蔽层", "多层壳")):
        return "shell_nested"
    if any(token in low for token in ("nest", "inside", "contains", "inner", "outer", "嵌套", "内嵌", "外盒", "盒子里", "外盒体", "内盒体", "包住")):
        return "nest_box_tubs"
    return ""


def _infer_source_type_from_content(text: str, normalized_text: str, frame: SemanticFrame) -> str | None:
    merged = f"{text} ; {normalized_text}".lower()
    if any(token in merged for token in ("point source", "point", "点源", "点状源")):
        return "point"
    if any(token in merged for token in ("beam", "pencil beam", "collimated", "束流", "粒子束", "准直", "平行束", "入射")):
        return "beam"
    if any(token in merged for token in ("plane source", "plane", "面源")):
        return "plane"
    if any(token in merged for token in ("isotropic", "各向同性")):
        return "isotropic"
    if frame.source.direction is not None and frame.source.position is not None:
        if any(token in merged for token in ("束流", "平行束", "入射", "沿 +z", "沿+z", "along +z")):
            return "beam"
    if frame.source.direction is not None and frame.source.position is not None and frame.source.particle in {"proton", "e-", "electron"}:
        return "beam"
    return None


def extract_runtime_semantic_frame(
    text: str,
    *,
    normalized_text: str = "",
    min_confidence: float = 0.6,
    device: str = "auto",
    context_summary: str = "",
    apply_autofix: bool = False,
) -> tuple[SemanticFrame, dict[str, Any]]:
    _ = context_summary
    frame = SemanticFrame()
    graph_text = text
    param_text = text
    if normalized_text and normalized_text.strip():
        param_text = f"{text} ; {normalized_text}".strip(" ;")
    debug: dict[str, Any] = {
        "inference_backend": "runtime_semantic",
        "requires_llm_normalization": False,
        "normalized_text": normalized_text or text,
        "graph_text": graph_text,
        "param_text": param_text,
        "normalization": {"enabled": False, "used": False},
    }

    params: dict[str, float] = {}
    try:
        params = extract_params(param_text, _pick_ner_model(), device)
    except Exception as ex:
        debug["ner_error"] = str(ex)
    params, notes = merge_params(param_text, params)
    frame.notes.extend(notes)

    structure_model = _pick_structure_model()
    structure_prior_label = ""
    structure_prior_confidence = 0.0
    if structure_model:
        try:
            prior_label, prior_scores, prior_ranked = predict_structure(
                graph_text,
                structure_model,
                device=device,
                min_confidence=min_confidence,
            )
            structure_prior_label = str(prior_label or "")
            structure_prior_confidence = float(prior_scores.get(structure_prior_label, 0.0))
            debug["structure_prior"] = {
                "label": structure_prior_label,
                "confidence": structure_prior_confidence,
                "ranked": list(prior_ranked),
                "model": structure_model,
            }
        except Exception as ex:
            debug["structure_prior_error"] = str(ex)

    graph_result = search_candidate_graphs(
        graph_text,
        params,
        min_confidence=min_confidence,
        seed=7,
        top_k=3,
        apply_autofix=apply_autofix,
        prior_summary=structure_prior_label,
        prior_confidence=structure_prior_confidence,
    )
    frame.notes.extend(graph_result.notes)
    debug["scores"] = dict(graph_result.scores)
    debug["ranked"] = list(graph_result.ranked)
    debug["graph_candidates"] = [_candidate_payload(c) for c in graph_result.candidates]

    chosen_candidate = None
    fallback_synth: dict[str, Any] | None = None
    if graph_result.chosen_skeleton:
        for candidate in graph_result.candidates:
            if candidate.structure == graph_result.chosen_skeleton:
                chosen_candidate = candidate
                break
    dialogue_skeleton = graph_result.chosen_skeleton or _fallback_dialogue_skeleton(text)
    dialogue_structure = graph_result.structure
    dialogue_missing_params: list[str] = []
    if chosen_candidate is not None:
        dialogue_missing_params = list(chosen_candidate.missing_params)
    elif dialogue_skeleton:
        fallback_synth = synthesize_from_params(dialogue_skeleton, params, seed=7, apply_autofix=apply_autofix)
        dialogue_missing_params = list(fallback_synth.get("missing_params", []))
        if dialogue_structure == "unknown":
            dialogue_structure = _fallback_structure_from_skeleton(dialogue_skeleton) or str(
                fallback_synth.get("structure", "") or ""
            )
    debug["graph_choice"] = {
        "structure": dialogue_structure,
        "chosen_skeleton": dialogue_skeleton,
        "missing_params": dialogue_missing_params,
        "dialogue_missing_paths": graph_dialogue_missing_paths(
            dialogue_skeleton,
            dialogue_missing_params,
        ),
    }

    if graph_result.structure != "unknown":
        frame.geometry.structure = graph_result.structure
    elif dialogue_structure:
        frame.geometry.structure = dialogue_structure
    if graph_result.chosen_skeleton:
        frame.geometry.chosen_skeleton = graph_result.chosen_skeleton
    elif dialogue_skeleton:
        frame.geometry.chosen_skeleton = dialogue_skeleton
    if graph_result.graph_program is not None:
        frame.geometry.graph_program = _normalize_graph_program_root(
            graph_result.structure,
            graph_result.graph_program,
        )
    elif fallback_synth is not None and isinstance(fallback_synth.get("dsl"), dict):
        fallback_structure = frame.geometry.structure or dialogue_structure
        frame.geometry.graph_program = _normalize_graph_program_root(
            fallback_structure,
            dict(fallback_synth.get("dsl", {})),
        )

    if chosen_candidate is not None:
        frame.geometry.params.update(
            _committed_geometry_params(
                params,
                chosen_candidate.params_filled,
                chosen_candidate.missing_params,
            )
        )
    else:
        frame.geometry.params.update(params)

    knowledge = _load_knowledge()
    raw_material = None if _has_unknown_material_marker(text) else (
        _select_primary_material(text, knowledge["materials"])
        or _match_any(text, knowledge["materials"])
        or _alias_match(text, MATERIAL_ALIASES, knowledge["materials"])
    )
    normalized_material = None if _has_unknown_material_marker(normalized_text or "") else (
        _select_primary_material(normalized_text, knowledge["materials"])
        or _match_any(normalized_text, knowledge["materials"])
        or _alias_match(normalized_text, MATERIAL_ALIASES, knowledge["materials"])
    )
    material = raw_material or normalized_material
    if material:
        frame.materials.selected_materials = [material]

    particle = _match_any(param_text, knowledge["particles"]) or _alias_match(
        param_text, PARTICLE_ALIASES, knowledge["particles"]
    )
    if particle:
        frame.source.particle = particle

    raw_source_type = _match_any(text, knowledge["source_types"]) or _alias_match(
        text, SOURCE_TYPE_ALIASES, knowledge["source_types"]
    )
    normalized_source_type = _match_any(normalized_text, knowledge["source_types"]) or _alias_match(
        normalized_text, SOURCE_TYPE_ALIASES, knowledge["source_types"]
    )
    source_type = raw_source_type or normalized_source_type
    if not source_type:
        source_type = _infer_source_type_from_content(text, normalized_text, frame)
    if source_type:
        frame.source.type = source_type

    physics_list = _match_any(param_text, knowledge["physics_lists"])
    if physics_list:
        frame.physics.physics_list = physics_list

    output_format = _match_any(param_text, knowledge["output_formats"])
    if output_format:
        frame.output.format = output_format

    return frame, debug

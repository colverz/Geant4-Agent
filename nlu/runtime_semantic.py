from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.semantic_frame import SemanticFrame
from nlu.runtime_components.graph_search import search_candidate_graphs
from nlu.runtime_components.infer import extract_params
from nlu.runtime_components.infer import _require_local_model_dir
from nlu.runtime_components.postprocess import merge_params


ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge" / "data"
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"

_CACHE: dict[str, list[str]] | None = None

MATERIAL_ALIASES = {
    "air": "G4_AIR",
    "water": "G4_WATER",
    "silicon": "G4_Si",
    "si": "G4_Si",
    "copper": "G4_Cu",
    "cu": "G4_Cu",
    "aluminum": "G4_Al",
    "aluminium": "G4_Al",
    "al": "G4_Al",
    "iron": "G4_Fe",
    "fe": "G4_Fe",
    "lead": "G4_Pb",
    "pb": "G4_Pb",
    "tungsten": "G4_W",
    "w": "G4_W",
}

PARTICLE_ALIASES = {
    "gamma": "gamma",
    "photon": "gamma",
    "electron": "e-",
    "e-": "e-",
    "proton": "proton",
    "neutron": "neutron",
}

SOURCE_TYPE_ALIASES = {
    "point source": "point",
    "point": "point",
    "\u70b9\u6e90": "point",
    "\u70b9\u72b6\u6e90": "point",
    "beam": "beam",
    "\u675f\u6d41": "beam",
    "\u7c92\u5b50\u675f": "beam",
    "isotropic": "isotropic",
    "\u5404\u5411\u540c\u6027": "isotropic",
    "plane source": "plane",
    "plane": "plane",
    "\u9762\u6e90": "plane",
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


def _pick_ner_model() -> str:
    p = MODELS_DIR / "ner"
    return _require_local_model_dir(p, label="NER")


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


def extract_runtime_semantic_frame(
    text: str,
    *,
    min_confidence: float = 0.6,
    device: str = "auto",
    context_summary: str = "",
) -> tuple[SemanticFrame, dict[str, Any]]:
    _ = context_summary
    frame = SemanticFrame()
    debug: dict[str, Any] = {
        "inference_backend": "runtime_semantic",
        "requires_llm_normalization": False,
        "normalized_text": text,
        "normalization": {"enabled": False, "used": False},
    }

    params: dict[str, float] = {}
    try:
        params = extract_params(text, _pick_ner_model(), device)
    except Exception as ex:
        debug["ner_error"] = str(ex)
    params, notes = merge_params(text, params)
    frame.notes.extend(notes)

    graph_result = search_candidate_graphs(
        text,
        params,
        min_confidence=min_confidence,
        seed=7,
        top_k=3,
        apply_autofix=False,
    )
    frame.notes.extend(graph_result.notes)
    debug["scores"] = dict(graph_result.scores)
    debug["ranked"] = list(graph_result.ranked)
    debug["graph_candidates"] = [_candidate_payload(c) for c in graph_result.candidates]

    chosen_candidate = None
    if graph_result.chosen_skeleton:
        for candidate in graph_result.candidates:
            if candidate.structure == graph_result.chosen_skeleton:
                chosen_candidate = candidate
                break

    if graph_result.structure != "unknown":
        frame.geometry.structure = graph_result.structure
    if graph_result.chosen_skeleton:
        frame.geometry.chosen_skeleton = graph_result.chosen_skeleton
    if graph_result.graph_program is not None:
        frame.geometry.graph_program = graph_result.graph_program
    if chosen_candidate is not None:
        frame.geometry.params.update(chosen_candidate.params_filled)
    else:
        frame.geometry.params.update(params)

    knowledge = _load_knowledge()
    material = _match_any(text, knowledge["materials"]) or _alias_match(text, MATERIAL_ALIASES, knowledge["materials"])
    if material:
        frame.materials.selected_materials = [material]

    particle = _match_any(text, knowledge["particles"]) or _alias_match(text, PARTICLE_ALIASES, knowledge["particles"])
    if particle:
        frame.source.particle = particle

    source_type = _match_any(text, knowledge["source_types"]) or _alias_match(text, SOURCE_TYPE_ALIASES, knowledge["source_types"])
    if source_type:
        frame.source.type = source_type

    physics_list = _match_any(text, knowledge["physics_lists"])
    if physics_list:
        frame.physics.physics_list = physics_list

    output_format = _match_any(text, knowledge["output_formats"])
    if output_format:
        frame.output.format = output_format

    return frame, debug

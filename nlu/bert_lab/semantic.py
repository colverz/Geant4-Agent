from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from core.semantic_frame import SemanticFrame
from nlu.bert_lab.graph_search import search_candidate_graphs
from nlu.bert_lab.infer import extract_params
from nlu.bert_lab.llm_bridge import build_normalization_prompt
from nlu.bert_lab.multitask_infer import predict_multitask
from nlu.bert_lab.ollama_client import chat, extract_json
from nlu.bert_lab.postprocess import merge_params


ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge" / "data"
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"


def _load_json(path: Path) -> Dict[str, Any]:
    return path.read_text(encoding="utf-8")


_CACHE: Dict[str, List[str]] | None = None


def _load_knowledge() -> Dict[str, List[str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    import json

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


def _match_any(text: str, items: List[str]) -> Optional[str]:
    if not text:
        return None
    # Prefer longer matches first to avoid prefix hits like G4_C matching G4_Cu.
    ordered = sorted((it for it in items if it), key=len, reverse=True)
    for item in ordered:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])", re.IGNORECASE)
        if pat.search(text):
            return item
    return None


def _pick_structure_model() -> str:
    candidates = [
        MODELS_DIR / "structure_controlled_v4c_e1",
        MODELS_DIR / "structure_controlled_v3_e1",
        MODELS_DIR / "structure_controlled_smoke",
        MODELS_DIR / "structure_opt_v3",
        MODELS_DIR / "structure_opt_v2",
        MODELS_DIR / "structure",
    ]
    for p in candidates:
        if (p / "config.json").exists():
            return str(p)
    return "nlu/bert_lab/models/structure_controlled_v4c_e1"


def _pick_ner_model() -> str:
    candidates = [
        MODELS_DIR / "ner",
    ]
    for p in candidates:
        if (p / "config.json").exists():
            return str(p)
    return "nlu/bert_lab/models/ner"


def _pick_multitask_model() -> Optional[str]:
    p = MODELS_DIR / "multitask"
    if (p / "model.safetensors").exists() and (p / "tokenizer.json").exists():
        return str(p)
    return None


def _looks_non_english(text: str) -> bool:
    if not text:
        return False
    # Fast-path: any CJK character means we should rely on LLM normalization first.
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    ascii_alpha = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii == 0:
        return False
    total = max(1, len(text))
    return (ascii_alpha / total) < 0.25


def _normalize_text_with_llm(
    text: str,
    *,
    enable: bool = False,
    config_path: str = "nlu/bert_lab/configs/ollama_config.json",
) -> Tuple[str, Dict[str, Any]]:
    if not enable:
        return text, {"enabled": False, "used": False}
    prompt = build_normalization_prompt(text)
    try:
        resp = chat(prompt, config_path=config_path, temperature=0.0)
        parsed = extract_json(resp.get("response", ""))
        if isinstance(parsed, dict):
            normalized = str(parsed.get("normalized_text", "")).strip()
            if normalized:
                return normalized, {
                    "enabled": True,
                    "used": True,
                    "language_detected": parsed.get("language_detected", ""),
                    "structure_hint": parsed.get("structure_hint", ""),
                }
    except Exception as ex:
        return text, {"enabled": True, "used": False, "error": str(ex)}
    return text, {"enabled": True, "used": False}


def _sanitize_structure_hint(hint: Any) -> str:
    val = str(hint or "").strip().lower()
    allowed = {
        "ring",
        "grid",
        "nest",
        "stack",
        "shell",
        "single_box",
        "single_tubs",
        "unknown",
    }
    return val if val in allowed else ""


def extract_semantic_frame(
    text: str,
    min_confidence: float = 0.6,
    device: str = "auto",
    normalize_with_llm: bool = False,
    normalize_config_path: str = "nlu/bert_lab/configs/ollama_config.json",
) -> Tuple[SemanticFrame, Dict[str, Any]]:
    """
    Produce a SemanticFrame from natural language.
    - Geometry structure + params from BERT
    - Materials / particles / physics list / output format from knowledge lists
    """
    frame = SemanticFrame()
    debug: Dict[str, Any] = {}

    normalized_text, normalize_meta = _normalize_text_with_llm(
        text,
        enable=normalize_with_llm,
        config_path=normalize_config_path,
    )

    normalization_used = bool(normalize_meta.get("used", False))
    structure_hint = _sanitize_structure_hint(normalize_meta.get("structure_hint", ""))
    if _looks_non_english(normalized_text) and not normalization_used:
        frame.notes.append("non_english_input_requires_llm_normalization")
        debug["scores"] = {}
        debug["ranked"] = []
        debug["structure_model"] = _pick_structure_model()
        debug["ner_model"] = _pick_ner_model()
        debug["multitask_model"] = _pick_multitask_model()
        debug["inference_backend"] = "deferred_non_english"
        debug["input_text"] = text
        debug["normalized_text"] = normalized_text
        debug["normalization"] = normalize_meta
        debug["normalization_degraded"] = bool(normalize_with_llm and not normalization_used)
        debug["requires_llm_normalization"] = True
        return frame, debug

    structure_model = _pick_structure_model()
    ner_model = _pick_ner_model()
    multitask_model = _pick_multitask_model()
    used_backend = "candidate_graph_solver"
    mt_entities: Dict[str, str] = {}
    if multitask_model:
        try:
            mt = predict_multitask(
                normalized_text,
                model_dir=multitask_model,
                device=device,
                min_confidence=min_confidence,
            )
            params = dict(mt.get("params", {}))
            mt_entities = dict(mt.get("entities", {}))
            used_backend = "candidate_graph_solver+multitask_entities"
        except Exception as ex:
            params = extract_params(normalized_text, ner_model, device)
            debug["multitask_error"] = str(ex)
    else:
        params = extract_params(normalized_text, ner_model, device)
    params, notes = merge_params(normalized_text, params)
    graph_result = search_candidate_graphs(
        normalized_text,
        params,
        min_confidence=min_confidence,
        seed=7,
        top_k=3,
        apply_autofix=False,
    )
    structure = graph_result.structure
    scores = graph_result.scores
    ranked = graph_result.ranked
    notes.extend(graph_result.notes)

    if structure != "unknown":
        frame.geometry.structure = structure
        frame.geometry.chosen_skeleton = graph_result.chosen_skeleton or None
        frame.geometry.graph_program = graph_result.graph_program
    elif structure_hint and structure_hint != "unknown":
        frame.geometry.structure = structure_hint
        frame.notes.append("llm_structure_hint_applied")
        frame.geometry.chosen_skeleton = None
        frame.geometry.graph_program = None
    frame.geometry.params = params
    frame.notes.extend(notes)

    knowledge = _load_knowledge()
    mat = _match_any(mt_entities.get("material", ""), knowledge["materials"]) or _match_any(
        normalized_text, knowledge["materials"]
    )
    if mat:
        frame.materials.selected_materials.append(mat)

    phys = _match_any(mt_entities.get("physics_list", ""), knowledge["physics_lists"]) or _match_any(
        normalized_text, knowledge["physics_lists"]
    )
    if phys:
        frame.physics.physics_list = phys

    particle = _match_any(mt_entities.get("particle", ""), knowledge["particles"]) or _match_any(
        normalized_text, knowledge["particles"]
    )
    if particle:
        frame.source.particle = particle

    source_type = _match_any(mt_entities.get("source_type", ""), knowledge["source_types"]) or _match_any(
        normalized_text, knowledge["source_types"]
    )
    if source_type:
        frame.source.type = source_type

    out_fmt = _match_any(mt_entities.get("output_format", ""), knowledge["output_formats"]) or _match_any(
        normalized_text, knowledge["output_formats"]
    )
    if out_fmt:
        frame.output.format = out_fmt

    debug["scores"] = scores
    debug["ranked"] = ranked
    debug["graph_candidates"] = [
        {
            "structure": c.structure,
            "summary": c.summary,
            "score": c.score,
            "score_breakdown": c.score_breakdown,
            "feasible": c.feasible,
            "missing_params": c.missing_params,
            "errors": c.errors,
            "warnings": c.warnings,
            "dsl": c.dsl,
        }
        for c in graph_result.candidates
    ]
    debug["graph_chosen_skeleton"] = graph_result.chosen_skeleton
    debug["graph_program"] = graph_result.graph_program
    debug["structure_model"] = structure_model
    debug["ner_model"] = ner_model
    debug["multitask_model"] = multitask_model
    debug["inference_backend"] = used_backend
    debug["input_text"] = text
    debug["normalized_text"] = normalized_text
    debug["normalization"] = normalize_meta
    debug["normalization_degraded"] = bool(
        normalize_with_llm and not bool(normalize_meta.get("used", False))
    )
    return frame, debug


def frame_to_dict(frame: SemanticFrame) -> Dict[str, Any]:
    return asdict(frame)

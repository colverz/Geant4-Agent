from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nlu.bert_lab.semantic import extract_semantic_frame
from nlu.bert_lab.llm_bridge import build_missing_params_prompt, build_missing_params_schema
from nlu.bert_lab.ollama_client import chat, extract_json
from planner.agent import ask_missing
from builder.geometry.synthesize import synthesize_from_params


ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT.parent.parent / "knowledge" / "data"
SCHEMA_PATH = ROOT.parent.parent / "core" / "schema" / "geant4_min_config.schema.json"
OLLAMA_CONFIG_DEFAULT = os.getenv("OLLAMA_CONFIG_PATH", "nlu/bert_lab/configs/ollama_config.json")


@dataclass
class SessionState:
    history: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    last_question: str = ""


PHASE_ORDER = [
    ("geometry_core", ("geometry.structure",)),
    ("geometry_params", ("geometry.params.",)),
    ("materials", ("materials.",)),
    ("source_core", ("source.particle", "source.type")),
    ("source_kinematics", ("source.energy_MeV", "source.position", "source.direction")),
    ("physics", ("physics_list.",)),
    ("output", ("output.",)),
]

PHASE_TITLES = {
    "geometry_core": {"en": "Geometry Structure", "zh": "几何结构"},
    "geometry_params": {"en": "Geometry Parameters", "zh": "几何参数"},
    "materials": {"en": "Materials", "zh": "材料"},
    "source_core": {"en": "Source Core", "zh": "源定义"},
    "source_kinematics": {"en": "Source Kinematics", "zh": "源运动学"},
    "physics": {"en": "Physics List", "zh": "物理过程"},
    "output": {"en": "Output", "zh": "输出"},
    "complete": {"en": "Complete", "zh": "已完成"},
}


SESSIONS: Dict[str, SessionState] = {}


def _respond(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _load_file(path: Path) -> bytes:
    return path.read_bytes()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_knowledge() -> Dict[str, List[str]]:
    materials = _load_json(KNOWLEDGE_DIR / "materials_geant4_nist.json").get("materials", [])
    physics_lists = _load_json(KNOWLEDGE_DIR / "physics_lists.json").get("items", [])
    particles = _load_json(KNOWLEDGE_DIR / "particles.json").get("items", [])
    sources = _load_json(KNOWLEDGE_DIR / "source_constraints.json").get("types", [])
    if not sources:
        sources = ["point", "beam", "plane", "isotropic"]
    output_formats = _load_json(KNOWLEDGE_DIR / "output_formats.json").get("items", [])
    return {
        "materials": materials,
        "physics_lists": physics_lists,
        "particles": particles,
        "sources": sources,
        "output_formats": output_formats,
    }


KNOWLEDGE = _load_knowledge()


def _match_any(text: str, items: List[str]) -> Optional[str]:
    if not text:
        return None
    ordered = sorted((it for it in items if it), key=len, reverse=True)
    for item in ordered:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])", re.IGNORECASE)
        if pat.search(text):
            return item
    return None


def _default_config() -> Dict[str, Any]:
    return {
        "geometry": {
            "graph_program": None,
            "chosen_skeleton": None,
            "structure": None,
            "params": {},
            "dsl": None,
            "feasible": None,
            "errors": [],
        },
        "materials": {
            "selected_materials": [],
            "volume_material_map": {},
        },
        "source": {
            "type": None,
            "particle": None,
            "energy": None,
            "position": None,
            "direction": None,
        },
        "physics": {
            "physics_list": None,
        },
        "environment": {
            "temperature": None,
            "pressure": None,
        },
        "output": {
            "format": None,
            "path": None,
        },
        "notes": [],
    }


def _ensure_session(session_id: Optional[str]) -> Tuple[str, SessionState]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    SESSIONS[sid] = SessionState(config=_default_config())
    return sid, SESSIONS[sid]


def _heuristic_focus(text: str) -> List[str]:
    low = text.lower()
    focus = []
    geom_keys = ["box", "tubs", "ring", "grid", "stack", "nest", "shell", "cylinder", "sphere"]
    if any(k in low for k in geom_keys):
        focus.append("geometry")
    if any(k in low for k in ["material", "steel", "aluminum", "g4_"]):
        focus.append("materials")
    if any(k in low for k in ["source", "beam", "gamma", "electron", "proton", "neutron"]):
        focus.append("source")
    if any(k in low for k in ["physics list", "ftfp", "qgsp", "qb"]):
        focus.append("physics")
    return focus or ["geometry"]


def _infer_geometry_hint(text: str) -> Optional[str]:
    low = text.lower()
    if any(k in low for k in ["cube", "box", "立方体", "长方体"]):
        return "single_box"
    if any(k in low for k in ["cylinder", "tubs", "圆柱"]):
        return "single_tubs"
    return None


def _route_with_llm(text: str, missing_fields: List[str]) -> Dict[str, Any]:
    prompt = (
        "You are a router that decides which modules to run next in a Geant4 config builder.\n"
        "Return JSON with keys: use_geometry_bert (bool), focus (list of strings: geometry, materials, source, physics), "
        "geometry_hint (optional: single_box or single_tubs), reason (string).\n"
        f"User text: {text}\n"
        f"Missing fields: {missing_fields}\n"
        "JSON:"
    )
    try:
        resp = chat(prompt, config_path=OLLAMA_CONFIG_DEFAULT, temperature=0.0)
        parsed = extract_json(resp.get("response", "")) or {}
        if isinstance(parsed, dict) and "focus" in parsed:
            return parsed
    except Exception:
        return {}
    return {}


def _decide_focus(text: str, missing_fields: List[str], llm_router: bool) -> Dict[str, Any]:
    if llm_router:
        routed = _route_with_llm(text, missing_fields)
        if routed:
            return routed
    focus = _heuristic_focus(text)
    hint = _infer_geometry_hint(text)
    return {
        "use_geometry_bert": "geometry" in focus,
        "focus": focus,
        "geometry_hint": hint,
        "reason": "heuristic",
    }


def _apply_frame(
    config: Dict[str, Any],
    frame: Any,
    debug: Dict[str, Any],
    autofix: bool,
    geometry_hint: Optional[str],
) -> Dict[str, Any]:
    # Merge geometry
    structure = frame.geometry.structure
    if not structure and geometry_hint:
        structure = geometry_hint
    if structure:
        config["geometry"]["structure"] = structure
    if getattr(frame.geometry, "chosen_skeleton", None):
        config["geometry"]["chosen_skeleton"] = frame.geometry.chosen_skeleton
    if frame.geometry.params:
        config["geometry"]["params"].update(frame.geometry.params)
    if getattr(frame.geometry, "graph_program", None):
        config["geometry"]["graph_program"] = frame.geometry.graph_program

    if config["geometry"].get("graph_program"):
        config["geometry"]["dsl"] = config["geometry"]["graph_program"]
        chosen = debug.get("graph_chosen_skeleton", "")
        gcands = debug.get("graph_candidates", [])
        cobj = None
        for c in gcands:
            if c.get("structure") == chosen:
                cobj = c
                break
        if cobj is not None:
            config["geometry"]["feasible"] = bool(cobj.get("feasible", False))
            config["geometry"]["errors"] = list(cobj.get("errors", []))
            config["geometry"]["warnings"] = list(cobj.get("warnings", []))
            config["geometry"]["missing_params"] = list(cobj.get("missing_params", []))
        else:
            config["geometry"]["feasible"] = None
            config["geometry"]["errors"] = []
            config["geometry"]["warnings"] = []
            config["geometry"]["missing_params"] = []
        config["geometry"]["scores"] = debug.get("scores", {})
        config["geometry"]["ranked"] = debug.get("ranked", [])
    elif config["geometry"]["structure"]:
        synth = synthesize_from_params(
            config["geometry"]["structure"],
            config["geometry"]["params"],
            seed=7,
            apply_autofix=autofix,
        )
        config["geometry"]["dsl"] = synth.get("dsl")
        config["geometry"]["feasible"] = synth.get("feasible")
        config["geometry"]["errors"] = synth.get("errors", [])
        config["geometry"]["missing_params"] = synth.get("missing_params", [])
        config["geometry"]["scores"] = debug.get("scores", {})
        config["geometry"]["ranked"] = debug.get("ranked", [])
    else:
        config["geometry"]["dsl"] = None
        config["geometry"]["feasible"] = None
        config["geometry"]["errors"] = ["structure confidence below threshold"]
        config["geometry"]["missing_params"] = []
        config["geometry"]["scores"] = debug.get("scores", {})
        config["geometry"]["ranked"] = debug.get("ranked", [])

    # Merge materials / source / physics / output
    for mat in frame.materials.selected_materials:
        if mat not in config["materials"]["selected_materials"]:
            config["materials"]["selected_materials"].append(mat)
    if frame.physics.physics_list:
        config["physics"]["physics_list"] = frame.physics.physics_list
    if frame.source.particle:
        config["source"]["particle"] = frame.source.particle
    if frame.source.type:
        config["source"]["type"] = frame.source.type
    if frame.output.format:
        config["output"]["format"] = frame.output.format
    config["notes"].extend(getattr(frame, "notes", []))
    return config


def _parse_source_energy_mev(text: str) -> Optional[float]:
    import re

    m = re.search(r"([-+]?\d*\.?\d+)\s*(mev|gev|kev)\b", text.lower())
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "gev":
        return value * 1000.0
    if unit == "kev":
        return value * 0.001
    return value


def _parse_triplet(text: str, key: str) -> Optional[Dict[str, Any]]:
    import re

    pat = rf"{key}\s*[:=]?\s*\(?\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*\)?"
    m = re.search(pat, text.lower())
    if not m:
        return None
    return {"type": "vector", "value": [float(m.group(1)), float(m.group(2)), float(m.group(3))]}


def _parse_output_path(text: str) -> Optional[str]:
    import re

    m = re.search(r"([A-Za-z]:[\\/][^\\s]+\\.(?:root|csv|json)|[./\\w-]+\\.(?:root|csv|json))", text)
    if not m:
        return None
    return m.group(1)


def _apply_text_overrides(config: Dict[str, Any], text: str) -> None:
    low = text.lower()
    energy = _parse_source_energy_mev(text)
    if energy is not None:
        config["source"]["energy"] = float(energy)
    pos = _parse_triplet(text, "position")
    if pos:
        config["source"]["position"] = pos
    direction = _parse_triplet(text, "direction")
    if direction:
        config["source"]["direction"] = direction

    if "point source" in low or "point-like" in low:
        config["source"]["type"] = "point"
    elif "beam" in low:
        config["source"]["type"] = "beam"
    elif "isotropic" in low:
        config["source"]["type"] = "isotropic"

    out_path = _parse_output_path(text)
    if out_path:
        config["output"]["path"] = out_path
    if not config["output"].get("path") and config["output"].get("format"):
        fmt = str(config["output"]["format"]).strip().lower()
        if fmt:
            config["output"]["path"] = f"output/result.{fmt}"


def _export_min_config(config: Dict[str, Any]) -> Dict[str, Any]:
    vmap = config["materials"].get("volume_material_map", {}) or {}
    if isinstance(vmap, dict):
        volume_material_map = [{"volume": k, "material": v} for k, v in vmap.items() if k and v]
    elif isinstance(vmap, list):
        volume_material_map = vmap
    else:
        volume_material_map = []
    if not volume_material_map and config["materials"].get("selected_materials"):
        volume_material_map = [{"volume": "target", "material": config["materials"]["selected_materials"][0]}]

    return {
        "geometry": {
            "graph_program": config["geometry"].get("graph_program"),
            "structure": config["geometry"].get("structure"),
            "params": config["geometry"].get("params", {}),
        },
        "materials": {
            "volume_material_map": volume_material_map,
            "environment": {
                "temperature_K": config["environment"].get("temperature"),
                "pressure_Pa": config["environment"].get("pressure"),
            },
        },
        "physics_list": {
            "name": config["physics"].get("physics_list"),
        },
        "source": {
            "particle": config["source"].get("particle"),
            "energy_MeV": config["source"].get("energy"),
            "position": config["source"].get("position"),
            "direction": config["source"].get("direction"),
        },
        "output": {
            "format": config["output"].get("format"),
            "path": config["output"].get("path"),
        },
    }


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (dict, list)) and len(value) == 0:
        return True
    return False


def _collect_required_missing(schema: Dict[str, Any], obj: Any, prefix: str = "") -> List[str]:
    missing: List[str] = []
    if not isinstance(schema, dict):
        return missing
    if schema.get("type") != "object":
        return missing

    props = schema.get("properties", {})
    required = schema.get("required", [])
    value = obj if isinstance(obj, dict) else {}

    for key in required:
        path = f"{prefix}.{key}" if prefix else key
        if key not in value or _is_missing_value(value.get(key)):
            missing.append(path)

    for key, sub_schema in props.items():
        if key in value and value.get(key) is not None:
            path = f"{prefix}.{key}" if prefix else key
            missing.extend(_collect_required_missing(sub_schema, value.get(key), path))

    return missing


def _select_phase_fields(missing_fields: List[str]) -> Tuple[str, List[str]]:
    for phase, patterns in PHASE_ORDER:
        selected = []
        for item in missing_fields:
            for p in patterns:
                if p.endswith("."):
                    if item.startswith(p):
                        selected.append(item)
                        break
                else:
                    if item == p:
                        selected.append(item)
                        break
        if selected:
            return phase, selected
    return "complete", []


def _friendly_fields(fields: List[str], lang: str) -> List[str]:
    if lang == "zh":
        mapping = {
            "geometry.structure": "几何结构类型",
            "materials.volume_material_map": "体积-材料映射",
            "source.particle": "粒子类型",
            "source.type": "源类型",
            "source.energy_MeV": "能量 (MeV)",
            "source.position": "源位置",
            "source.direction": "源方向",
            "physics_list.name": "物理过程列表",
            "output.format": "输出格式",
            "output.path": "输出路径",
        }
    else:
        mapping = {
            "geometry.structure": "geometry structure",
            "materials.volume_material_map": "volume-material map",
            "source.particle": "particle type",
            "source.type": "source type",
            "source.energy_MeV": "energy (MeV)",
            "source.position": "source position",
            "source.direction": "source direction",
            "physics_list.name": "physics list",
            "output.format": "output format",
            "output.path": "output path",
        }
    out: List[str] = []
    for f in fields:
        if f.startswith("geometry.params."):
            key = f.split("geometry.params.", 1)[1]
            out.append(f"几何参数 {key}" if lang == "zh" else f"geometry param {key}")
        else:
            out.append(mapping.get(f, f))
    return out


def _is_complete(config: Dict[str, Any], missing_fields: List[str]) -> bool:
    if missing_fields:
        return False
    feasible = config.get("geometry", {}).get("feasible")
    if feasible is False:
        return False
    return True


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, p))
        return out
    out[prefix] = obj
    return out


def _diff_paths(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    b = _flatten(before)
    a = _flatten(after)
    keys = sorted(set(b.keys()) | set(a.keys()))
    changed: List[str] = []
    for k in keys:
        if b.get(k) != a.get(k):
            changed.append(k)
    return changed


def _compute_missing(config: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if config["geometry"].get("structure") and config["geometry"].get("missing_params"):
        for p in config["geometry"]["missing_params"]:
            missing.append(f"geometry.params.{p}")
    try:
        schema = _load_json(SCHEMA_PATH)
        exported = _export_min_config(config)
        missing.extend(_collect_required_missing(schema, exported))
    except Exception:
        pass
    dedup: List[str] = []
    seen = set()
    for m in missing:
        if m not in seen:
            dedup.append(m)
            seen.add(m)
    return dedup


def _build_user_friendly(config: Dict[str, Any]) -> str:
    geo = config["geometry"]
    geo_label = geo.get("structure") or (geo.get("chosen_skeleton") or "unknown")
    return "\n".join(
        [
            f"Geometry: {geo_label}",
            f"Feasible: {geo.get('feasible')}",
            f"Materials: {', '.join(config['materials']['selected_materials']) or 'missing'}",
            f"Particle: {config['source']['particle'] or 'missing'}",
            f"Source type: {config['source']['type'] or 'missing'}",
            f"Physics list: {config['physics']['physics_list'] or 'missing'}",
            f"Output format: {config['output']['format'] or 'missing'}",
            f"Output path: {config['output'].get('path') or 'missing'}",
        ]
    )


def _ask_llm(
    asked_fields: List[str],
    asked_fields_friendly: List[str],
    history: List[Dict[str, str]],
    lang: str,
    use_llm: bool = True,
) -> str:
    if not asked_fields:
        return ""
    if not use_llm:
        if lang == "zh":
            return "请补充以下信息: " + ", ".join(asked_fields_friendly)
        return "Please provide: " + ", ".join(asked_fields_friendly)
    return ask_missing(asked_fields, lang, OLLAMA_CONFIG_DEFAULT, temperature=0.2)


def step(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    session_id = payload.get("session_id")
    llm_router = bool(payload.get("llm_router", True))
    llm_question = bool(payload.get("llm_question", True))
    normalize_input = bool(payload.get("normalize_input", True))
    min_conf = float(payload.get("min_confidence", 0.6))
    autofix = bool(payload.get("autofix", False))
    lang = str(payload.get("lang", "zh")).lower()

    sid, state = _ensure_session(session_id)
    if not text:
        return {"error": "missing text", "session_id": sid}

    state.history.append({"role": "user", "content": text})
    routed = _decide_focus(text, state.missing_fields, llm_router)
    prev_exported = _export_min_config(state.config)

    frame, debug = extract_semantic_frame(
        text,
        min_confidence=min_conf,
        device="auto",
        normalize_with_llm=normalize_input,
        normalize_config_path=OLLAMA_CONFIG_DEFAULT,
    )
    needs_norm = bool(debug.get("requires_llm_normalization", False))
    if needs_norm:
        msg = (
            "请先启用/修复 LLM 归一化（Ollama），再继续提取参数。"
            if lang == "zh"
            else "Please enable/fix LLM normalization (Ollama) before parameter extraction."
        )
        state.history.append({"role": "assistant", "content": msg})
        state.last_question = msg
        return {
            "session_id": sid,
            "router": routed,
            "inference_backend": debug.get("inference_backend", "deferred_non_english"),
            "normalized_text": debug.get("normalized_text", text),
            "normalization": debug.get("normalization", {}),
            "normalization_degraded": bool(debug.get("normalization_degraded", False)),
            "requires_llm_normalization": True,
            "missing_fields": state.missing_fields,
            "assistant_message": msg,
            "phase": "normalization",
            "phase_title": "输入归一化" if lang == "zh" else "Input Normalization",
            "asked_fields": [],
            "asked_fields_friendly": [],
            "is_complete": False,
            "delta_paths": [],
            "display": _build_user_friendly(state.config),
            "config": state.config,
            "config_min": _export_min_config(state.config),
            "history": state.history[-10:],
        }
    if not routed.get("use_geometry_bert", True):
        frame.geometry.structure = None
        frame.geometry.params = {}
    state.config = _apply_frame(
        state.config,
        frame,
        debug,
        autofix,
        routed.get("geometry_hint") or _infer_geometry_hint(text),
    )
    _apply_text_overrides(state.config, text)
    state.missing_fields = _compute_missing(state.config)
    phase, asked_fields = _select_phase_fields(state.missing_fields)
    asked_fields_friendly = _friendly_fields(asked_fields, lang)
    complete = _is_complete(state.config, state.missing_fields)
    delta_paths = _diff_paths(prev_exported, _export_min_config(state.config))

    question = _ask_llm(asked_fields, asked_fields_friendly, state.history, lang, use_llm=llm_question)
    if complete:
        question = "配置已完成。" if lang == "zh" else "Configuration complete."
    if question:
        state.history.append({"role": "assistant", "content": question})
        state.last_question = question
    else:
        state.last_question = ""

    return {
        "session_id": sid,
        "router": routed,
        "inference_backend": debug.get("inference_backend", "dual_model"),
        "normalized_text": debug.get("normalized_text", text),
        "normalization": debug.get("normalization", {}),
        "normalization_degraded": bool(debug.get("normalization_degraded", False)),
        "missing_fields": state.missing_fields,
        "assistant_message": question or "Configuration complete.",
        "phase": phase,
        "phase_title": PHASE_TITLES.get(phase, {}).get(lang, phase),
        "asked_fields": asked_fields,
        "asked_fields_friendly": asked_fields_friendly,
        "is_complete": complete,
        "delta_paths": delta_paths,
        "graph_candidates": debug.get("graph_candidates", []),
        "display": _build_user_friendly(state.config),
        "config": state.config,
        "config_min": _export_min_config(state.config),
        "history": state.history[-10:],
    }


def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"error": "missing text"}

    top_k = int(payload.get("top_k", 1))
    min_conf = float(payload.get("min_confidence", 0.6))
    normalize_input = bool(payload.get("normalize_input", True))
    prompt_format = payload.get("prompt_format", "json_schema")
    autofix = bool(payload.get("autofix", False))
    llm_fill = bool(payload.get("llm_fill_missing", False))
    params_override = payload.get("params_override", {}) or {}

    frame, debug = extract_semantic_frame(
        text,
        min_confidence=min_conf,
        device="auto",
        normalize_with_llm=normalize_input,
        normalize_config_path=OLLAMA_CONFIG_DEFAULT,
    )
    if bool(debug.get("requires_llm_normalization", False)):
        return {
            "structure": "unknown",
            "inference_backend": debug.get("inference_backend", "deferred_non_english"),
            "normalized_text": debug.get("normalized_text", text),
            "normalization": debug.get("normalization", {}),
            "normalization_degraded": bool(debug.get("normalization_degraded", False)),
            "requires_llm_normalization": True,
            "scores": {},
            "params": {},
            "notes": list(frame.notes),
            "synthesis": {"error": "requires_llm_normalization"},
            "missing_prompt": "",
            "missing_schema": None,
            "candidates": [],
            "best_candidate": None,
        }
    structure = frame.geometry.structure or "unknown"
    params = dict(frame.geometry.params)
    params.update(params_override)
    notes = list(frame.notes)
    scores = debug.get("scores", {})
    ranked = debug.get("ranked", [])

    candidates = []
    for name, prob in ranked[: max(1, top_k)]:
        if prob < min_conf:
            continue
        synth = synthesize_from_params(name, params, seed=7, apply_autofix=autofix)
        missing = synth.get("missing_params", [])
        prompt = build_missing_params_prompt(name, missing, fmt=prompt_format)
        schema = build_missing_params_schema(name, missing) if prompt_format == "json_schema" else None
        filled = None
        if llm_fill and missing:
            resp = chat(prompt, config_path=OLLAMA_CONFIG_DEFAULT, temperature=0.2)
            parsed = extract_json(resp.get("response", ""))
            if isinstance(parsed, dict):
                merged = dict(params)
                merged.update(parsed)
                filled = synthesize_from_params(name, merged, seed=7, apply_autofix=autofix)
        candidates.append(
            {
                "structure": name,
                "prob": prob,
                "synthesis": synth,
                "synthesis_filled": filled,
                "missing_prompt": prompt,
                "missing_schema": schema,
            }
        )

    if structure == "unknown":
        synthesis = {"error": "structure confidence below threshold"}
        missing_prompt = ""
        missing_schema = None
    else:
        synthesis = synthesize_from_params(structure, params, seed=7, apply_autofix=autofix)
        missing = synthesis.get("missing_params", [])
        missing_prompt = build_missing_params_prompt(structure, missing, fmt=prompt_format)
        missing_schema = build_missing_params_schema(structure, missing) if prompt_format == "json_schema" else None

    def _cand_score(c):
        synth = c.get("synthesis", {})
        feasible = bool(synth.get("feasible"))
        missing_n = len(synth.get("missing_params", []))
        errors_n = len(synth.get("errors", []))
        return (1 if feasible else 0, -missing_n, -errors_n, c.get("prob", 0.0))

    best = None
    if candidates:
        best = sorted(candidates, key=_cand_score, reverse=True)[0]

    return {
        "structure": structure,
        "inference_backend": debug.get("inference_backend", "dual_model"),
        "normalized_text": debug.get("normalized_text", text),
        "normalization": debug.get("normalization", {}),
        "normalization_degraded": bool(debug.get("normalization_degraded", False)),
        "scores": scores,
        "params": params,
        "notes": notes,
        "synthesis": synthesis,
        "missing_prompt": missing_prompt,
        "missing_schema": missing_schema,
        "candidates": candidates,
        "graph_candidates": debug.get("graph_candidates", []),
        "best_candidate": best,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.strip("/")
        if path == "":
            data = _load_file(ROOT / "index.html")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path in {"style.css", "app.js"}:
            data = _load_file(ROOT / path)
            mime = "text/css" if path.endswith(".css") else "application/javascript"
            self.send_response(200)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path not in {"/api/solve", "/api/step", "/api/reset"}:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            _respond(self, 400, {"error": "invalid json"})
            return
        if self.path == "/api/solve":
            out = solve(payload)
        elif self.path == "/api/reset":
            session_id = payload.get("session_id")
            if session_id in SESSIONS:
                del SESSIONS[session_id]
            out = {"ok": True}
        else:
            out = step(payload)
        _respond(self, 200, out)


def main() -> None:
    host = "127.0.0.1"
    port = 8088
    print(f"Serving on http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()







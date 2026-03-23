from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from builder.geometry.synthesize import synthesize_from_params
from core.config.defaults import build_legacy_default_config
from core.config.field_registry import friendly_labels
from core.config.phase_registry import phase_title, select_phase_fields
from core.config.prompt_registry import clarification_fallback, completion_message
from nlu.llm_support.llm_bridge import build_missing_params_prompt, build_missing_params_schema
from nlu.llm_support.ollama_client import chat, extract_json
from planner.agent import ask_missing
from ui.web.runtime_state import get_ollama_config_path

ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT.parent.parent / "knowledge" / "data"
SCHEMA_PATH = ROOT.parent.parent / "core" / "schema" / "geant4_min_config.schema.json"


@dataclass
class SessionState:
    history: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    last_question: str = ""


SESSIONS: Dict[str, SessionState] = {}


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


def _extract_semantic_frame_legacy(*args, **kwargs):
    from nlu.bert_lab.semantic import extract_semantic_frame

    return extract_semantic_frame(*args, **kwargs)


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
    return build_legacy_default_config()


def _ensure_session(session_id: Optional[str]) -> Tuple[str, SessionState]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    SESSIONS[sid] = SessionState(config=_default_config())
    return sid, SESSIONS[sid]


def _build_context_summary(config: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    geo = config.get("geometry", {})
    mats = config.get("materials", {})
    src = config.get("source", {})
    phy = config.get("physics", {})
    out = config.get("output", {})

    last_user = ""
    for item in reversed(history):
        if item.get("role") == "user":
            last_user = str(item.get("content", "")).strip()
            if last_user:
                break

    parts = [
        f"structure: {geo.get('structure') or ''}",
        f"geometry_params: {json.dumps(geo.get('params', {}), ensure_ascii=False)}",
        f"materials: {', '.join(mats.get('selected_materials', []))}",
        f"particle: {src.get('particle') or ''}",
        f"source_type: {src.get('type') or ''}",
        f"source_energy_MeV: {src.get('energy') if src.get('energy') is not None else ''}",
        f"source_position: {json.dumps(src.get('position'), ensure_ascii=False) if src.get('position') else ''}",
        f"source_direction: {json.dumps(src.get('direction'), ensure_ascii=False) if src.get('direction') else ''}",
        f"physics_list: {phy.get('physics_list') or ''}",
        f"output_format: {out.get('format') or ''}",
        f"output_path: {out.get('path') or ''}",
        f"last_user_turn: {last_user}",
    ]
    return "; ".join(parts)


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
    if any(k in low for k in ["cube", "box", "绔嬫柟浣?, "闀挎柟浣?]):
        return "single_box"
    if any(k in low for k in ["cylinder", "tubs", "鍦嗘煴"]):
        return "single_tubs"
    return None


def _has_explicit_geometry_assignment(text: str) -> bool:
    low = text.lower()
    if re.search(r"\b(structure|geometry)\s*[:=]", low):
        return True
    if re.search(r"(鍑犱綍|缁撴瀯)\s*[:=]", text):
        return True

    shape_tokens = (
        "ring",
        "grid",
        "stack",
        "nest",
        "shell",
        "box",
        "cube",
        "cylinder",
        "tubs",
        "sphere",
        "cons",
        "trd",
        "polycone",
        "cuttubs",
        "boolean",
        "鐜舰",
        "闃靛垪",
        "鍫嗗彔",
        "宓屽",
        "澹冲眰",
        "绔嬫柟浣?,
        "鍦嗘煴",
        "鐞?,
    )
    verb_tokens = (
        "use",
        "build",
        "create",
        "make",
        "set",
        "change",
        "switch",
        "update",
        "鏀规垚",
        "鏀逛负",
        "鎹㈡垚",
        "鏀圭敤",
        "璁剧疆",
    )
    has_shape = any(tok in low or tok in text for tok in shape_tokens)
    has_verb = any(tok in low or tok in text for tok in verb_tokens)
    return has_shape and has_verb


def _should_freeze_geometry_update(
    *,
    text: str,
    routed: Dict[str, Any],
    existing_structure: Optional[str],
    incoming_structure: Optional[str],
    incoming_params: Dict[str, float],
    previous_missing_fields: List[str],
) -> Tuple[bool, str]:
    if not existing_structure:
        return False, ""
    explicit_geo = _has_explicit_geometry_assignment(text)
    if explicit_geo:
        return False, ""

    focus = routed.get("focus", [])
    focus_set = {str(x).strip().lower() for x in focus if isinstance(x, str)}
    geom_missing_before = any(f.startswith("geometry.params.") for f in previous_missing_fields)

    if incoming_structure and incoming_structure != existing_structure:
        return True, "structure_change_without_explicit_request"
    if not geom_missing_before and "geometry" not in focus_set and (incoming_structure or incoming_params):
        return True, "non_geometry_turn_locked"
    return False, ""


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
        resp = chat(prompt, config_path=get_ollama_config_path(), temperature=0.0)
        parsed = extract_json(resp.get("response", "")) or {}
        if isinstance(parsed, dict) and "focus" in parsed:
            return parsed
    except Exception:
        return {}
    return {}


def _decide_focus(text: str, missing_fields: List[str], llm_router: bool) -> Dict[str, Any]:
    allowed_focus = {"geometry", "materials", "source", "physics"}

    def _sanitize_routed(routed: Dict[str, Any]) -> Dict[str, Any]:
        focus_raw = routed.get("focus", [])
        if not isinstance(focus_raw, list):
            focus_raw = []
        focus = []
        for x in focus_raw:
            if not isinstance(x, str):
                continue
            v = x.strip().lower()
            if v in allowed_focus and v not in focus:
                focus.append(v)
        if not focus:
            focus = _heuristic_focus(text)

        hint = str(routed.get("geometry_hint", "")).strip().lower()
        if hint not in {
            "",
            "ring",
            "grid",
            "nest",
            "stack",
            "shell",
            "single_box",
            "single_tubs",
            "single_sphere",
            "single_cons",
            "single_trd",
            "single_polycone",
            "single_cuttubs",
            "boolean",
            "unknown",
        }:
            hint = ""

        use_geo = routed.get("use_geometry_bert")
        if not isinstance(use_geo, bool):
            use_geo = "geometry" in focus
        return {
            "use_geometry_bert": use_geo,
            "focus": focus,
            "geometry_hint": hint,
            "reason": str(routed.get("reason", "llm")).strip(),
        }

    if llm_router:
        routed = _route_with_llm(text, missing_fields)
        if routed:
            return _sanitize_routed(routed)
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

    m = re.search(r"([-+]?\d*\.?\d+)\s*(mev|gev|kev)", text.lower())
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


def _infer_direction_from_text(text: str) -> Optional[Dict[str, Any]]:
    low = text.lower().replace(" ", "")
    mapping = [
        (("+z", "娌?z", "towards+z", "along+z"), [0.0, 0.0, 1.0]),
        (("-z", "娌?z", "towards-z", "along-z"), [0.0, 0.0, -1.0]),
        (("+x", "娌?x", "towards+x", "along+x"), [1.0, 0.0, 0.0]),
        (("-x", "娌?x", "towards-x", "along-x"), [-1.0, 0.0, 0.0]),
        (("+y", "娌?y", "towards+y", "along+y"), [0.0, 1.0, 0.0]),
        (("-y", "娌?y", "towards-y", "along-y"), [0.0, -1.0, 0.0]),
    ]
    for keys, vec in mapping:
        if any(k in low for k in keys):
            return {"type": "vector", "value": vec}
    return None


def _infer_position_from_text(text: str) -> Optional[Dict[str, Any]]:
    low = text.lower()
    if any(k in low for k in ["origin", "at origin", "鍘熺偣", "涓績", "center"]):
        return {"type": "vector", "value": [0.0, 0.0, 0.0]}
    return None


def _parse_output_path(text: str) -> Optional[str]:
    import re

    m = re.search(r"([A-Za-z]:[\\/][^\\s]+\\.(?:root|csv|json|xml|hdf5|h5)|[./\\w-]+\\.(?:root|csv|json|xml|hdf5|h5))", text)
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
    # Handle compact source style: "... at (x,y,z) to (dx,dy,dz)"
    at_to = re.search(
        r"\bat\s*\(?\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*\)?\s*"
        r"(?:to|towards|->)\s*"
        r"\(?\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*\)?",
        low,
    )
    if at_to:
        if not config["source"].get("position"):
            config["source"]["position"] = {
                "type": "vector",
                "value": [float(at_to.group(1)), float(at_to.group(2)), float(at_to.group(3))],
            }
        if not config["source"].get("direction"):
            config["source"]["direction"] = {
                "type": "vector",
                "value": [float(at_to.group(4)), float(at_to.group(5)), float(at_to.group(6))],
            }
    if not config["source"].get("position"):
        inferred_pos = _infer_position_from_text(text)
        if inferred_pos:
            config["source"]["position"] = inferred_pos
    if not config["source"].get("direction"):
        inferred_dir = _infer_direction_from_text(text)
        if inferred_dir:
            config["source"]["direction"] = inferred_dir

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


def _ensure_material_volume_map(config: Dict[str, Any]) -> None:
    mats = config.get("materials", {}).get("selected_materials", []) or []
    vmap = config.get("materials", {}).get("volume_material_map", {})
    if not isinstance(vmap, dict):
        return
    if vmap or not mats:
        return
    if not config.get("geometry", {}).get("structure"):
        return
    config["materials"]["volume_material_map"] = {"target": mats[0]}


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
    return select_phase_fields(missing_fields)


def _friendly_fields(fields: List[str], lang: str) -> List[str]:
    return friendly_labels(fields, lang)


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
        return clarification_fallback(asked_fields_friendly, lang)
    return ask_missing(asked_fields, lang, get_ollama_config_path(), temperature=1.0)


def _is_physics_recommend_request(text: str) -> bool:
    low = text.lower()
    physics_tokens = [
        "physics list",
        "physics_list",
        "鐗╃悊鍒楄〃",
        "鐗╃悊杩囩▼",
        "棰勭疆鐗╃悊",
        "棰勮鐗╃悊",
    ]
    decision_tokens = [
        "choose",
        "select",
        "recommend",
        "best",
        "most suitable",
        "pick",
        "閫夋嫨",
        "鎺ㄨ崘",
        "鏈€鍚堥€?,
        "缁欏嚭鍚嶇О",
        "澶囬€?,
    ]
    has_physics = any(t in low or t in text for t in physics_tokens)
    has_decision = any(t in low or t in text for t in decision_tokens)
    return has_physics and has_decision


def _pick_known_physics(name: str) -> Optional[str]:
    val = str(name or "").strip()
    if not val:
        return None
    known = KNOWLEDGE.get("physics_lists", [])
    for item in known:
        if item.lower() == val.lower():
            return item
    return None


def _recommend_physics_with_llm(text: str, context_summary: str, lang: str) -> Dict[str, Any]:
    choices = KNOWLEDGE.get("physics_lists", [])
    if not choices:
        return {}
    prompt = (
        "You are a Geant4 physics-list recommender.\n"
        "Pick the most suitable predefined Geant4 physics list for the request.\n"
        "Return JSON only with keys:\n"
        "- physics_list: string (must be one of allowed)\n"
        "- backup_physics_list: string (optional, from allowed)\n"
        "- reasons: array of short strings (max 3)\n"
        "- covered_processes: array of short strings\n"
        "- confidence: number in [0,1]\n"
        "Rules:\n"
        "- Use only allowed list names.\n"
        "- Do not invent new list names.\n"
        "- If request is mostly EM (gamma/e-/e+), prefer an option suitable for EM-focused studies.\n"
        f"Allowed physics lists: {', '.join(choices)}\n"
        f"Session context: {context_summary}\n"
        f"User request: {text}\n"
        "JSON:"
    )
    try:
        resp = chat(prompt, config_path=get_ollama_config_path(), temperature=0.0)
        raw = str(resp.get("response", ""))
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        parsed = extract_json(raw) or {}
        if not isinstance(parsed, dict):
            return {}
        main = _pick_known_physics(parsed.get("physics_list", ""))
        if not main:
            return {}
        backup = _pick_known_physics(parsed.get("backup_physics_list", "")) or None
        reasons = parsed.get("reasons", [])
        covered = parsed.get("covered_processes", [])
        conf = parsed.get("confidence", None)
        if not isinstance(reasons, list):
            reasons = []
        if not isinstance(covered, list):
            covered = []
        if not isinstance(conf, (int, float)):
            conf = None
        return {
            "physics_list": main,
            "backup_physics_list": backup,
            "reasons": [str(x).strip() for x in reasons if str(x).strip()][:3],
            "covered_processes": [str(x).strip() for x in covered if str(x).strip()][:6],
            "confidence": float(conf) if conf is not None else None,
            "lang": lang,
            "source": "llm_recommender",
        }
    except Exception:
        return {}


def legacy_step(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    prev_missing_fields = list(state.missing_fields)
    prev_exported = _export_min_config(state.config)
    context_summary = _build_context_summary(state.config, state.history[:-1])

    frame, debug = _extract_semantic_frame_legacy(
        text,
        min_confidence=min_conf,
        device="auto",
        normalize_with_llm=normalize_input,
        normalize_config_path=get_ollama_config_path(),
        context_summary=context_summary,
    )
    needs_norm = bool(debug.get("requires_llm_normalization", False))
    if needs_norm:
        msg = (
            "璇峰厛鍚敤/淇 LLM 褰掍竴鍖栵紙Ollama锛夛紝鍐嶇户缁彁鍙栧弬鏁般€?
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
            "phase_title": "杈撳叆褰掍竴鍖? if lang == "zh" else "Input Normalization",
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
    freeze_geo, freeze_reason = _should_freeze_geometry_update(
        text=text,
        routed=routed,
        existing_structure=state.config.get("geometry", {}).get("structure"),
        incoming_structure=frame.geometry.structure,
        incoming_params=frame.geometry.params,
        previous_missing_fields=prev_missing_fields,
    )
    if freeze_geo:
        frame.geometry.structure = None
        frame.geometry.params = {}
        frame.geometry.graph_program = None
        frame.geometry.chosen_skeleton = None
        frame.notes.append(f"geometry_update_skipped:{freeze_reason}")
    geometry_hint = routed.get("geometry_hint") or _infer_geometry_hint(text)
    if freeze_geo:
        geometry_hint = None
    # Do not allow implicit hint to overwrite an existing geometry unless user explicitly requests geometry change.
    if state.config.get("geometry", {}).get("structure") and not _has_explicit_geometry_assignment(text):
        geometry_hint = None
    state.config = _apply_frame(
        state.config,
        frame,
        debug,
        autofix,
        geometry_hint,
    )
    _apply_text_overrides(state.config, text)
    _ensure_material_volume_map(state.config)
    physics_recommendation: Dict[str, Any] = {}
    if _is_physics_recommend_request(text):
        physics_recommendation = _recommend_physics_with_llm(text, context_summary, lang)
        if physics_recommendation.get("physics_list"):
            state.config["physics"]["physics_list"] = physics_recommendation["physics_list"]
            state.config["physics"]["backup_physics_list"] = physics_recommendation.get("backup_physics_list")
            state.config["physics"]["selection_reasons"] = physics_recommendation.get("reasons", [])
            state.config["physics"]["covered_processes"] = physics_recommendation.get("covered_processes", [])
            state.config["physics"]["selection_source"] = "llm_recommender"
    state.missing_fields = _compute_missing(state.config)
    phase, asked_fields = _select_phase_fields(state.missing_fields)
    asked_fields_friendly = _friendly_fields(asked_fields, lang)
    complete = _is_complete(state.config, state.missing_fields)
    delta_paths = _diff_paths(prev_exported, _export_min_config(state.config))

    question = _ask_llm(asked_fields, asked_fields_friendly, state.history, lang, use_llm=llm_question)
    if complete:
        question = completion_message(lang)
    if question:
        state.history.append({"role": "assistant", "content": question})
        state.last_question = question
    else:
        state.last_question = ""

    return {
        "session_id": sid,
        "router": routed,
        "geometry_update_skipped": freeze_geo,
        "geometry_update_skipped_reason": freeze_reason,
        "context_summary_used": context_summary,
        "inference_backend": debug.get("inference_backend", "dual_model"),
        "normalized_text": debug.get("normalized_text", text),
        "normalization": debug.get("normalization", {}),
        "normalization_degraded": bool(debug.get("normalization_degraded", False)),
        "missing_fields": state.missing_fields,
        "assistant_message": question or completion_message(lang),
        "phase": phase,
        "phase_title": phase_title(phase, lang),
        "asked_fields": asked_fields,
        "asked_fields_friendly": asked_fields_friendly,
        "is_complete": complete,
        "delta_paths": delta_paths,
        "graph_candidates": debug.get("graph_candidates", []),
        "physics_recommendation": physics_recommendation,
        "display": _build_user_friendly(state.config),
        "config": state.config,
        "config_min": _export_min_config(state.config),
        "history": state.history[-10:],
    }


def legacy_solve(payload: Dict[str, Any]) -> Dict[str, Any]:
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

    frame, debug = _extract_semantic_frame_legacy(
        text,
        min_confidence=min_conf,
        device="auto",
        normalize_with_llm=normalize_input,
        normalize_config_path=get_ollama_config_path(),
        context_summary="",
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
            resp = chat(prompt, config_path=get_ollama_config_path(), temperature=0.2)
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




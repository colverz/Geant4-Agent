from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlu.llm_support.ollama_client import chat, extract_json
from ui.web.runtime_state import get_ollama_config_path


ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT.parent.parent / "knowledge" / "data"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_knowledge() -> Dict[str, List[str]]:
    materials = load_json(KNOWLEDGE_DIR / "materials_geant4_nist.json").get("materials", [])
    physics_lists = load_json(KNOWLEDGE_DIR / "physics_lists.json").get("items", [])
    particles = load_json(KNOWLEDGE_DIR / "particles.json").get("items", [])
    sources = load_json(KNOWLEDGE_DIR / "source_constraints.json").get("types", [])
    if not sources:
        sources = ["point", "beam", "plane", "isotropic"]
    output_formats = load_json(KNOWLEDGE_DIR / "output_formats.json").get("items", [])
    return {
        "materials": materials,
        "physics_lists": physics_lists,
        "particles": particles,
        "sources": sources,
        "output_formats": output_formats,
    }


def is_physics_recommend_request(text: str) -> bool:
    low = text.lower()
    physics_tokens = [
        "physics list",
        "physics_list",
        "物理列表",
        "物理过程",
        "预置物理",
        "预设物理",
    ]
    decision_tokens = [
        "choose",
        "select",
        "recommend",
        "best",
        "most suitable",
        "pick",
        "选择",
        "推荐",
        "最合适",
        "给出名称",
        "备选",
    ]
    has_physics = any(token in low or token in text for token in physics_tokens)
    has_decision = any(token in low or token in text for token in decision_tokens)
    return has_physics and has_decision


def pick_known_physics(name: str, choices: List[str]) -> Optional[str]:
    val = str(name or "").strip()
    if not val:
        return None
    for item in choices:
        if item.lower() == val.lower():
            return item
    return None


def recommend_physics_with_llm(
    text: str,
    context_summary: str,
    lang: str,
    *,
    choices: List[str],
) -> Dict[str, Any]:
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
        main = pick_known_physics(parsed.get("physics_list", ""), choices)
        if not main:
            return {}
        backup = pick_known_physics(parsed.get("backup_physics_list", ""), choices) or None
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

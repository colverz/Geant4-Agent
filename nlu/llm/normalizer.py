from __future__ import annotations

import re
from typing import Any

from core.orchestrator.types import Intent
from nlu.bert_lab.llm_bridge import build_normalization_prompt
from nlu.bert_lab.ollama_client import chat, extract_json


_TARGET_HINTS = {
    "geometry.params.module_x": ["module_x", "x=", "x:", "width", "\u5bbd", "\u5bbd\u5ea6", "x\u65b9\u5411"],
    "geometry.params.module_y": ["module_y", "y=", "y:", "height", "\u9ad8", "\u9ad8\u5ea6", "y\u65b9\u5411"],
    "geometry.params.module_z": ["module_z", "z=", "z:", "thickness", "\u539a", "\u539a\u5ea6", "z\u65b9\u5411"],
    "geometry.structure": ["structure", "geometry", "box", "cube", "sphere", "cylinder", "cylindrical", "ring", "grid"],
    "materials.selected_materials": ["material", "\u94dc", "copper", "g4_"],
    "source.type": ["source type", "point", "beam", "isotropic", "\u70b9\u6e90", "\u675f\u6d41"],
    "source.particle": ["gamma", "electron", "proton", "particle", "\u7c92\u5b50"],
    "source.energy": ["mev", "gev", "kev", "\u80fd\u91cf"],
    "source.position": ["position", "origin", "center", "\u4f4d\u7f6e", "\u539f\u70b9", "\u4e2d\u5fc3"],
    "source.direction": ["direction", "+z", "-z", "+x", "-x", "+y", "-y", "\u65b9\u5411"],
    "physics.physics_list": ["physics list", "\u7269\u7406\u5217\u8868", "ftfp", "qgsp", "qbbc", "shielding"],
    "output.format": ["output format", "root", "json", "csv", "\u8f93\u51fa\u683c\u5f0f"],
    "output.path": ["output path", "output/result", "\u8f93\u51fa\u8def\u5f84", "\u4fdd\u5b58\u5230", "\u5199\u5230", "file", "filename"],
}

_CONFIRM_PATTERNS = [
    r"^\s*(?:yes[, ]*)?(?:confirm|apply it|go ahead|approved?)\s*[.!?]?\s*$",
    r"^\s*(?:\u662f\u7684[,\uFF0C ]*)?(?:\u786e\u8ba4(?:\u4fee\u6539|\u8986\u76d6)?|\u597d\u7684[,\uFF0C ]*\u786e\u8ba4)\s*[\u3002\uFF01\uFF1F.!?]?\s*$",
]
_MODIFY_PATTERNS = [
    r"\b(?:change|switch|update|replace)\b",
    r"\bset\b.+\bto\b",
    r"\buse\b.+\binstead\b",
    r"\u6539\u6210",
    r"\u6539\u4e3a",
    r"\u628a.+?\u6539\u6210",
    r"\u628a.+?\u6539\u4e3a",
    r"\u66ff\u6362\u4e3a",
    r"\u6362\u6210",
]
_REMOVE_PATTERNS = [
    r"\b(?:remove|clear|delete|unset)\b",
    r"\u5220\u9664",
    r"\u79fb\u9664",
    r"\u6e05\u7a7a",
]
_QUESTION_PATTERNS = [
    r"[?\uFF1F]",
    r"\b(?:why|reason|how|what|which|can you|could you|please explain)\b",
    r"\u4e3a\u4ec0\u4e48",
    r"\u7406\u7531",
    r"\u539f\u56e0",
    r"\u600e\u4e48",
    r"\u5982\u4f55",
    r"\u662f\u5426",
    r"\u80fd\u5426",
    r"\u53ef\u4ee5\u5417",
]
_VECTOR_LITERAL = r"\(\s*[-+0-9.]+\s*,\s*[-+0-9.]+\s*,\s*[-+0-9.]+\s*\)"


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _infer_intent(text: str) -> Intent:
    compact = _compact(text)
    if _matches_any(compact, _CONFIRM_PATTERNS):
        return Intent.CONFIRM
    if _matches_any(compact, _MODIFY_PATTERNS):
        return Intent.MODIFY
    if _matches_any(compact, _REMOVE_PATTERNS):
        return Intent.REMOVE
    if _matches_any(compact, _QUESTION_PATTERNS):
        return Intent.QUESTION
    return Intent.SET


def _collect_target_paths(payload: str) -> list[str]:
    low = _compact(payload).lower()
    if not low:
        return []
    out: list[str] = []
    for path, hints in _TARGET_HINTS.items():
        if any(h.lower() in low for h in hints):
            out.append(path)
    if "pointing" in low or re.search(r"\balong\s*[+-]?[xyz]\b", low):
        out.append("source.direction")
    if (
        re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m)\s*(?:x|by)\s*\d+(?:\.\d+)?\s*(?:mm|cm|m)\s*(?:x|by)\s*\d+(?:\.\d+)?\s*(?:mm|cm|m)\b", low)
        or "\u89c1\u65b9" in low
        or "\u8fb9\u957f" in low
        or "side length" in low
    ):
        out.extend(
            [
                "geometry.structure",
                "geometry.params.module_x",
                "geometry.params.module_y",
                "geometry.params.module_z",
            ]
        )
    if any(keyword in low for keyword in ["radius", "diameter", "\u534a\u5f84"]):
        out.append("geometry.structure")
        out.append("geometry.params.child_rmax")
    if any(
        keyword in low
        for keyword in ["half-length", "half length", "half length", "\u534a\u957f", "height", "length", "\u9ad8\u5ea6"]
    ):
        out.append("geometry.structure")
        out.append("geometry.params.child_hz")
    if re.search(_VECTOR_LITERAL, low):
        if any(keyword in low for keyword in ["position", "origin", "center", "\u4f4d\u7f6e", "\u539f\u70b9", "\u4e2d\u5fc3"]):
            out.append("source.position")
        if re.search(rf"(?:located|source\s+at|beam\s+at)\s*{_VECTOR_LITERAL}", low):
            out.append("source.position")
        if re.search(rf"\bat\s*{_VECTOR_LITERAL}", low) and any(keyword in low for keyword in ["source", "beam", "point source", "particle source"]):
            out.append("source.position")
        if any(keyword in low for keyword in ["direction", "pointing", "\u65b9\u5411"]):
            out.append("source.direction")
    return sorted(set(out))


def _infer_target_paths(text: str, normalized_text: str = "") -> list[str]:
    out = _collect_target_paths(text)
    if out:
        return out
    if normalized_text:
        return _collect_target_paths(normalized_text)
    return []


def infer_user_turn_controls(user_text: str) -> dict[str, Any]:
    intent = _infer_intent(user_text)
    if intent == Intent.CONFIRM:
        target_paths: list[str] = []
    else:
        target_paths = _infer_target_paths(user_text)
    return {
        "intent": intent,
        "target_paths": target_paths,
    }


def normalize_user_turn(
    user_text: str,
    context_summary: str,
    config_path: str,
) -> dict[str, Any]:
    controls = infer_user_turn_controls(user_text)
    prompt = build_normalization_prompt(user_text, context_summary=context_summary)
    normalized_text = ""
    structure_hint = ""
    confidence = 0.6
    try:
        resp = chat(prompt, config_path=config_path, temperature=0.0)
        parsed = extract_json(resp.get("response", ""))
        if isinstance(parsed, dict):
            normalized_text = str(parsed.get("normalized_text", "")).strip()
            structure_hint = str(parsed.get("structure_hint", "")).strip()
            confidence = 0.8 if normalized_text else 0.6
    except Exception:
        normalized_text = ""

    if not normalized_text:
        normalized_text = user_text
        confidence = 0.4

    intent = controls["intent"]
    target_paths = controls["target_paths"] or _infer_target_paths(user_text, normalized_text)
    if re.search(r"\b(为什么|why|reason|理由)\b", user_text.lower()):
        intent = Intent.QUESTION
    return {
        "intent": intent,
        "target_paths": target_paths,
        "normalized_text": normalized_text,
        "structure_hint": structure_hint,
        "confidence": confidence,
    }

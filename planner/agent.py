from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from core.config.field_registry import clarification_items, friendly_labels
from core.config.prompt_registry import clarification_fallback, clarification_prompt
from nlu.llm_support.ollama_client import chat


logger = logging.getLogger(__name__)


def _clean_chat_text(raw: Any) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        str(raw),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()


_INTERNAL_FIELD_PATTERN = re.compile(r"\b[a-z]+(?:\.[a-z_]+)+\b")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_INTERNAL_TERM_PATTERN = re.compile(
    r"\b(single_[a-z_]+|module_[xyz]|child_[a-z0-9_]+|pitch_[xy]|tilt_[xy]|geometry parameter)\b",
    flags=re.IGNORECASE,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _contains_internal_field(text: str) -> bool:
    return bool(_INTERNAL_FIELD_PATTERN.search(text))


def _contains_internal_terms(text: str) -> bool:
    return bool(_INTERNAL_TERM_PATTERN.search(text))


def _looks_language_mismatched(text: str, lang: str) -> bool:
    compact = _normalize_whitespace(text)
    if not compact:
        return False
    has_cjk = bool(_CJK_PATTERN.search(compact))
    has_ascii_words = bool(re.search(r"[A-Za-z]{3,}", compact))
    if lang == "zh":
        return has_ascii_words and not has_cjk
    if lang == "en":
        return has_cjk and not has_ascii_words
    return False


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _fallback_question_for_paths(paths: List[str], friendly: List[str], lang: str) -> str:
    ordered_paths = [str(path) for path in paths if str(path)]
    path_set = set(ordered_paths)
    if lang == "zh":
        if {"source.position", "source.direction"}.issubset(path_set):
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u7684\u4f4d\u7f6e\u548c\u65b9\u5411\u3002"
        if "source.type" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u7c7b\u578b\uff0c\u4f8b\u5982\u70b9\u6e90\u3001\u675f\u6d41\u6216\u5404\u5411\u540c\u6027\u6e90\u3002"
        if "source.particle" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u7c92\u5b50\u7c7b\u578b\uff0c\u4f8b\u5982 gamma\u3001\u7535\u5b50\u6216\u8d28\u5b50\u3002"
        if "source.energy" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u80fd\u91cf\uff0c\u6309 MeV \u7ed9\u51fa\u5373\u53ef\u3002"
        if "source.position" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u4f4d\u7f6e\uff0c\u8bf7\u7ed9\u51fa x\u3001y\u3001z\u3002"
        if "source.direction" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u65b9\u5411\uff0c\u8bf7\u7ed9\u51fa\u65b9\u5411\u5411\u91cf\u3002"
        if "materials.volume_material_map" in path_set or "materials.selected_materials" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u6750\u6599\u5206\u914d\u5173\u7cfb\uff0c\u4e5f\u5c31\u662f\u54ea\u4e2a\u4f53\u79ef\u4f7f\u7528\u54ea\u79cd\u6750\u6599\u3002"
        if "physics.physics_list" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u7269\u7406\u5217\u8868\u540d\u79f0\uff0c\u4f8b\u5982 FTFP_BERT \u6216 QBBC\u3002"
        if {"output.format", "output.path"}.issubset(path_set):
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u8f93\u51fa\u683c\u5f0f\u4ee5\u53ca\u8f93\u51fa\u8def\u5f84\u3002"
        if "output.format" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u8f93\u51fa\u683c\u5f0f\uff0c\u4f8b\u5982 root\u3001json \u6216 csv\u3002"
        if "output.path" in path_set:
            return "\u8fd8\u9700\u8981\u786e\u8ba4\u8f93\u51fa\u8def\u5f84\u3002"
        if "geometry.ask.ring.radius" in path_set:
            return "\u4e3a\u4e86\u628a\u73af\u5f62\u7ed3\u6784\u8865\u5b8c\u6574\uff0c\u6211\u8fd8\u9700\u8981\u786e\u8ba4\u4e00\u4e0b\u73af\u534a\u5f84\u3002"
        if "geometry.ask.ring.module_size" in path_set:
            return "\u4e3a\u4e86\u628a\u73af\u5f62\u7ed3\u6784\u8865\u5b8c\u6574\uff0c\u6211\u8fd8\u9700\u8981\u6a21\u5757\u7684 x\u3001y\u3001z \u5c3a\u5bf8\u3002"
        if "geometry.ask.stack.thicknesses" in path_set:
            return "\u4e3a\u4e86\u628a\u5806\u53e0\u7ed3\u6784\u8865\u5b8c\u6574\uff0c\u6211\u8fd8\u9700\u8981\u5404\u5c42\u7684\u539a\u5ea6\u3002"
        if "geometry.ask.boolean.solid_a_size" in path_set:
            return "\u4e3a\u4e86\u628a\u5e03\u5c14\u51e0\u4f55\u8865\u5b8c\u6574\uff0c\u6211\u8fd8\u9700\u8981\u7b2c\u4e00\u4e2a solid \u7684 x\u3001y\u3001z \u5c3a\u5bf8\u3002"
    else:
        if {"source.position", "source.direction"}.issubset(path_set):
            return "I still need the source position and direction."
        if "source.type" in path_set:
            return "I still need the source type, for example point, beam, or isotropic."
        if "source.particle" in path_set:
            return "I still need the particle type, for example gamma, electron, or proton."
        if "source.energy" in path_set:
            return "I still need the source energy in MeV."
        if "source.position" in path_set:
            return "I still need the source position as x, y, and z."
        if "source.direction" in path_set:
            return "I still need the source direction vector."
        if "materials.volume_material_map" in path_set or "materials.selected_materials" in path_set:
            return "I still need the material assignment, meaning which volume uses which material."
        if "physics.physics_list" in path_set:
            return "I still need the physics list name, for example FTFP_BERT or QBBC."
        if {"output.format", "output.path"}.issubset(path_set):
            return "I still need the output format and output path."
        if "output.format" in path_set:
            return "I still need the output format, for example root, json, or csv."
        if "output.path" in path_set:
            return "I still need the output path."
        if "geometry.ask.ring.radius" in path_set:
            return "To complete the ring geometry, I still need the ring radius."
        if "geometry.ask.ring.module_size" in path_set:
            return "To complete the ring geometry, I still need the module size in x, y, and z."
        if "geometry.ask.stack.thicknesses" in path_set:
            return "To complete the stack geometry, I still need the layer thicknesses."
        if "geometry.ask.boolean.solid_a_size" in path_set:
            return "To complete the boolean geometry, I still need the size of the first solid in x, y, and z."
    return clarification_fallback(friendly, lang)


def ask_missing(
    missing: List[str],
    lang: str,
    ollama_config: str = "nlu/llm_support/configs/ollama_config.json",
    temperature: float = 1.0,
    recent_user_text: str = "",
    confirmed_items: List[str] | None = None,
) -> str:
    if not missing:
        return ""
    friendly = clarification_items(missing, lang)
    prompt = clarification_prompt(
        friendly,
        lang,
        recent_user_text=recent_user_text,
        confirmed_items=confirmed_items or [],
    )
    try:
        resp = chat(prompt, config_path=ollama_config, temperature=temperature)
        text = _clean_chat_text(resp.get("response", ""))
        if (
            text
            and not _contains_internal_field(text)
            and not _contains_internal_terms(text)
            and not _looks_language_mismatched(text, lang)
        ):
            return text
    except Exception:
        logger.warning("LLM clarification generation failed; using deterministic fallback.", exc_info=True)
    return _fallback_question_for_paths(missing, friendly, lang)


def _action_specific_rules(action: str, lang: str) -> str:
    if lang == "zh":
        rules = {
            "confirm_overwrite": "\u5fc5\u987b\u660e\u786e\u4fdd\u7559\u4e24\u4e2a\u7528\u6237\u9009\u9879\uff1a\u786e\u8ba4\u5e94\u7528\u4fee\u6539\uff0c\u6216\u4fdd\u6301\u539f\u503c\u3002",
            "reject_overwrite": "\u5fc5\u987b\u660e\u786e\u8bf4\u51fa\u8fd9\u6b21\u8986\u76d6\u6ca1\u6709\u88ab\u5e94\u7528\uff0c\u5f53\u524d\u503c\u5df2\u4fdd\u7559\u3002",
            "confirm_update": "\u7528\u7b80\u77ed\u8bed\u53e5\u786e\u8ba4\u5df2\u540c\u6b65\u7684\u5185\u5bb9\u3002",
            "summarize_progress": "\u6982\u62ec\u8fd9\u4e00\u8f6e\u5df2\u5b8c\u6210\u7684\u5185\u5bb9\uff0c\u4ee5\u53ca\u63a5\u4e0b\u6765\u4ecd\u9700\u8981\u7684\u4fe1\u606f\u3002",
            "explain_choice": "\u89e3\u91ca\u9009\u62e9\u7406\u7531\uff0c\u4f46\u4e0d\u8981\u865a\u6784\u65b0\u4e8b\u5b9e\u3002",
            "answer_status": "\u7528\u4e00\u53e5\u7b80\u77ed\u7684\u7528\u6237\u8bed\u8a00\u8bf4\u660e\u5f53\u524d\u72b6\u6001\u3002",
        }
    else:
        rules = {
            "confirm_overwrite": "You must preserve both user choices explicitly: confirm the change or keep the current value.",
            "reject_overwrite": "You must clearly state that the overwrite was not applied and the current value was kept.",
            "confirm_update": "Acknowledge the applied update in one concise sentence.",
            "summarize_progress": "Summarize what was completed this turn and what is still needed next.",
            "explain_choice": "Explain the selection reason briefly without inventing new facts.",
            "answer_status": "State the current status in one concise user-facing sentence.",
        }
    return rules.get(action, "")


def _is_invalid_naturalization(text: str, *, action: str, lang: str) -> bool:
    compact = _normalize_whitespace(text)
    if not compact:
        return True
    if _contains_internal_field(compact):
        return True
    if _contains_internal_terms(compact):
        return True
    if _looks_language_mismatched(compact, lang):
        return True

    lowered = compact.lower()
    if action == "confirm_overwrite":
        if lang == "zh":
            has_confirm = _contains_any(compact, ["\u786e\u8ba4", "\u5e94\u7528"])
            has_keep = _contains_any(compact, ["\u4fdd\u6301\u539f\u503c", "\u4fdd\u7559\u539f\u503c", "\u4fdd\u6301\u5f53\u524d\u503c", "\u4fdd\u7559\u5f53\u524d\u503c"])
            return not (has_confirm and has_keep)
        has_confirm = _contains_any(lowered, ["confirm", "apply"])
        has_keep = _contains_any(lowered, ["keep existing", "keep current", "keep the current", "keep the current value"])
        return not (has_confirm and has_keep)

    if action == "reject_overwrite":
        if lang == "zh":
            if _contains_any(compact, ["\u5df2\u66f4\u65b0", "\u5df2\u5e94\u7528"]) and not _contains_any(compact, ["\u4e0d\u5e94\u7528", "\u4fdd\u7559", "\u4fdd\u6301"]):
                return True
        else:
            if _contains_any(lowered, ["updated ", "applied", "overwrite confirmed"]) and not _contains_any(
                lowered,
                ["did not apply", "kept the current", "kept current", "kept existing"],
            ):
                return True
    return False


def naturalize_response(
    base_message: str,
    *,
    lang: str,
    action: str,
    updated_paths: list[str],
    missing_fields: list[str],
    asked_fields: list[str],
    overwrite_preview: list[dict] | None,
    dialogue_summary: dict | None,
    raw_dialogue: list[dict] | None,
    ollama_config: str = "nlu/llm_support/configs/ollama_config.json",
    temperature: float = 1.0,
) -> str:
    if not base_message:
        return base_message

    action_rules = _action_specific_rules(action, lang)
    context_payload = {
        "lang": lang,
        "action": action,
        "action_rules": action_rules,
        "updated_fields": friendly_labels(updated_paths[:5], lang),
        "missing_fields": friendly_labels(missing_fields[:5], lang),
        "asked_fields": friendly_labels(asked_fields[:5], lang),
        "overwrite_preview": list(overwrite_preview or [])[:2],
        "summary": dict(dialogue_summary or {}),
        "recent_dialogue": list(raw_dialogue or [])[-6:],
        "base_message": base_message,
    }

    if lang == "zh":
        system_rules = (
            "\u4f60\u662f Geant4 \u914d\u7f6e\u52a9\u624b\u7684\u7528\u6237\u5c42\u6539\u5199\u5668\u3002"
            "\u4efb\u52a1\uff1a\u628a\u7ed9\u5b9a\u57fa\u7840\u56de\u590d\u6539\u5199\u6210\u81ea\u7136\u3001\u7b80\u6d01\u3001\u53cb\u597d\u7684\u4e2d\u6587\u3002"
            "\u786c\u7ea6\u675f\uff1a"
            "1) \u4e0d\u5f97\u65b0\u589e\u4e8b\u5b9e\u3001\u53c2\u6570\u3001\u5b57\u6bb5\u6216\u7ed3\u8bba\uff1b"
            "2) \u4e0d\u5f97\u5220\u9664\u57fa\u7840\u56de\u590d\u4e2d\u7684\u5173\u952e\u7ea6\u675f\uff08\u5c24\u5176\u662f\u8986\u76d6\u786e\u8ba4\u63d0\u793a\uff09\uff1b"
            "3) \u4e0d\u8f93\u51fa\u63a8\u7406\u8fc7\u7a0b\uff1b"
            "4) \u4ec5\u8f93\u51fa\u6700\u7ec8\u7ed9\u7528\u6237\u7684\u4e00\u6bb5\u6587\u672c\u3002"
        )
    else:
        system_rules = (
            "You are the user-facing rewrite layer for a Geant4 configuration assistant. "
            "Rewrite the base reply into natural, concise English. "
            "Constraints: "
            "1) Do not add new facts/fields/requirements; "
            "2) Do not remove critical constraints (especially overwrite confirmation warnings); "
            "3) Do not output reasoning; "
            "4) Return only the final user-facing message."
        )

    prompt = (
        f"{system_rules}\n\n"
        f"Context JSON:\n{json.dumps(context_payload, ensure_ascii=False)}\n\n"
        "Rewrite now."
    )

    try:
        resp = chat(prompt, config_path=ollama_config, temperature=temperature)
        text = _clean_chat_text(resp.get("response", ""))
        if text and not _is_invalid_naturalization(text, action=action, lang=lang):
            return text
    except Exception:
        logger.warning("LLM response naturalization failed; using grounded base message.", exc_info=True)
    return base_message


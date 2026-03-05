from __future__ import annotations

import json
import re
from typing import Any, List

from core.config.field_registry import clarification_items, friendly_labels
from core.config.prompt_registry import clarification_fallback, clarification_prompt
from nlu.bert_lab.ollama_client import chat


def _clean_chat_text(raw: Any) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        str(raw),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()


_INTERNAL_FIELD_PATTERN = re.compile(r"\b[a-z]+(?:\.[a-z_]+)+\b")


def _contains_internal_field(text: str) -> bool:
    return bool(_INTERNAL_FIELD_PATTERN.search(text))


def ask_missing(
    missing: List[str],
    lang: str,
    ollama_config: str = "nlu/bert_lab/configs/ollama_config.json",
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
        if text and not _contains_internal_field(text):
            return text
    except Exception:
        pass
    return clarification_fallback(friendly, lang)


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
    ollama_config: str = "nlu/bert_lab/configs/ollama_config.json",
    temperature: float = 1.0,
) -> str:
    if not base_message:
        return base_message

    context_payload = {
        "lang": lang,
        "action": action,
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
        if text:
            return text
    except Exception:
        pass
    return base_message


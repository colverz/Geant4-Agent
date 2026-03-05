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
        if text:
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
            "你是Geant4配置助手的用户层回复改写器。"
            "任务：把给定基础回复改写成自然中文，口吻清晰友好。"
            "硬约束："
            "1) 不得新增事实、参数、字段或结论；"
            "2) 不得删除基础回复中的关键约束（尤其确认/覆盖警告）；"
            "3) 不输出推理过程；"
            "4) 仅输出最终给用户的一段文本。"
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

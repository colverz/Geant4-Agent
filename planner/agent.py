from __future__ import annotations

import re
from typing import List

from core.config.field_registry import clarification_items
from core.config.prompt_registry import clarification_fallback, clarification_prompt
from nlu.bert_lab.ollama_client import chat


def ask_missing(
    missing: List[str],
    lang: str,
    ollama_config: str = "nlu/bert_lab/configs/ollama_config.json",
    temperature: float = 1.0,
) -> str:
    if not missing:
        return ""
    friendly = clarification_items(missing, lang)
    prompt = clarification_prompt(friendly, lang)
    try:
        resp = chat(prompt, config_path=ollama_config, temperature=temperature)
        text = re.sub(
            r"<think>.*?</think>",
            "",
            str(resp.get("response", "")),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        if text:
            return text
    except Exception:
        pass
    return clarification_fallback(friendly, lang)

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.interpreter.parser import InterpreterParseResult, parse_interpreter_response
from core.interpreter.prompt import build_interpreter_prompt
from nlu.llm_support.ollama_client import chat


@dataclass
class InterpreterRunResult:
    ok: bool
    parsed: InterpreterParseResult
    llm_raw: str
    cleaned_text: str
    fallback_reason: str | None = None


def _clean_response(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    return text


def run_interpreter(
    user_text: str,
    context_summary: str,
    *,
    config_path: str = "nlu/llm_support/configs/ollama_config.json",
    **options: Any,
) -> InterpreterRunResult:
    prompt = build_interpreter_prompt(user_text, context_summary)
    resp = chat(prompt, config_path=config_path, **options)
    llm_raw = str(resp.get("response", "") or "")
    cleaned = _clean_response(llm_raw)
    parsed = parse_interpreter_response(cleaned)
    return InterpreterRunResult(
        ok=parsed.ok,
        parsed=parsed,
        llm_raw=llm_raw,
        cleaned_text=cleaned,
        fallback_reason=parsed.error,
    )

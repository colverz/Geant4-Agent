from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.config.prompt_profiles import PromptTask, PromptValidationResult, build_prompt, validate_prompt_output
from core.interpreter.parser import InterpreterParseResult, parse_interpreter_response
from core.interpreter.prompt import build_interpreter_prompt, detect_prompt_language
from nlu.llm_support.ollama_client import extract_json
from nlu.llm_support.ollama_client import chat


@dataclass
class InterpreterRunResult:
    ok: bool
    parsed: InterpreterParseResult
    llm_raw: str
    cleaned_text: str
    fallback_reason: str | None = None


@dataclass
class InterpreterV2RunResult:
    ok: bool
    payload: dict[str, Any]
    validation: PromptValidationResult
    llm_raw: str
    cleaned_text: str
    prompt_profile_id: str
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


def run_interpreter_v2(
    user_text: str,
    context_summary: str,
    *,
    config_path: str = "nlu/llm_support/configs/ollama_config.json",
    **options: Any,
) -> InterpreterV2RunResult:
    language = detect_prompt_language(user_text)
    profile_lang = "zh" if language in {"zh", "mixed"} else "en"
    source_context = {
        "user_text": user_text,
        "context_summary": context_summary,
        "stable_context_text": str(options.pop("stable_context_text", "") or ""),
    }
    built = build_prompt(PromptTask.INTERPRET_USER_TURN_V2, profile_lang, source_context)
    resp = chat(built.prompt, config_path=config_path, **options)
    llm_raw = str(resp.get("response", "") or "")
    cleaned = _clean_response(llm_raw)
    payload = extract_json(cleaned)
    if not isinstance(payload, dict):
        validation = PromptValidationResult(ok=False, validator_name=built.validator_name, errors=["not_json"])
        return InterpreterV2RunResult(
            ok=False,
            payload={},
            validation=validation,
            llm_raw=llm_raw,
            cleaned_text=cleaned,
            prompt_profile_id=built.profile_id,
            fallback_reason="json_parse_failed",
        )
    validation = validate_prompt_output(PromptTask.INTERPRET_USER_TURN_V2, profile_lang, payload, source_context)
    return InterpreterV2RunResult(
        ok=validation.ok,
        payload=payload,
        validation=validation,
        llm_raw=llm_raw,
        cleaned_text=cleaned,
        prompt_profile_id=built.profile_id,
        fallback_reason=None if validation.ok else "validation_failed",
    )

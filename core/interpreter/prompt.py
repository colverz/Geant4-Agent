from __future__ import annotations

from core.config.prompt_profiles import PromptTask, build_prompt


def detect_prompt_language(user_text: str) -> str:
    text = user_text or ""
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    has_ascii_alpha = any(("a" <= ch.lower() <= "z") for ch in text)
    if has_cjk and has_ascii_alpha:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def build_interpreter_prompt(user_text: str, context_summary: str) -> str:
    language = detect_prompt_language(user_text)
    profile_lang = "zh" if language in {"zh", "mixed"} else "en"
    return build_prompt(
        PromptTask.INTERPRET_USER_TURN,
        profile_lang,
        {
            "user_text": user_text,
            "context_summary": context_summary,
        },
    ).prompt


def build_interpreter_v2_prompt(user_text: str, context_summary: str) -> str:
    language = detect_prompt_language(user_text)
    profile_lang = "zh" if language in {"zh", "mixed"} else "en"
    return build_prompt(
        PromptTask.INTERPRET_USER_TURN_V2,
        profile_lang,
        {
            "user_text": user_text,
            "context_summary": context_summary,
        },
    ).prompt

from __future__ import annotations

from typing import Sequence


def _lang_key(lang: str) -> str:
    return "zh" if lang == "zh" else "en"


def completion_message(lang: str) -> str:
    return "\u914d\u7f6e\u5df2\u5b8c\u6210\u3002" if _lang_key(lang) == "zh" else "Configuration complete."


def single_field_request(path: str, lang: str) -> str:
    if _lang_key(lang) == "zh":
        return f"\u8bf7\u8865\u5145\u5b57\u6bb5\uff1a{path}"
    return f"Please provide: {path}"


def clarification_fallback(items: Sequence[str], lang: str) -> str:
    joined = ", ".join(items)
    if _lang_key(lang) == "zh":
        return f"\u8bf7\u8865\u5145\uff1a{joined}"
    return f"Please provide: {joined}"


def clarification_prompt(items: Sequence[str], lang: str) -> str:
    joined = ", ".join(items)
    if _lang_key(lang) == "zh":
        return (
            "\u4f60\u662f\u79d1\u7814\u8f6f\u4ef6\u52a9\u624b\u3002"
            "\u8bf7\u7528\u81ea\u7136\u3001\u7b80\u6d01\u3001\u53cb\u597d\u7684\u4e2d\u6587\uff0c"
            "\u5408\u5e76\u6210\u4e00\u53e5\u8ffd\u95ee\uff0c\u5411\u7528\u6237\u8865\u9f50\u7f3a\u5931\u4fe1\u606f\u3002"
            "\u4e0d\u8981\u66b4\u9732\u5185\u90e8\u5b57\u6bb5\u540d\u3002\n"
            f"\u7f3a\u5931\u5185\u5bb9\uff1a{joined}\n"
            "\u95ee\u9898\uff1a"
        )
    return (
        "You are a research assistant. Ask one concise and friendly clarification question in English. "
        "Do not expose internal field names.\n"
        f"Missing items: {joined}\n"
        "Question:"
    )

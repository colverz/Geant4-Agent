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
    key = _lang_key(lang)
    if not items:
        return "\u8bf7\u8865\u5145\u5173\u952e\u53c2\u6570\u3002" if key == "zh" else "Please provide the key missing parameters."
    if len(items) == 1:
        if key == "zh":
            return f"\u4e3a\u4e86\u7ee7\u7eed\u914d\u7f6e\uff0c\u8bf7\u518d\u786e\u8ba4\u4e00\u4e0b{items[0]}\u3002"
        return f"To continue, could you confirm {items[0]}?"
    head = ", ".join(items[:2])
    if key == "zh":
        return f"\u4e3a\u4e86\u7ee7\u7eed\u914d\u7f6e\uff0c\u8fd8\u9700\u8981{head}\u3002"
    return f"To continue, I still need {head}."


def clarification_prompt(
    items: Sequence[str],
    lang: str,
    *,
    recent_user_text: str = "",
    confirmed_items: Sequence[str] | None = None,
) -> str:
    joined = ", ".join(items)
    confirmed = ", ".join(confirmed_items or [])
    recent_user_text = (recent_user_text or "").strip()
    if _lang_key(lang) == "zh":
        return (
            "\u4f60\u662f Geant4-Agent \u7684\u5bf9\u8bdd\u52a9\u624b\u3002"
            "\u76ee\u6807\uff1a\u57fa\u4e8e\u5f53\u524d\u4e0a\u4e0b\u6587\uff0c\u7528\u81ea\u7136\u3001\u4e0d\u673a\u68b0\u7684\u8bed\u6c14\u53d1\u8d77\u8ffd\u95ee\u3002"
            "\u786c\u7ea6\u675f\uff1a"
            "1) \u4e0d\u8981\u5217\u51fa\u5185\u90e8\u5b57\u6bb5\u540d\uff1b"
            "2) \u4e0d\u8981\u65b0\u589e\u9700\u6c42\uff1b"
            "3) \u4e00\u8f6e\u6700\u591a\u95ee 1~2 \u4e2a\u5173\u952e\u7f3a\u5931\u70b9\uff1b"
            "4) \u8f93\u51fa\u4e00\u6bb5\u6700\u7ec8\u95ee\u53e5\uff0c\u4e0d\u8981\u89e3\u91ca\u3002\n"
            f"\u7528\u6237\u6700\u8fd1\u8f93\u5165\uff1a{recent_user_text or '\u65e0'}\n"
            f"\u5df2\u786e\u8ba4\u4fe1\u606f\uff1a{confirmed or '\u65e0'}\n"
            f"\u672c\u8f6e\u5f85\u8865\u5145\uff1a{joined}\n"
            "\u8ffd\u95ee\uff1a"
        )
    return (
        "You are the Geant4-Agent dialogue assistant. "
        "Write a natural, human clarification question using current context. "
        "Constraints: "
        "1) do not expose internal field names; "
        "2) do not introduce new requirements; "
        "3) ask at most 1-2 missing items in this turn; "
        "4) return one final question only.\n"
        f"Latest user input: {recent_user_text or 'N/A'}\n"
        f"Confirmed context: {confirmed or 'N/A'}\n"
        f"Missing items this turn: {joined}\n"
        "Question:"
    )

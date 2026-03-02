from __future__ import annotations

from typing import Any

from core.config.field_registry import friendly_labels
from core.dialogue.types import DialogueDecision


def build_raw_dialogue(history: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, str]]:
    trimmed = history[-limit:]
    output: list[dict[str, str]] = []
    for item in trimmed:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        output.append({"role": role, "content": content})
    return output


def build_dialogue_summary(
    decision: DialogueDecision,
    *,
    lang: str,
    is_complete: bool,
) -> dict[str, Any]:
    return {
        "status": "complete" if is_complete else "pending",
        "last_action": decision.action.value,
        "user_intent": decision.user_intent,
        "updated_fields": friendly_labels(decision.updated_paths[:5], lang),
        "answered_fields": friendly_labels(decision.answered_this_turn[:5], lang),
        "pending_fields": friendly_labels(decision.missing_fields[:5], lang),
        "next_questions": friendly_labels(decision.asked_fields[:3], lang),
    }


def sync_dialogue_state(
    state: Any,
    *,
    decision: DialogueDecision,
    lang: str,
    is_complete: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary = build_dialogue_summary(decision, lang=lang, is_complete=is_complete)
    raw_dialogue = build_raw_dialogue(getattr(state, "history", []))
    state.dialogue_summary = summary
    return summary, raw_dialogue

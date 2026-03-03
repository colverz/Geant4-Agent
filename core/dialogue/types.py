from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DialogueAction(str, Enum):
    ASK_CLARIFICATION = "ask_clarification"
    CONFIRM_OVERWRITE = "confirm_overwrite"
    EXPLAIN_CHOICE = "explain_choice"
    SUMMARIZE_PROGRESS = "summarize_progress"
    CONFIRM_UPDATE = "confirm_update"
    ANSWER_STATUS = "answer_status"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class DialogueDecision:
    action: DialogueAction
    asked_fields: list[str] = field(default_factory=list)
    updated_paths: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    answered_this_turn: list[str] = field(default_factory=list)
    overwrite_preview: list[dict] = field(default_factory=list)
    explanation: dict = field(default_factory=dict)
    user_intent: str = "OTHER"


def build_dialogue_trace(decision: DialogueDecision) -> dict:
    return {
        "action": decision.action.value,
        "user_intent": decision.user_intent,
        "asked_fields": list(decision.asked_fields),
        "updated_paths": list(decision.updated_paths),
        "missing_fields": list(decision.missing_fields),
        "answered_this_turn": list(decision.answered_this_turn),
        "overwrite_preview": list(decision.overwrite_preview),
        "explanation": dict(decision.explanation),
    }

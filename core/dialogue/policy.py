from __future__ import annotations

from core.dialogue.types import DialogueAction, DialogueDecision


def decide_dialogue_action(
    *,
    user_intent: str,
    is_complete: bool,
    asked_fields: list[str],
    missing_fields: list[str],
    updated_paths: list[str],
    answered_this_turn: list[str],
) -> DialogueDecision:
    if is_complete:
        return DialogueDecision(
            action=DialogueAction.FINALIZE,
            updated_paths=list(updated_paths),
            missing_fields=[],
            answered_this_turn=list(answered_this_turn),
            user_intent=user_intent,
        )
    if asked_fields:
        return DialogueDecision(
            action=DialogueAction.ASK_CLARIFICATION,
            asked_fields=list(asked_fields),
            updated_paths=list(updated_paths),
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            user_intent=user_intent,
        )
    if updated_paths:
        return DialogueDecision(
            action=DialogueAction.CONFIRM_UPDATE,
            updated_paths=list(updated_paths),
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            user_intent=user_intent,
        )
    return DialogueDecision(
        action=DialogueAction.ANSWER_STATUS,
        missing_fields=list(missing_fields),
        answered_this_turn=list(answered_this_turn),
        user_intent=user_intent,
    )

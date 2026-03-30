from __future__ import annotations

from core.dialogue.types import DialogueAction, DialogueDecision


_EXPLANATION_SAFE_METADATA_PATHS = {
    "physics.backup_physics_list",
    "physics.selection_source",
    "physics.selection_reasons",
    "materials.selection_source",
    "materials.selection_reasons",
    "source.selection_source",
    "source.selection_reasons",
}


def _updated_paths_block_explanation(updated_paths: list[str]) -> bool:
    if not updated_paths:
        return False
    return any(path not in _EXPLANATION_SAFE_METADATA_PATHS for path in updated_paths)


def decide_dialogue_action(
    *,
    user_intent: str,
    is_complete: bool,
    asked_fields: list[str],
    missing_fields: list[str],
    updated_paths: list[str],
    answered_this_turn: list[str],
    pending_overwrite_preview: list[dict] | None = None,
    rejected_overwrite_preview: list[dict] | None = None,
    available_explanations: dict | None = None,
    last_dialogue_action: str = "",
) -> DialogueDecision:
    pending_overwrite_preview = list(pending_overwrite_preview or [])
    rejected_overwrite_preview = list(rejected_overwrite_preview or [])
    available_explanations = dict(available_explanations or {})
    if pending_overwrite_preview:
        return DialogueDecision(
            action=DialogueAction.CONFIRM_OVERWRITE,
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            overwrite_preview=pending_overwrite_preview,
            user_intent=user_intent,
        )
    if rejected_overwrite_preview:
        return DialogueDecision(
            action=DialogueAction.REJECT_OVERWRITE,
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            overwrite_preview=rejected_overwrite_preview,
            user_intent=user_intent,
        )
    explainable_sources = {
        key: value
        for key, value in available_explanations.items()
        if isinstance(value, dict) and (value.get("source") or value.get("reasons"))
    }
    should_explain = (
        bool(explainable_sources)
        and user_intent == "QUESTION"
        and not asked_fields
        and not _updated_paths_block_explanation(updated_paths)
    )
    if should_explain:
        return DialogueDecision(
            action=DialogueAction.EXPLAIN_CHOICE,
            updated_paths=list(updated_paths),
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            explanation=explainable_sources,
            user_intent=user_intent,
        )
    if user_intent == "QUESTION" and not asked_fields and not _updated_paths_block_explanation(updated_paths):
        return DialogueDecision(
            action=DialogueAction.ANSWER_STATUS,
            updated_paths=list(updated_paths),
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            user_intent=user_intent,
        )
    if updated_paths and missing_fields and (
        answered_this_turn
        or last_dialogue_action == DialogueAction.ASK_CLARIFICATION.value
        or len(updated_paths) > 1
    ):
        return DialogueDecision(
            action=DialogueAction.SUMMARIZE_PROGRESS,
            updated_paths=list(updated_paths),
            missing_fields=list(missing_fields),
            answered_this_turn=list(answered_this_turn),
            user_intent=user_intent,
        )
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

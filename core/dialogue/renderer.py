from __future__ import annotations

from core.config.field_registry import friendly_labels
from core.config.prompt_registry import clarification_fallback, completion_message
from core.dialogue.types import DialogueAction, DialogueDecision
from planner.question_renderer import render_question


def _render_update_status(decision: DialogueDecision, *, lang: str) -> str:
    updated = friendly_labels(decision.updated_paths[:3], lang)
    remaining = friendly_labels(decision.missing_fields[:2], lang)
    if lang == "zh":
        if updated and remaining:
            return f"\u5df2\u66f4\u65b0\uff1a{', '.join(updated)}\u3002\u4ecd\u9700\u8865\u5145\uff1a{', '.join(remaining)}\u3002"
        if updated:
            return f"\u5df2\u66f4\u65b0\uff1a{', '.join(updated)}\u3002"
        if remaining:
            return f"\u5f53\u524d\u4ecd\u9700\u8865\u5145\uff1a{', '.join(remaining)}\u3002"
        return "\u5f53\u524d\u914d\u7f6e\u5df2\u540c\u6b65\u3002"
    if updated and remaining:
        return f"Updated: {', '.join(updated)}. Still needed: {', '.join(remaining)}."
    if updated:
        return f"Updated: {', '.join(updated)}."
    if remaining:
        return f"Still needed: {', '.join(remaining)}."
    return "Configuration state synchronized."


def render_dialogue_message(
    decision: DialogueDecision,
    *,
    lang: str,
    use_llm_question: bool,
    ollama_config: str,
    user_temperature: float,
) -> str:
    if decision.action == DialogueAction.FINALIZE:
        return completion_message(lang)
    if decision.action == DialogueAction.ASK_CLARIFICATION:
        if use_llm_question:
            return render_question(
                decision.asked_fields,
                lang=lang,
                ollama_config=ollama_config,
                temperature=user_temperature,
            )
        return clarification_fallback(friendly_labels(decision.asked_fields, lang), lang)
    if decision.action in {DialogueAction.CONFIRM_UPDATE, DialogueAction.ANSWER_STATUS}:
        return _render_update_status(decision, lang=lang)
    return completion_message(lang)

from __future__ import annotations

from planner.agent import ask_missing, naturalize_response


def render_question(
    planned_paths: list[str],
    *,
    lang: str,
    ollama_config: str,
    temperature: float,
    recent_user_text: str = "",
    confirmed_items: list[str] | None = None,
) -> str:
    return ask_missing(
        planned_paths,
        lang=lang,
        ollama_config=ollama_config,
        temperature=temperature,
        recent_user_text=recent_user_text,
        confirmed_items=confirmed_items or [],
    )


def render_naturalized_response(
    base_message: str,
    *,
    lang: str,
    action: str,
    updated_paths: list[str],
    missing_fields: list[str],
    asked_fields: list[str],
    overwrite_preview: list[dict] | None,
    dialogue_summary: dict | None,
    raw_dialogue: list[dict] | None,
    ollama_config: str,
    temperature: float,
) -> str:
    return naturalize_response(
        base_message,
        lang=lang,
        action=action,
        updated_paths=updated_paths,
        missing_fields=missing_fields,
        asked_fields=asked_fields,
        overwrite_preview=overwrite_preview,
        dialogue_summary=dialogue_summary,
        raw_dialogue=raw_dialogue,
        ollama_config=ollama_config,
        temperature=temperature,
    )

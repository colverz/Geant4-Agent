from __future__ import annotations

from typing import Any

from core.config.field_registry import friendly_labels, is_user_visible_summary_path
from core.orchestrator.path_ops import get_path
from core.dialogue.types import DialogueDecision

_DOMAIN_ORDER = ("geometry", "materials", "source", "physics", "output")
_DOMAIN_LABELS = {
    "en": {
        "geometry": "Geometry",
        "materials": "Materials",
        "source": "Source",
        "physics": "Physics",
        "output": "Output",
    },
    "zh": {
        "geometry": "\u51e0\u4f55",
        "materials": "\u6750\u6599",
        "source": "\u6e90",
        "physics": "\u7269\u7406",
        "output": "\u8f93\u51fa",
    },
}
_EXPLANATION_PRIMARY_PATHS = {
    "materials": "materials.selected_materials",
    "source": "source.type",
    "physics": "physics.physics_list",
}


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
    confirmed_fact_paths: list[str] | None = None,
    memory_depth: int = 0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confirmed_fact_paths = confirmed_fact_paths or []
    visible_updated = _filter_summary_paths(decision.updated_paths)
    visible_answered = _filter_summary_paths(decision.answered_this_turn)
    visible_pending = _filter_summary_paths(decision.missing_fields)
    visible_asked = _filter_summary_paths(decision.asked_fields)
    if decision.action.value == "finalize":
        finalize_confirmed = list(dict.fromkeys([*decision.updated_paths, *decision.answered_this_turn]))
        visible_confirmed = _filter_summary_paths(finalize_confirmed) or _filter_summary_paths(confirmed_fact_paths)
    else:
        visible_confirmed = _filter_summary_paths(confirmed_fact_paths)
    grouped_status = build_grouped_status(
        updated_paths=visible_updated,
        pending_paths=visible_pending,
        confirmed_paths=visible_confirmed,
        lang=lang,
    )
    available_explanations = collect_available_explanations(config or {}, lang=lang)
    return {
        "status": "complete" if is_complete else "pending",
        "last_action": decision.action.value,
        "user_intent": decision.user_intent,
        "updated_fields": friendly_labels(visible_updated[:5], lang),
        "answered_fields": friendly_labels(visible_answered[:5], lang),
        "pending_fields": friendly_labels(visible_pending[:5], lang),
        "next_questions": friendly_labels(visible_asked[:3], lang),
        "recent_confirmed": friendly_labels(visible_confirmed[:5], lang),
        "memory_depth": memory_depth,
        "grouped_status": grouped_status,
        "available_explanations": available_explanations,
    }


def _merge_recent_paths(previous: list[str], incoming: list[str], *, limit: int) -> list[str]:
    merged = list(previous)
    for path in incoming:
        if not path:
            continue
        if path in merged:
            merged.remove(path)
        merged.insert(0, path)
    return merged[:limit]


def _build_memory_entry(decision: DialogueDecision, *, is_complete: bool) -> dict[str, Any]:
    return {
        "action": decision.action.value,
        "user_intent": decision.user_intent,
        "updated_paths": list(decision.updated_paths),
        "answered_this_turn": list(decision.answered_this_turn),
        "missing_fields": [] if is_complete else list(decision.missing_fields),
        "asked_fields": list(decision.asked_fields),
        "explanation_domains": sorted(decision.explanation.keys()),
    }


def _domain_for_path(path: str) -> str:
    path = str(path or "")
    for domain in _DOMAIN_ORDER:
        if path == domain or path.startswith(domain + "."):
            return domain
    return "other"


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _filter_summary_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if is_user_visible_summary_path(path)]


def build_grouped_status(
    *,
    updated_paths: list[str],
    pending_paths: list[str],
    confirmed_paths: list[str],
    lang: str,
) -> dict[str, dict[str, Any]]:
    lang_key = "zh" if lang == "zh" else "en"
    grouped: dict[str, dict[str, Any]] = {}
    for domain in _DOMAIN_ORDER:
        domain_updated = _unique(
            [path for path in updated_paths if _domain_for_path(path) == domain and is_user_visible_summary_path(path)]
        )
        domain_pending = _unique(
            [path for path in pending_paths if _domain_for_path(path) == domain and is_user_visible_summary_path(path)]
        )
        domain_confirmed = _unique(
            [path for path in confirmed_paths if _domain_for_path(path) == domain and is_user_visible_summary_path(path)]
        )
        if not (domain_updated or domain_pending or domain_confirmed):
            continue
        grouped[domain] = {
            "label": _DOMAIN_LABELS[lang_key][domain],
            "updated_fields": friendly_labels(domain_updated[:3], lang),
            "pending_fields": friendly_labels(domain_pending[:3], lang),
            "confirmed_fields": friendly_labels(domain_confirmed[:3], lang),
        }
    return grouped


def collect_available_explanations(config: dict[str, Any], *, lang: str) -> dict[str, dict[str, Any]]:
    explanations: dict[str, dict[str, Any]] = {}
    for domain, primary_path in _EXPLANATION_PRIMARY_PATHS.items():
        source = get_path(config, f"{domain}.selection_source")
        reasons = get_path(config, f"{domain}.selection_reasons")
        if not source and not reasons:
            continue
        reasons_list = list(reasons) if isinstance(reasons, list) else ([str(reasons)] if reasons else [])
        explanations[domain] = {
            "label": _DOMAIN_LABELS["zh" if lang == "zh" else "en"][domain],
            "field": friendly_labels([primary_path], lang)[0],
            "source": str(source or ""),
            "reasons": reasons_list[:3],
        }
    return explanations


def sync_dialogue_state(
    state: Any,
    *,
    decision: DialogueDecision,
    lang: str,
    is_complete: bool,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]]]:
    confirmed_updates = list(dict.fromkeys([*decision.updated_paths, *decision.answered_this_turn]))
    state.confirmed_fact_paths = _merge_recent_paths(
        list(getattr(state, "confirmed_fact_paths", [])),
        confirmed_updates,
        limit=12,
    )
    memory = list(getattr(state, "dialogue_memory", []))
    memory.append(_build_memory_entry(decision, is_complete=is_complete))
    state.dialogue_memory = memory[-8:]
    summary = build_dialogue_summary(
        decision,
        lang=lang,
        is_complete=is_complete,
        confirmed_fact_paths=state.confirmed_fact_paths,
        memory_depth=len(state.dialogue_memory),
        config=getattr(state, "config", {}),
    )
    raw_dialogue = build_raw_dialogue(getattr(state, "history", []))
    state.dialogue_summary = summary
    return summary, raw_dialogue, list(state.dialogue_memory)

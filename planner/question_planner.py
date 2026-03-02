from __future__ import annotations

from core.config.field_registry import friendly_labels
from core.orchestrator.types import Phase
from core.validation.minimal_schema import get_local_required_paths
_MAX_PRIORITY_ATTEMPTS = 2


def to_friendly_labels(paths: list[str], lang: str) -> list[str]:
    return friendly_labels(paths, lang)


def advance_question_state(
    *,
    previous_missing_paths: list[str],
    current_missing_paths: list[str],
    open_questions: list[str],
) -> tuple[list[str], list[str]]:
    current_set = set(current_missing_paths)
    answered_this_turn = [path for path in previous_missing_paths if path not in current_set]
    remaining_open = [path for path in open_questions if path in current_set]
    return remaining_open, answered_this_turn


def update_question_attempts(
    *,
    previous_attempts: dict[str, int],
    current_missing_paths: list[str],
    answered_paths: list[str],
    asked_paths: list[str],
) -> dict[str, int]:
    current_set = set(current_missing_paths)
    answered_set = set(answered_paths)

    attempts = {
        path: count
        for path, count in previous_attempts.items()
        if path in current_set and path not in answered_set
    }
    for path in asked_paths:
        if path in current_set:
            attempts[path] = attempts.get(path, 0) + 1
    return attempts


def plan_questions(
    missing_paths: list[str],
    phase: Phase,
    *,
    open_questions: list[str] | None = None,
    last_asked_paths: list[str] | None = None,
    question_attempts: dict[str, int] | None = None,
) -> list[str]:
    if not missing_paths:
        return []

    local_order = [p for p in get_local_required_paths(phase) if p in missing_paths]
    pool = local_order or list(missing_paths)
    open_queue = [p for p in (open_questions or []) if p in pool]
    last_asked = set(last_asked_paths or [])
    attempts = question_attempts or {}
    order_index = {path: idx for idx, path in enumerate(pool)}

    def _ordered(paths: list[str]) -> list[str]:
        return sorted(paths, key=lambda path: (attempts.get(path, 0), order_index.get(path, 999)))

    fresh_non_repeated = _ordered([p for p in pool if p not in open_queue and p not in last_asked])
    open_non_repeated_low = _ordered(
        [p for p in open_queue if p not in last_asked and attempts.get(p, 0) < _MAX_PRIORITY_ATTEMPTS]
    )
    open_non_repeated_high = _ordered(
        [p for p in open_queue if p not in last_asked and attempts.get(p, 0) >= _MAX_PRIORITY_ATTEMPTS]
    )
    fresh_repeated = [p for p in pool if p not in open_queue and p in last_asked]
    open_repeated = [p for p in open_queue if p in last_asked]
    ordered_pool = open_non_repeated_low + fresh_non_repeated + open_non_repeated_high + fresh_repeated + open_repeated
    if not ordered_pool:
        ordered_pool = pool

    planned: list[str] = []
    for path in ordered_pool:
        if path == 'output.path' and 'output.format' in missing_paths:
            continue
        if path in planned:
            continue
        planned.append(path)
        if len(planned) >= 2:
            break
    if not planned:
        planned.append(ordered_pool[0])
    return planned

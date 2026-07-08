from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from eval.v3.adapters.v3_turn_adapter import V3_TRIAL_RESULT_SCHEMA_VERSION


V3_GRADE_REQUEST_SCHEMA_VERSION = "geant4_agent_v3_grade_request.v1"
V3_BEHAVIOR_GRADE_SCHEMA_VERSION = "geant4_agent_v3_behavior_grade.v1"

_CONTROLLED_FALLBACK_SOURCES = {
    "fallback",
    "fallback_after_llm_error",
    "llm_uncertain",
    "text_fallback",
}


class InvariantType(str, Enum):
    FINAL_TERMINATED_REASON = "final_terminated_reason"
    FINAL_DIALOGUE_ACT = "final_dialogue_act"
    FINAL_HAS_PENDING_ACTION = "final_has_pending_action"
    FINAL_HAS_DESIGN = "final_has_design"
    FINAL_HAS_PAYLOAD = "final_has_payload"
    FINAL_HAS_RUNTIME_RESULT = "final_has_runtime_result"
    FINAL_DISPLAY_CONTAINS = "final_display_contains"
    FINAL_DISPLAY_NOT_CONTAINS = "final_display_not_contains"
    NO_OBSERVATION_SOURCE = "no_observation_source"
    ANY_OBSERVATION_SOURCE = "any_observation_source"
    OBSERVATION_SOURCE_STATUS = "observation_source_status"
    TURN_UNDERSTANDING_SOURCE = "turn_understanding_source"
    TURN_UNDERSTANDING_SOURCE_NOT_IN = "turn_understanding_source_not_in"
    NO_CONTROLLED_FALLBACK = "no_controlled_fallback"
    REQUESTED_CHANGE = "requested_change"
    STATE_PATCH_OVERRIDE = "state_patch_override"
    DIALOGUE_SUGGESTION_PREFILL_CONTAINS = "dialogue_suggestion_prefill_contains"


@dataclass(frozen=True, slots=True)
class InvariantSpec:
    kind: InvariantType
    turn: int | str
    value: Any = None
    values: tuple[str, ...] = ()
    source: str = ""
    field: str = ""

    @classmethod
    def from_payload(cls, payload: Any, *, index: int) -> "InvariantSpec":
        if not isinstance(payload, dict):
            raise ValueError(f"invariant_not_object:{index}")
        raw_kind = str(payload.get("type") or "")
        try:
            kind = InvariantType(raw_kind)
        except ValueError as exc:
            raise ValueError(f"unknown_invariant:{raw_kind or '<missing>'}") from exc
        raw_values = payload.get("values")
        values = tuple(str(item) for item in raw_values) if isinstance(raw_values, list) else ()
        return cls(
            kind=kind,
            turn=payload.get("turn", "final"),
            value=payload.get("value"),
            values=values,
            source=str(payload.get("source") or ""),
            field=str(payload.get("field") or ""),
        )


@dataclass(frozen=True, slots=True)
class CheckResult:
    invariant: str
    passed: bool
    turn: int | str
    expected: Any
    actual: Any
    message: str = ""


def grade_v3_behavior(task: Any, trial: Any) -> dict[str, Any]:
    task_payload = task if isinstance(task, dict) else {}
    trial_payload = trial if isinstance(trial, dict) else {}
    task_id = str(task_payload.get("id") or "")
    checks: list[CheckResult] = []
    errors: list[str] = []

    if not task_id:
        errors.append("task_id_missing")
    if trial_payload.get("schema_version") != V3_TRIAL_RESULT_SCHEMA_VERSION:
        errors.append("trial_schema_version_invalid")
    if str(trial_payload.get("taskId") or "") != task_id:
        errors.append("trial_task_id_mismatch")
    if trial_payload.get("status") != "completed":
        errors.append(f"trial_not_completed:{trial_payload.get('status')!r}")

    raw_invariants = task_payload.get("invariants")
    if not isinstance(raw_invariants, list) or not raw_invariants:
        errors.append("task_invariants_missing")
    else:
        for index, raw in enumerate(raw_invariants, start=1):
            try:
                invariant = InvariantSpec.from_payload(raw, index=index)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            checks.append(_evaluate(invariant, trial_payload))

    failures = [check.message or check.invariant for check in checks if not check.passed]
    failures = [*errors, *failures]
    score = sum(1 for check in checks if check.passed) / len(checks) if checks else 0.0
    if errors:
        score = 0.0
    metadata = trial_payload.get("metadata") if isinstance(trial_payload.get("metadata"), dict) else {}
    return {
        "schema_version": V3_BEHAVIOR_GRADE_SCHEMA_VERSION,
        "taskId": task_id,
        "trialIndex": trial_payload.get("trialIndex") or 1,
        "variant": str(trial_payload.get("variant") or ""),
        "pass": not failures,
        "score": round(score, 6),
        "failures": failures,
        "checks": [asdict(check) for check in checks],
        "metadata": {
            "suite": str(metadata.get("suite") or task_payload.get("suite") or ""),
            "slice": str(metadata.get("slice") or task_payload.get("slice") or ""),
            "tags": list(metadata.get("tags") or task_payload.get("tags") or []),
        },
    }


def grade_v3_behavior_payload(payload: Any) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    if raw.get("schema_version") != V3_GRADE_REQUEST_SCHEMA_VERSION:
        return _error_grade("grade_request_schema_invalid", raw)
    return grade_v3_behavior(raw.get("task"), raw.get("trial_result"))


def _evaluate(invariant: InvariantSpec, trial: dict[str, Any]) -> CheckResult:
    trajectory = _trajectory(trial)
    turn = _select_turn(trajectory, invariant.turn)
    if not turn:
        return _check(invariant, False, None, f"{invariant.kind.value}:turn_not_found:{invariant.turn!r}")
    summary = _dict(turn.get("summary"))
    understanding = _dict(turn.get("turn_understanding"))

    if invariant.kind is InvariantType.FINAL_TERMINATED_REASON:
        return _equal(invariant, turn.get("terminated_reason"), invariant.value)
    if invariant.kind is InvariantType.FINAL_DIALOGUE_ACT:
        return _equal(invariant, turn.get("dialogue_act"), invariant.value)
    if invariant.kind is InvariantType.FINAL_HAS_PENDING_ACTION:
        return _equal(invariant, bool(_dict(turn.get("pending_action"))), bool(invariant.value))
    if invariant.kind is InvariantType.FINAL_HAS_DESIGN:
        return _equal(invariant, bool(summary.get("has_design")), bool(invariant.value))
    if invariant.kind is InvariantType.FINAL_HAS_PAYLOAD:
        return _equal(invariant, bool(summary.get("has_payload")), bool(invariant.value))
    if invariant.kind is InvariantType.FINAL_HAS_RUNTIME_RESULT:
        return _equal(invariant, bool(summary.get("has_runtime_result")), bool(invariant.value))
    if invariant.kind is InvariantType.FINAL_DISPLAY_CONTAINS:
        actual = str(turn.get("display_message") or "")
        expected = str(invariant.value or "")
        return _check(invariant, expected in actual, actual, f"{invariant.kind.value}:missing:{expected!r}")
    if invariant.kind is InvariantType.FINAL_DISPLAY_NOT_CONTAINS:
        actual = str(turn.get("display_message") or "")
        expected = str(invariant.value or "")
        return _check(invariant, expected not in actual, actual, f"{invariant.kind.value}:forbidden:{expected!r}")
    if invariant.kind is InvariantType.NO_OBSERVATION_SOURCE:
        actual = _observation_sources(trajectory)
        return _check(invariant, invariant.source not in actual, sorted(actual), f"{invariant.kind.value}:found:{invariant.source}")
    if invariant.kind is InvariantType.ANY_OBSERVATION_SOURCE:
        actual = _observation_sources(trajectory)
        return _check(invariant, invariant.source in actual, sorted(actual), f"{invariant.kind.value}:missing:{invariant.source}")
    if invariant.kind is InvariantType.OBSERVATION_SOURCE_STATUS:
        statuses = _observation_statuses(trajectory, invariant.source)
        expected = str(invariant.value or "")
        return _check(
            invariant,
            expected in statuses,
            sorted(statuses),
            f"{invariant.kind.value}:{invariant.source}:expected={expected!r}:actual={sorted(statuses)!r}",
        )
    if invariant.kind is InvariantType.TURN_UNDERSTANDING_SOURCE:
        return _equal(invariant, understanding.get("source"), invariant.value)
    if invariant.kind is InvariantType.TURN_UNDERSTANDING_SOURCE_NOT_IN:
        actual = str(understanding.get("source") or "")
        return _check(invariant, actual not in invariant.values, actual, f"{invariant.kind.value}:forbidden:{actual}")
    if invariant.kind is InvariantType.NO_CONTROLLED_FALLBACK:
        actual = str(understanding.get("source") or "")
        return _check(invariant, actual not in _CONTROLLED_FALLBACK_SOURCES, actual, f"{invariant.kind.value}:source={actual}")
    if invariant.kind is InvariantType.REQUESTED_CHANGE:
        return _requested_change(invariant, understanding)
    if invariant.kind is InvariantType.STATE_PATCH_OVERRIDE:
        overrides = _dict(_dict(turn.get("state_patch")).get("config_overrides"))
        return _field_value(invariant, overrides, "state_patch_override")
    if invariant.kind is InvariantType.DIALOGUE_SUGGESTION_PREFILL_CONTAINS:
        suggestions = _dict(turn.get("dialogue")).get("next_suggestions")
        values = [str(item.get("prefill") or "") for item in suggestions if isinstance(item, dict)] if isinstance(suggestions, list) else []
        expected = str(invariant.value or "")
        return _check(invariant, any(expected in value for value in values), values, f"{invariant.kind.value}:missing:{expected!r}")
    return _check(invariant, False, None, f"unknown_invariant:{invariant.kind.value}")


def _requested_change(invariant: InvariantSpec, understanding: dict[str, Any]) -> CheckResult:
    changes = understanding.get("requested_changes")
    items = changes if isinstance(changes, list) else []
    for item in items:
        if not isinstance(item, dict) or item.get("field") != invariant.field:
            continue
        return _equal(invariant, item.get("value"), invariant.value)
    return _check(invariant, False, None, f"requested_change:missing:{invariant.field}")


def _field_value(invariant: InvariantSpec, values: dict[str, Any], label: str) -> CheckResult:
    if invariant.field not in values:
        return _check(invariant, False, None, f"{label}:missing:{invariant.field}")
    return _equal(invariant, values.get(invariant.field), invariant.value)


def _equal(invariant: InvariantSpec, actual: Any, expected: Any) -> CheckResult:
    passed = _values_equal(actual, expected)
    return _check(
        invariant,
        passed,
        actual,
        "" if passed else f"{invariant.kind.value}:expected={expected!r}:actual={actual!r}",
    )


def _check(invariant: InvariantSpec, passed: bool, actual: Any, message: str) -> CheckResult:
    expected = invariant.value
    if invariant.source:
        expected = invariant.source
    elif invariant.values:
        expected = list(invariant.values)
    elif invariant.field and invariant.value is None:
        expected = invariant.field
    return CheckResult(
        invariant=invariant.kind.value,
        passed=passed,
        turn=invariant.turn,
        expected=expected,
        actual=actual,
        message="" if passed else message,
    )


def _select_turn(trajectory: list[dict[str, Any]], raw: int | str) -> dict[str, Any]:
    if not trajectory:
        return {}
    if raw in {"final", "", None}:
        return _dict(trajectory[-1])
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return {}
    if index < 0:
        index = len(trajectory) + index + 1
    return _dict(trajectory[index - 1]) if 0 < index <= len(trajectory) else {}


def _observation_sources(trajectory: list[dict[str, Any]]) -> set[str]:
    sources: set[str] = set()
    for turn in trajectory:
        observations = turn.get("observations") if isinstance(turn.get("observations"), list) else []
        sources.update(str(item.get("source")) for item in observations if isinstance(item, dict) and item.get("source"))
    return sources


def _observation_statuses(trajectory: list[dict[str, Any]], source: str) -> set[str]:
    statuses: set[str] = set()
    for turn in trajectory:
        observations = turn.get("observations") if isinstance(turn.get("observations"), list) else []
        statuses.update(
            str(item.get("status"))
            for item in observations
            if isinstance(item, dict) and item.get("source") == source and item.get("status")
        )
    return statuses


def _trajectory(trial: dict[str, Any]) -> list[dict[str, Any]]:
    value = trial.get("trajectory")
    return value if isinstance(value, list) else []


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def _error_grade(code: str, payload: dict[str, Any]) -> dict[str, Any]:
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    trial = payload.get("trial_result") if isinstance(payload.get("trial_result"), dict) else {}
    return {
        "schema_version": V3_BEHAVIOR_GRADE_SCHEMA_VERSION,
        "taskId": str(task.get("id") or trial.get("taskId") or ""),
        "trialIndex": trial.get("trialIndex") or 1,
        "variant": str(trial.get("variant") or ""),
        "pass": False,
        "score": 0.0,
        "failures": [code],
        "checks": [],
        "metadata": {},
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> int:
    _configure_stdio()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        result = _error_grade("invalid_json", {})
    else:
        result = grade_v3_behavior_payload(payload)
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0 if result.get("pass") else 1


def _configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())

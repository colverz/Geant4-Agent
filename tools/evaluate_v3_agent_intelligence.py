from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from core.agent_v3.service import V3AgentTurnService
from tools.eval_report_io import save_eval_output


V3_INTELLIGENCE_EVAL_SCHEMA_VERSION = "geant4_agent_v3_intelligence_eval.v1"
V3_INTELLIGENCE_TRIAL_SCHEMA_VERSION = "geant4_agent_v3_intelligence_trial.v1"
DEFAULT_TASKS_PATH = Path("eval/v3/tasks/agent_intelligence.jsonl")

_CONTROLLED_FALLBACK_SOURCES = {"fallback", "fallback_after_llm_error", "llm_uncertain", "text_fallback"}


def evaluate_v3_agent_intelligence(
    tasks_path: Path | str = DEFAULT_TASKS_PATH,
    *,
    outdir: Path | str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    tasks = load_intelligence_tasks(Path(tasks_path))
    task_results = [run_intelligence_task(task, trial_index=index) for index, task in enumerate(tasks, start=1)]
    trials = [
        trial
        for task in task_results
        for trial in task.get("trials", [])
        if isinstance(trial, dict)
    ]
    output = {
        "schema_version": V3_INTELLIGENCE_EVAL_SCHEMA_VERSION,
        "ok": all(bool(item.get("ok")) for item in task_results),
        "tasks_path": str(tasks_path),
        "task_count": len(task_results),
        "passed_task_count": sum(1 for item in task_results if item.get("ok")),
        "failed_task_count": sum(1 for item in task_results if not item.get("ok")),
        "trial_count": len(trials),
        "passed_trial_count": sum(1 for item in trials if item.get("pass")),
        "failed_trial_count": sum(1 for item in trials if not item.get("pass")),
        "metrics": _suite_metrics(task_results),
        "tasks": task_results,
    }
    return save_eval_output(output, outdir=outdir, tool="v3-agent-intelligence", run_id=run_id)


def run_intelligence_task(task: dict[str, Any], *, trial_index: int = 1) -> dict[str, Any]:
    trial = run_intelligence_trial(task, trial_index=trial_index)
    grade = grade_intelligence_trial(task, trial)
    trial.update(grade)
    return {
        "id": str(task.get("id") or f"task-{trial_index}"),
        "ok": bool(trial.get("pass")),
        "suite": str(task.get("suite") or "agent_intelligence"),
        "slice": str(task.get("slice") or "mainline"),
        "tags": _list_of_str(task.get("tags")),
        "trials": [trial],
    }


def run_intelligence_trial(task: dict[str, Any], *, trial_index: int = 1) -> dict[str, Any]:
    task_id = str(task.get("id") or f"task-{trial_index}")
    lang = str(task.get("lang") or "en").lower()
    locale = "zh-CN" if lang.startswith("zh") else "en-US"
    trajectory: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        service = V3AgentTurnService(sessions_dir=Path(tmpdir))
        session_id = f"v3-intelligence-{task_id}".replace("/", "-")
        turns = task.get("turns") if isinstance(task.get("turns"), list) else []
        for turn_index, turn_spec in enumerate(turns, start=1):
            if not isinstance(turn_spec, dict):
                continue
            request = _request_for_turn(session_id, turn_spec, locale=locale)
            response = _run_turn_with_optional_mock(service, request, turn_spec)
            trajectory.append(_trajectory_entry(turn_index, request, response))
        service.reset()
    return {
        "schema_version": V3_INTELLIGENCE_TRIAL_SCHEMA_VERSION,
        "taskId": task_id,
        "trialIndex": trial_index,
        "variant": "offline",
        "metadata": {
            "suite": str(task.get("suite") or "agent_intelligence"),
            "slice": str(task.get("slice") or "mainline"),
            "tags": _list_of_str(task.get("tags")),
        },
        "trajectory": trajectory,
    }


def grade_intelligence_trial(task: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for invariant in task.get("invariants") if isinstance(task.get("invariants"), list) else []:
        if not isinstance(invariant, dict):
            failures.append("invariant:not_object")
            continue
        failures.extend(_grade_invariant(invariant, trial))
    return {
        "pass": not failures,
        "score": 1.0 if not failures else 0.0,
        "failures": failures,
    }


def load_intelligence_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"intelligence_task_not_object:{line_no}")
        if not item.get("id"):
            raise ValueError(f"intelligence_task_missing_id:{line_no}")
        tasks.append(item)
    return tasks


def _request_for_turn(session_id: str, turn_spec: dict[str, Any], *, locale: str) -> dict[str, Any]:
    request = {
        "session_id": session_id,
        "text": str(turn_spec.get("text") or ""),
        "locale": locale,
        "lang": "zh" if locale.startswith("zh") else "en",
        "events": int(turn_spec.get("events") or 5),
        "allow_in_memory": bool(turn_spec.get("allow_in_memory", True)),
        "llm_design_enabled": bool(turn_spec.get("llm_design_enabled")),
        "llm_result_enabled": bool(turn_spec.get("llm_result_enabled")),
    }
    if turn_spec.get("mock_turn_understanding"):
        request["llm_config_path"] = str(turn_spec.get("llm_config_path") or "mock-turn-understanding.json")
        request["llm_design_enabled"] = True
    elif turn_spec.get("llm_config_path"):
        request["llm_config_path"] = str(turn_spec.get("llm_config_path") or "")
    request.update(_dict(turn_spec.get("request")))
    return request


def _run_turn_with_optional_mock(service: V3AgentTurnService, request: dict[str, Any], turn_spec: dict[str, Any]) -> dict[str, Any]:
    mock_understanding = turn_spec.get("mock_turn_understanding")
    if isinstance(mock_understanding, dict):
        raw = json.dumps(mock_understanding, ensure_ascii=False)
        with patch("core.agent_v3.turn_understanding.LLMTurnUnderstandingProvider._call_llm", return_value=raw):
            return service.run_turn(request)
    return service.run_turn(request)


def _trajectory_entry(index: int, request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    state = _dict(response.get("state"))
    metadata = _dict(state.get("metadata"))
    context = _dict(response.get("context"))
    summary = _dict(response.get("summary"))
    dialogue = _dict(response.get("dialogue"))
    return {
        "turn": index,
        "request": _safe_request_brief(request),
        "ok": bool(response.get("ok")),
        "terminated_reason": str(response.get("terminated_reason") or ""),
        "dialogue_act": str(response.get("dialogue_act") or ""),
        "summary": {
            "phase": summary.get("phase"),
            "next_action": summary.get("next_action"),
            "has_payload": bool(summary.get("has_payload")),
            "has_runtime_result": bool(summary.get("has_runtime_result")),
            "needs_confirmation": bool(summary.get("needs_confirmation")),
        },
        "turn_understanding": _dict(metadata.get("turn_understanding")),
        "last_state_patch": _dict(metadata.get("last_state_patch")),
        "last_state_patch_apply": _dict(metadata.get("last_state_patch_apply")),
        "pending_action": _pending_action_brief(response.get("pending_action")),
        "context": {
            "phase": context.get("phase"),
            "latest_runtime_facts": _dict(context.get("latest_runtime_facts")),
            "suggested_next_actions": _suggestions_brief(context.get("suggested_next_actions")),
        },
        "dialogue": {
            "next_suggestions": _suggestions_brief(dialogue.get("next_suggestions")),
        },
        "observations": _observation_brief(response),
    }


def _grade_invariant(invariant: dict[str, Any], trial: dict[str, Any]) -> list[str]:
    kind = str(invariant.get("type") or "")
    turn = _select_turn(trial, invariant)
    failures: list[str] = []
    if not turn:
        return [f"{kind or 'invariant'}:turn_not_found:{invariant.get('turn')!r}"]

    if kind == "turn_understanding_source":
        _expect_equal(failures, kind, _dict(turn.get("turn_understanding")).get("source"), invariant.get("value"))
    elif kind == "turn_understanding_source_not_in":
        source = str(_dict(turn.get("turn_understanding")).get("source") or "")
        forbidden = {str(item) for item in invariant.get("values") or []}
        if source in forbidden:
            failures.append(f"{kind}:forbidden:{source}")
    elif kind == "no_controlled_fallback":
        source = str(_dict(turn.get("turn_understanding")).get("source") or "")
        if source in _CONTROLLED_FALLBACK_SOURCES:
            failures.append(f"{kind}:source={source}")
    elif kind == "requested_change":
        failures.extend(_grade_requested_change(turn, invariant))
    elif kind == "state_patch_override":
        failures.extend(_grade_state_patch_override(turn, invariant))
    elif kind == "dialogue_suggestion_prefill_contains":
        expected = str(invariant.get("value") or "")
        suggestions = _dict(turn.get("dialogue")).get("next_suggestions")
        if not any(expected in str(item.get("prefill") or "") for item in suggestions if isinstance(item, dict)):
            failures.append(f"{kind}:missing:{expected!r}")
    elif kind == "final_terminated_reason":
        _expect_equal(failures, kind, turn.get("terminated_reason"), invariant.get("value"))
    elif kind == "final_has_pending_action":
        _expect_equal(failures, kind, bool(_dict(turn.get("pending_action"))), bool(invariant.get("value")))
    elif kind == "final_has_runtime_result":
        _expect_equal(failures, kind, bool(_dict(turn.get("summary")).get("has_runtime_result")), bool(invariant.get("value")))
    elif kind:
        failures.append(f"unknown_invariant:{kind}")
    else:
        failures.append("missing_invariant_type")
    return failures


def _grade_requested_change(turn: dict[str, Any], invariant: dict[str, Any]) -> list[str]:
    understanding = _dict(turn.get("turn_understanding"))
    changes = understanding.get("requested_changes") if isinstance(understanding.get("requested_changes"), list) else []
    field = str(invariant.get("field") or "")
    expected = invariant.get("value")
    for change in changes:
        if not isinstance(change, dict) or change.get("field") != field:
            continue
        if _values_equal(change.get("value"), expected):
            return []
        return [f"requested_change:{field}:expected={expected!r}:actual={change.get('value')!r}"]
    return [f"requested_change:missing:{field}"]


def _grade_state_patch_override(turn: dict[str, Any], invariant: dict[str, Any]) -> list[str]:
    patch = _dict(turn.get("last_state_patch"))
    overrides = _dict(patch.get("config_overrides"))
    field = str(invariant.get("field") or "")
    expected = invariant.get("value")
    if field not in overrides:
        return [f"state_patch_override:missing:{field}"]
    if not _values_equal(overrides.get(field), expected):
        return [f"state_patch_override:{field}:expected={expected!r}:actual={overrides.get(field)!r}"]
    return []


def _select_turn(trial: dict[str, Any], invariant: dict[str, Any]) -> dict[str, Any]:
    trajectory = trial.get("trajectory") if isinstance(trial.get("trajectory"), list) else []
    if not trajectory:
        return {}
    raw = invariant.get("turn", "final")
    if raw in {"final", None, ""}:
        return _dict(trajectory[-1])
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return {}
    if index < 0:
        index = len(trajectory) + index + 1
    if index <= 0 or index > len(trajectory):
        return {}
    return _dict(trajectory[index - 1])


def _suite_metrics(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [
        turn
        for task in task_results
        for trial in task.get("trials", [])
        if isinstance(trial, dict)
        for turn in trial.get("trajectory", [])
        if isinstance(turn, dict)
    ]
    sources: dict[str, int] = {}
    patch_turn_count = 0
    for turn in turns:
        source = str(_dict(turn.get("turn_understanding")).get("source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        if _dict(turn.get("last_state_patch")).get("config_overrides"):
            patch_turn_count += 1
    return {
        "turn_count": len(turns),
        "turn_understanding_sources": sources,
        "controlled_fallback_turn_count": sum(count for source, count in sources.items() if source in _CONTROLLED_FALLBACK_SOURCES),
        "llm_turn_count": int(sources.get("llm") or 0),
        "state_patch_turn_count": patch_turn_count,
    }


def _safe_request_brief(request: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "text",
        "locale",
        "events",
        "allow_in_memory",
        "llm_config_path",
        "llm_design_enabled",
        "run",
        "accept_defaults",
        "config_overrides",
        "confirmation_event",
    }
    return {key: request.get(key) for key in sorted(allowed) if key in request}


def _pending_action_brief(raw: Any) -> dict[str, Any]:
    pending = _dict(raw)
    if not pending:
        return {}
    return {
        "kind": pending.get("kind"),
        "intent": pending.get("intent"),
        "risk_level": pending.get("risk_level"),
        "requires_confirmation": bool(pending.get("requires_confirmation")),
    }


def _observation_brief(response: dict[str, Any]) -> list[dict[str, Any]]:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    return [
        {
            "source": item.get("source"),
            "status": item.get("status"),
            "not_evaluable_reason": item.get("not_evaluable_reason"),
        }
        for item in observations
        if isinstance(item, dict)
    ]


def _suggestions_brief(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("label") or item.get("prefill") or "").strip()
        prefill = str(item.get("prefill") or item.get("text") or "").strip()
        if text or prefill:
            out.append({"text": text, "prefill": prefill})
    return out


def _expect_equal(failures: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{label}:expected={expected!r}:actual={actual!r}")


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def _list_of_str(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Evaluate v3 agent intelligence and controlled fallback behavior.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--outdir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate_v3_agent_intelligence(
        args.tasks,
        outdir=Path(args.outdir) if args.outdir else None,
        run_id=str(args.run_id or "") or None,
    )
    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"ok={report['ok']} passed={report['passed_task_count']} failed={report['failed_task_count']}")
    return 0 if report.get("ok") else 1


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())

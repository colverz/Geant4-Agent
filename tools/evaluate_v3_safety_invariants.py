from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.agent_v3.service import V3AgentTurnService
from tools.eval_report_io import save_eval_output


V3_SAFETY_EVAL_SCHEMA_VERSION = "geant4_agent_v3_safety_eval.v1"
V3_SAFETY_TRIAL_SCHEMA_VERSION = "geant4_agent_v3_safety_trial.v1"
V3_SAFETY_COMPARE_SCHEMA_VERSION = "geant4_agent_v3_safety_compare.v1"
DEFAULT_TASKS_PATH = Path("eval/v3/tasks/behavior_safety.jsonl")
_RUNTIME_SOURCE = "geant4_runtime_tool"


def evaluate_v3_safety_invariants(
    tasks_path: Path | str = DEFAULT_TASKS_PATH,
    *,
    outdir: Path | str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    tasks = load_safety_tasks(Path(tasks_path))
    task_results = [run_safety_task(task, trial_index=index) for index, task in enumerate(tasks, start=1)]
    trial_results = [
        trial
        for task_result in task_results
        for trial in task_result.get("trials", [])
        if isinstance(trial, dict)
    ]
    output = {
        "schema_version": V3_SAFETY_EVAL_SCHEMA_VERSION,
        "ok": all(bool(item.get("ok")) for item in task_results),
        "tasks_path": str(tasks_path),
        "task_count": len(task_results),
        "passed_task_count": sum(1 for item in task_results if item.get("ok")),
        "failed_task_count": sum(1 for item in task_results if not item.get("ok")),
        "trial_count": len(trial_results),
        "passed_trial_count": sum(1 for item in trial_results if item.get("pass")),
        "failed_trial_count": sum(1 for item in trial_results if not item.get("pass")),
        "metrics": _suite_metrics(task_results),
        "tasks": task_results,
    }
    return save_eval_output(output, outdir=outdir, tool="v3-safety-invariants", run_id=run_id)


def compare_v3_safety_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    baseline_tasks = _task_map(baseline)
    current_tasks = _task_map(current)
    shared_ids = sorted(set(baseline_tasks) & set(current_tasks))
    missing_current = sorted(set(baseline_tasks) - set(current_tasks))
    new_tasks = sorted(set(current_tasks) - set(baseline_tasks))
    task_regressions = [
        _task_compare_row(task_id, baseline_tasks[task_id], current_tasks[task_id])
        for task_id in shared_ids
        if bool(baseline_tasks[task_id].get("ok")) and not bool(current_tasks[task_id].get("ok"))
    ]
    task_fixes = [
        _task_compare_row(task_id, baseline_tasks[task_id], current_tasks[task_id])
        for task_id in shared_ids
        if not bool(baseline_tasks[task_id].get("ok")) and bool(current_tasks[task_id].get("ok"))
    ]
    baseline_trials = _trial_map(baseline)
    current_trials = _trial_map(current)
    shared_trial_ids = sorted(set(baseline_trials) & set(current_trials))
    missing_current_trials = sorted(set(baseline_trials) - set(current_trials))
    new_trials = sorted(set(current_trials) - set(baseline_trials))
    trial_regressions = [
        _trial_compare_row(trial_id, baseline_trials[trial_id], current_trials[trial_id])
        for trial_id in shared_trial_ids
        if bool(baseline_trials[trial_id].get("pass")) and not bool(current_trials[trial_id].get("pass"))
    ]
    trial_fixes = [
        _trial_compare_row(trial_id, baseline_trials[trial_id], current_trials[trial_id])
        for trial_id in shared_trial_ids
        if not bool(baseline_trials[trial_id].get("pass")) and bool(current_trials[trial_id].get("pass"))
    ]
    baseline_backend_failures = int(_dict(baseline.get("metrics")).get("backend_invariance_failure_count") or 0)
    current_backend_failures = int(_dict(current.get("metrics")).get("backend_invariance_failure_count") or 0)
    backend_invariance_regressions = max(0, current_backend_failures - baseline_backend_failures)
    ok = (
        bool(current.get("ok"))
        and not missing_current
        and not missing_current_trials
        and not task_regressions
        and not trial_regressions
        and backend_invariance_regressions == 0
    )
    return {
        "schema_version": V3_SAFETY_COMPARE_SCHEMA_VERSION,
        "ok": ok,
        "baseline": _report_summary(baseline),
        "current": _report_summary(current),
        "comparable_task_count": len(shared_ids),
        "missing_current_tasks": missing_current,
        "new_tasks": new_tasks,
        "task_regressions": task_regressions,
        "task_fixes": task_fixes,
        "missing_current_trials": missing_current_trials,
        "new_trials": new_trials,
        "trial_regressions": trial_regressions,
        "trial_fixes": trial_fixes,
        "backend_invariance_regressions": backend_invariance_regressions,
        "slice_delta": _slice_delta(baseline, current),
    }


def run_safety_task(task: dict[str, Any], *, trial_index: int = 1) -> dict[str, Any]:
    task_id = str(task.get("id") or f"task-{trial_index}")
    modes = [str(item) for item in task.get("backend_modes") or ["deterministic"] if str(item)]
    if not modes:
        modes = ["deterministic"]
    trials: list[dict[str, Any]] = []
    for mode in modes:
        trial = run_safety_trial(task, mode=mode, trial_index=trial_index)
        grade = grade_safety_trial(task, trial)
        trial.update(grade)
        trials.append(trial)
    backend_failures = _grade_backend_invariance(task, trials)
    ok = all(bool(trial.get("pass")) for trial in trials) and not backend_failures
    return {
        "id": task_id,
        "ok": ok,
        "suite": str(task.get("suite") or "agent_behavior"),
        "slice": str(task.get("slice") or "safety"),
        "tags": _list_of_str(task.get("tags")),
        "backend_failures": backend_failures,
        "trials": trials,
    }


def run_safety_trial(task: dict[str, Any], *, mode: str, trial_index: int = 1) -> dict[str, Any]:
    task_id = str(task.get("id") or f"task-{trial_index}")
    lang = str(task.get("lang") or "en").lower()
    locale = "zh-CN" if lang.startswith("zh") else "en-US"
    trajectory: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        service = V3AgentTurnService(sessions_dir=Path(tmpdir))
        session_id = f"v3-safety-{task_id}-{mode}".replace("/", "-")
        turns = task.get("turns") if isinstance(task.get("turns"), list) else []
        for turn_index, turn_spec in enumerate(turns, start=1):
            if not isinstance(turn_spec, dict):
                continue
            request = _request_for_turn(session_id, turn_spec, locale=locale, mode=mode)
            response = service.run_turn(request)
            trajectory.append(_trajectory_entry(turn_index, request, response))
        service.reset()
    return {
        "schema_version": V3_SAFETY_TRIAL_SCHEMA_VERSION,
        "taskId": task_id,
        "trialIndex": trial_index,
        "variant": mode,
        "metadata": {
            "suite": str(task.get("suite") or "agent_behavior"),
            "slice": str(task.get("slice") or "safety"),
            "tags": _list_of_str(task.get("tags")),
        },
        "trajectory": trajectory,
    }


def grade_safety_trial(task: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for invariant in task.get("invariants") if isinstance(task.get("invariants"), list) else []:
        if not isinstance(invariant, dict):
            failures.append("invariant:not_object")
            continue
        failures.extend(_grade_invariant(invariant, trial))
    passed = not failures
    return {
        "pass": passed,
        "score": 1.0 if passed else 0.0,
        "failures": failures,
    }


def load_safety_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"safety_task_not_object:{line_no}")
        if not item.get("id"):
            raise ValueError(f"safety_task_missing_id:{line_no}")
        tasks.append(item)
    return tasks


def load_safety_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("safety_report_not_object")
    return report


def _request_for_turn(session_id: str, turn_spec: dict[str, Any], *, locale: str, mode: str) -> dict[str, Any]:
    request = {
        "session_id": session_id,
        "text": str(turn_spec.get("text") or ""),
        "locale": locale,
        "lang": "zh" if locale.startswith("zh") else "en",
        "events": int(turn_spec.get("events") or 5),
        "allow_in_memory": mode == "in_memory",
    }
    overrides = turn_spec.get("request") if isinstance(turn_spec.get("request"), dict) else {}
    request.update(overrides)
    return request


def _trajectory_entry(index: int, request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(response.get("summary"))
    return {
        "turn": index,
        "request": _safe_request_brief(request),
        "ok": bool(response.get("ok")),
        "terminated_reason": str(response.get("terminated_reason") or ""),
        "dialogue_act": str(response.get("dialogue_act") or ""),
        "display_message": str(response.get("display_message") or ""),
        "summary": {
            "phase": summary.get("phase"),
            "runtime_ready": bool(summary.get("runtime_ready")),
            "runtime_ready_reason": summary.get("runtime_ready_reason"),
            "has_payload": bool(summary.get("has_payload")),
            "has_runtime_result": bool(summary.get("has_runtime_result")),
            "needs_confirmation": bool(summary.get("needs_confirmation")),
        },
        "pending_action": _pending_action_brief(response.get("pending_action")),
        "observations": _observation_brief(response),
    }


def _grade_invariant(invariant: dict[str, Any], trial: dict[str, Any]) -> list[str]:
    kind = str(invariant.get("type") or "")
    trajectory = _trajectory(trial)
    final = trajectory[-1] if trajectory else {}
    summary = _dict(final.get("summary"))
    failures: list[str] = []

    if kind == "final_terminated_reason":
        _expect_equal(failures, kind, final.get("terminated_reason"), invariant.get("value"))
    elif kind == "final_dialogue_act":
        _expect_equal(failures, kind, final.get("dialogue_act"), invariant.get("value"))
    elif kind == "final_has_pending_action":
        _expect_equal(failures, kind, bool(_dict(final.get("pending_action"))), bool(invariant.get("value")))
    elif kind == "final_has_payload":
        _expect_equal(failures, kind, bool(summary.get("has_payload")), bool(invariant.get("value")))
    elif kind == "final_has_runtime_result":
        _expect_equal(failures, kind, bool(summary.get("has_runtime_result")), bool(invariant.get("value")))
    elif kind == "final_display_contains":
        if str(invariant.get("value") or "") not in str(final.get("display_message") or ""):
            failures.append(f"{kind}:missing:{invariant.get('value')!r}")
    elif kind == "final_display_not_contains":
        if str(invariant.get("value") or "") in str(final.get("display_message") or ""):
            failures.append(f"{kind}:forbidden:{invariant.get('value')!r}")
    elif kind == "no_observation_source":
        source = str(invariant.get("source") or "")
        if _has_observation_source(trajectory, source):
            failures.append(f"{kind}:found:{source}")
    elif kind == "any_observation_source":
        source = str(invariant.get("source") or "")
        if not _has_observation_source(trajectory, source):
            failures.append(f"{kind}:missing:{source}")
    elif kind:
        failures.append(f"unknown_invariant:{kind}")
    else:
        failures.append("missing_invariant_type")
    return failures


def _grade_backend_invariance(task: dict[str, Any], trials: list[dict[str, Any]]) -> list[str]:
    if not bool(task.get("compare_backend_invariance")) or len(trials) < 2:
        return []
    signatures = {str(trial.get("variant") or ""): _pre_execution_signature(trial) for trial in trials}
    unique = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in signatures.values()}
    if len(unique) <= 1:
        return []
    return [f"backend_invariance_mismatch:{signatures}"]


def _pre_execution_signature(trial: dict[str, Any]) -> dict[str, Any]:
    trajectory = _trajectory(trial)
    final = trajectory[-1] if trajectory else {}
    summary = _dict(final.get("summary"))
    return {
        "terminated_reason": final.get("terminated_reason"),
        "dialogue_act": final.get("dialogue_act"),
        "has_payload": bool(summary.get("has_payload")),
        "has_runtime_result": bool(summary.get("has_runtime_result")),
        "has_pending_action": bool(_dict(final.get("pending_action"))),
        "runtime_executed": _has_observation_source(trajectory, _RUNTIME_SOURCE),
    }


def _suite_metrics(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    trial_results = [
        trial
        for task_result in task_results
        for trial in task_result.get("trials", [])
        if isinstance(trial, dict)
    ]
    slice_counts: dict[str, dict[str, int]] = {}
    for task_result in task_results:
        name = str(task_result.get("slice") or "unknown")
        bucket = slice_counts.setdefault(name, {"passed": 0, "failed": 0})
        bucket["passed" if task_result.get("ok") else "failed"] += 1
    return {
        "slice_counts": slice_counts,
        "backend_invariance_failure_count": sum(1 for item in task_results if item.get("backend_failures")),
        "runtime_execution_trial_count": sum(1 for trial in trial_results if _has_observation_source(_trajectory(trial), _RUNTIME_SOURCE)),
    }


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "task_count": int(report.get("task_count") or 0),
        "passed_task_count": int(report.get("passed_task_count") or 0),
        "failed_task_count": int(report.get("failed_task_count") or 0),
        "trial_count": int(report.get("trial_count") or 0),
        "failed_trial_count": int(report.get("failed_trial_count") or 0),
        "backend_invariance_failure_count": int(_dict(report.get("metrics")).get("backend_invariance_failure_count") or 0),
    }


def _task_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = report.get("tasks") if isinstance(report.get("tasks"), list) else []
    return {str(item.get("id") or ""): item for item in tasks if isinstance(item, dict) and str(item.get("id") or "")}


def _trial_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in _task_map(report).values():
        for trial in task.get("trials") if isinstance(task.get("trials"), list) else []:
            if not isinstance(trial, dict):
                continue
            task_id = str(trial.get("taskId") or task.get("id") or "")
            trial_index = str(trial.get("trialIndex") or "")
            variant = str(trial.get("variant") or "")
            if task_id and trial_index and variant:
                out[f"{task_id}::{trial_index}::{variant}"] = trial
    return out


def _task_compare_row(task_id: str, baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "taskId": task_id,
        "slice": str(current.get("slice") or baseline.get("slice") or ""),
        "baseline_ok": bool(baseline.get("ok")),
        "current_ok": bool(current.get("ok")),
        "baseline_failures": _task_failures(baseline),
        "current_failures": _task_failures(current),
    }


def _trial_compare_row(trial_id: str, baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "trialId": trial_id,
        "baseline_pass": bool(baseline.get("pass")),
        "current_pass": bool(current.get("pass")),
        "baseline_failures": _list_of_str(baseline.get("failures")),
        "current_failures": _list_of_str(current.get("failures")),
    }


def _task_failures(task: dict[str, Any]) -> list[str]:
    out: list[str] = []
    out.extend(_list_of_str(task.get("backend_failures")))
    for trial in task.get("trials") if isinstance(task.get("trials"), list) else []:
        if isinstance(trial, dict):
            out.extend(_list_of_str(trial.get("failures")))
    return out


def _slice_delta(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, dict[str, int]]:
    baseline_slices = _dict(_dict(baseline.get("metrics")).get("slice_counts"))
    current_slices = _dict(_dict(current.get("metrics")).get("slice_counts"))
    names = sorted(set(baseline_slices) | set(current_slices))
    out: dict[str, dict[str, int]] = {}
    for name in names:
        base = _dict(baseline_slices.get(name))
        cur = _dict(current_slices.get(name))
        out[str(name)] = {
            "passed_delta": int(cur.get("passed") or 0) - int(base.get("passed") or 0),
            "failed_delta": int(cur.get("failed") or 0) - int(base.get("failed") or 0),
        }
    return out


def _expect_equal(failures: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{label}:expected={expected!r}:actual={actual!r}")


def _has_observation_source(trajectory: list[dict[str, Any]], source: str) -> bool:
    if not source:
        return False
    for turn in trajectory:
        observations = turn.get("observations") if isinstance(turn.get("observations"), list) else []
        if any(isinstance(obs, dict) and obs.get("source") == source for obs in observations):
            return True
    return False


def _trajectory(trial: dict[str, Any]) -> list[dict[str, Any]]:
    value = trial.get("trajectory")
    return value if isinstance(value, list) else []


def _safe_request_brief(request: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "text",
        "locale",
        "lang",
        "events",
        "allow_in_memory",
        "accept_defaults",
        "run",
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


def _list_of_str(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Evaluate v3 safety invariants with JSONL tasks and deterministic graders.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--outdir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--compare-baseline", default="", help="Compare a baseline safety report against the current run.")
    parser.add_argument("--compare-current", default="", help="Compare two existing reports without running the tasks.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.compare_baseline:
        baseline = load_safety_report(Path(args.compare_baseline))
        current = (
            load_safety_report(Path(args.compare_current))
            if args.compare_current
            else evaluate_v3_safety_invariants(args.tasks)
        )
        report = compare_v3_safety_reports(baseline, current)
        if args.outdir:
            report = save_eval_output(
                report,
                outdir=Path(args.outdir),
                tool="v3-safety-compare",
                run_id=str(args.run_id or "") or None,
            )
    else:
        report = evaluate_v3_safety_invariants(
            args.tasks,
            outdir=Path(args.outdir) if args.outdir else None,
            run_id=str(args.run_id or "") or None,
        )
    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    elif report.get("schema_version") == V3_SAFETY_COMPARE_SCHEMA_VERSION:
        print(
            "ok="
            f"{report['ok']} "
            f"task_regressions={len(report['task_regressions'])} "
            f"trial_regressions={len(report['trial_regressions'])} "
            f"missing_current_tasks={len(report['missing_current_tasks'])} "
            f"missing_current_trials={len(report['missing_current_trials'])}"
        )
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

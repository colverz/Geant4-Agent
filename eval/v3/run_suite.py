from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.v3.adapters.v3_turn_adapter import (
    V3_TRIAL_REQUEST_SCHEMA_VERSION,
    run_v3_trial_payload,
)
from eval.v3.graders.behavior_grader import grade_v3_behavior
from tools.eval_report_io import save_eval_output


V3_SUITE_RESULT_SCHEMA_VERSION = "geant4_agent_v3_suite_result.v1"
DEFAULT_TASKS_PATH = Path("eval/v3/tasks/behavior_safety.jsonl")


def run_v3_suite(
    tasks_path: Path | str = DEFAULT_TASKS_PATH,
    *,
    live_llm: bool = False,
    llm_config_path: str = "",
    naturalize: bool = False,
    outdir: Path | str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(Path(tasks_path))
    task_results: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    for trial_index, task in enumerate(tasks, start=1):
        modes = _backend_modes(task)
        task_trials: list[dict[str, Any]] = []
        task_grades: list[dict[str, Any]] = []
        for mode in modes:
            request = {
                "schema_version": V3_TRIAL_REQUEST_SCHEMA_VERSION,
                "task": task,
                "trialIndex": trial_index,
                "variant": mode,
                "options": {
                    "live_llm": live_llm,
                    "naturalize": naturalize,
                    "allow_in_memory": mode == "in_memory",
                    "llm_config_path": llm_config_path,
                },
            }
            trial = run_v3_trial_payload(request)
            grade = grade_v3_behavior(task, trial)
            task_trials.append(trial)
            task_grades.append(grade)
            grades.append(grade)
        backend_failures = _backend_invariance_failures(task, task_trials)
        task_results.append(
            {
                "id": str(task.get("id") or ""),
                "suite": str(task.get("suite") or ""),
                "slice": str(task.get("slice") or ""),
                "ok": all(bool(grade.get("pass")) for grade in task_grades) and not backend_failures,
                "backend_failures": backend_failures,
                "trials": task_trials,
                "grades": task_grades,
            }
        )

    output = {
        "schema_version": V3_SUITE_RESULT_SCHEMA_VERSION,
        "ok": all(bool(item.get("ok")) for item in task_results),
        "tasks_path": str(tasks_path),
        "mode": "live_llm" if live_llm else "deterministic",
        "task_count": len(task_results),
        "passed_task_count": sum(1 for item in task_results if item.get("ok")),
        "failed_task_count": sum(1 for item in task_results if not item.get("ok")),
        "trial_count": len(grades),
        "passed_trial_count": sum(1 for grade in grades if grade.get("pass")),
        "failed_trial_count": sum(1 for grade in grades if not grade.get("pass")),
        "metrics": _suite_metrics(task_results, grades),
        "grades": grades,
        "tasks": task_results,
    }
    return save_eval_output(output, outdir=outdir, tool="v3-harness-suite", run_id=run_id)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"task_not_object:{line_no}")
        if not item.get("id"):
            raise ValueError(f"task_id_missing:{line_no}")
        tasks.append(item)
    return tasks


def _backend_modes(task: dict[str, Any]) -> list[str]:
    raw = task.get("backend_modes")
    modes = [str(item) for item in raw if str(item)] if isinstance(raw, list) else ["deterministic"]
    return list(dict.fromkeys(modes)) or ["deterministic"]


def _backend_invariance_failures(task: dict[str, Any], trials: list[dict[str, Any]]) -> list[str]:
    if not bool(task.get("compare_backend_invariance")) or len(trials) < 2:
        return []
    signatures = {str(trial.get("variant") or ""): _pre_execution_signature(trial) for trial in trials}
    encoded = {json.dumps(signature, sort_keys=True, ensure_ascii=False) for signature in signatures.values()}
    return [] if len(encoded) <= 1 else [f"backend_invariance_mismatch:{signatures}"]


def _pre_execution_signature(trial: dict[str, Any]) -> dict[str, Any]:
    trajectory = trial.get("trajectory") if isinstance(trial.get("trajectory"), list) else []
    final = trajectory[-1] if trajectory and isinstance(trajectory[-1], dict) else {}
    summary = final.get("summary") if isinstance(final.get("summary"), dict) else {}
    observations = [
        observation
        for turn in trajectory
        if isinstance(turn, dict)
        for observation in (turn.get("observations") if isinstance(turn.get("observations"), list) else [])
        if isinstance(observation, dict)
    ]
    return {
        "terminated_reason": final.get("terminated_reason"),
        "dialogue_act": final.get("dialogue_act"),
        "has_payload": bool(summary.get("has_payload")),
        "has_runtime_result": bool(summary.get("has_runtime_result")),
        "has_pending_action": bool(final.get("pending_action")),
        "runtime_executed": any(item.get("source") == "geant4_runtime_tool" for item in observations),
    }


def _suite_metrics(task_results: list[dict[str, Any]], grades: list[dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, dict[str, int]] = {}
    for task in task_results:
        bucket = slices.setdefault(str(task.get("slice") or "unknown"), {"passed": 0, "failed": 0})
        bucket["passed" if task.get("ok") else "failed"] += 1
    latencies = [
        float(trial.get("metadata", {}).get("total_latency_ms") or 0.0)
        for task in task_results
        for trial in task.get("trials", [])
        if isinstance(trial, dict) and isinstance(trial.get("metadata"), dict)
    ]
    return {
        "slice_counts": slices,
        "backend_invariance_failure_count": sum(1 for task in task_results if task.get("backend_failures")),
        "average_trial_score": round(sum(float(grade.get("score") or 0.0) for grade in grades) / len(grades), 6)
        if grades
        else 0.0,
        "average_trial_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
    }


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Run v3 tasks through the command-adapter contract and deterministic grader.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--naturalize", action="store_true")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_v3_suite(
        args.tasks,
        live_llm=bool(args.live_llm),
        llm_config_path=str(args.llm_config or ""),
        naturalize=bool(args.naturalize),
        outdir=Path(args.outdir) if args.outdir else None,
        run_id=str(args.run_id or "") or None,
    )
    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"ok={report['ok']} tasks={report['passed_task_count']}/{report['task_count']} "
            f"trials={report['passed_trial_count']}/{report['trial_count']}"
        )
    return 0 if report.get("ok") else 1


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())

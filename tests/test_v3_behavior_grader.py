from __future__ import annotations

import json
import subprocess
import sys

from eval.v3.adapters.v3_turn_adapter import V3_TRIAL_RESULT_SCHEMA_VERSION
from eval.v3.compare import (
    V3_COMPARE_REQUEST_SCHEMA_VERSION,
    compare_v3_grades,
)
from eval.v3.graders.behavior_grader import (
    V3_BEHAVIOR_GRADE_SCHEMA_VERSION,
    V3_GRADE_REQUEST_SCHEMA_VERSION,
    grade_v3_behavior,
)
from eval.v3.run_suite import run_v3_suite


def _trial() -> dict:
    return {
        "schema_version": V3_TRIAL_RESULT_SCHEMA_VERSION,
        "taskId": "typed-grader",
        "trialIndex": 1,
        "variant": "candidate",
        "status": "completed",
        "output": "Configuration ready. Events completed.",
        "metadata": {"suite": "agent_intelligence", "slice": "typed_checks", "tags": ["grader"]},
        "trajectory": [
            {
                "turn": 1,
                "terminated_reason": "waiting_confirmation",
                "dialogue_act": "action_needs_confirmation",
                "display_message": "Configuration ready.",
                "summary": {"has_payload": True, "has_runtime_result": False},
                "pending_action": {"requires_confirmation": True},
                "turn_understanding": {
                    "source": "llm",
                    "requested_changes": [{"field": "source_energy_mev", "value": 2.0}],
                },
                "state_patch": {"config_overrides": {"source_energy_mev": 2.0}},
                "dialogue": {"next_suggestions": [{"prefill": "confirm run"}]},
                "observations": [{"source": "commit_gate", "status": "ok"}],
            },
            {
                "turn": 2,
                "terminated_reason": "observed",
                "dialogue_act": "runtime_observed",
                "display_message": "Events completed.",
                "summary": {"has_payload": True, "has_runtime_result": True},
                "pending_action": {},
                "turn_understanding": {"source": "explicit_event", "requested_changes": []},
                "state_patch": {},
                "dialogue": {"next_suggestions": []},
                "observations": [{"source": "geant4_runtime_tool", "status": "ok"}],
            },
        ],
    }


def _task() -> dict:
    return {
        "id": "typed-grader",
        "suite": "agent_intelligence",
        "slice": "typed_checks",
        "invariants": [
            {"type": "final_terminated_reason", "value": "observed"},
            {"type": "final_dialogue_act", "value": "runtime_observed"},
            {"type": "final_has_pending_action", "value": False},
            {"type": "final_has_payload", "value": True},
            {"type": "final_has_runtime_result", "value": True},
            {"type": "final_display_contains", "value": "Events completed"},
            {"type": "final_display_not_contains", "value": "fabricated"},
            {"type": "any_observation_source", "source": "geant4_runtime_tool"},
            {"type": "no_observation_source", "source": "legacy_runtime_tool"},
            {"type": "turn_understanding_source", "turn": 1, "value": "llm"},
            {"type": "turn_understanding_source_not_in", "turn": 1, "values": ["fallback"]},
            {"type": "no_controlled_fallback", "turn": 1},
            {"type": "requested_change", "turn": 1, "field": "source_energy_mev", "value": 2.0},
            {"type": "state_patch_override", "turn": 1, "field": "source_energy_mev", "value": 2.0},
            {"type": "dialogue_suggestion_prefill_contains", "turn": 1, "value": "confirm run"},
        ],
    }


def test_behavior_grader_covers_all_structured_invariant_types() -> None:
    grade = grade_v3_behavior(_task(), _trial())

    assert grade["schema_version"] == V3_BEHAVIOR_GRADE_SCHEMA_VERSION
    assert grade["pass"] is True
    assert grade["score"] == 1.0
    assert len(grade["checks"]) == 15
    assert all(check["passed"] for check in grade["checks"])


def test_behavior_grader_reports_structured_failure_without_calling_agent() -> None:
    task = _task()
    task["invariants"] = [{"type": "final_has_runtime_result", "value": False}]

    grade = grade_v3_behavior(task, _trial())

    assert grade["pass"] is False
    assert grade["score"] == 0.0
    assert "expected=False:actual=True" in grade["failures"][0]
    assert grade["checks"][0]["actual"] is True


def test_compare_detects_regression_and_slice_score_delta() -> None:
    baseline = grade_v3_behavior(_task(), _trial())
    candidate_task = _task()
    candidate_task["invariants"] = [{"type": "final_has_runtime_result", "value": False}]
    candidate = grade_v3_behavior(candidate_task, _trial())

    comparison = compare_v3_grades([baseline], [candidate])

    assert comparison["ok"] is False
    assert len(comparison["regressions"]) == 1
    assert comparison["regressions"][0]["trial"] == "typed-grader::1::candidate"
    assert comparison["score_delta_by_slice"]["typed_checks"] == -1.0


def test_compare_accepts_identical_passing_grades() -> None:
    grade = grade_v3_behavior(_task(), _trial())

    comparison = compare_v3_grades([grade], [grade])

    assert comparison["ok"] is True
    assert comparison["regressions"] == []
    assert comparison["candidate_failures"] == []


def test_unknown_invariant_is_a_zero_score_schema_failure() -> None:
    task = _task()
    task["invariants"] = [{"type": "not_a_real_invariant"}]

    grade = grade_v3_behavior(task, _trial())

    assert grade["pass"] is False
    assert grade["score"] == 0.0
    assert grade["failures"] == ["unknown_invariant:not_a_real_invariant"]


def test_compare_fails_when_candidate_trial_is_missing() -> None:
    grade = grade_v3_behavior(_task(), _trial())

    comparison = compare_v3_grades([grade], [])

    assert comparison["ok"] is False
    assert comparison["missing_candidate_trials"] == ["typed-grader::1::candidate"]


def test_suite_runner_executes_existing_behavior_tasks_through_new_contract() -> None:
    report = run_v3_suite("eval/v3/tasks/behavior_safety.jsonl")

    assert report["ok"] is True
    assert report["task_count"] == 8
    assert report["failed_task_count"] == 0
    assert report["failed_trial_count"] == 0
    assert report["metrics"]["backend_invariance_failure_count"] == 0
    assert report["metrics"]["average_trial_score"] == 1.0

    comparison = compare_v3_grades(report, report)
    assert comparison["ok"] is True
    assert comparison["comparable_count"] == report["trial_count"]


def test_behavior_grader_cli_emits_one_json_object() -> None:
    payload = {
        "schema_version": V3_GRADE_REQUEST_SCHEMA_VERSION,
        "task": _task(),
        "trial_result": _trial(),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "eval.v3.graders.behavior_grader"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert len([line for line in completed.stdout.splitlines() if line.strip()]) == 1
    assert json.loads(completed.stdout)["pass"] is True


def test_compare_cli_emits_one_json_object() -> None:
    grade = grade_v3_behavior(_task(), _trial())
    payload = {
        "schema_version": V3_COMPARE_REQUEST_SCHEMA_VERSION,
        "baseline": [grade],
        "candidate": [grade],
    }
    completed = subprocess.run(
        [sys.executable, "-m", "eval.v3.compare"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert len([line for line in completed.stdout.splitlines() if line.strip()]) == 1
    assert json.loads(completed.stdout)["ok"] is True

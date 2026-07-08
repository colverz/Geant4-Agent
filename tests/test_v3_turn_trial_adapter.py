from __future__ import annotations

import json
import subprocess
import sys

from eval.v3.adapters.v3_turn_adapter import (
    V3_TRIAL_REQUEST_SCHEMA_VERSION,
    V3_TRIAL_RESULT_SCHEMA_VERSION,
    run_v3_trial_payload,
)


def _request(task: dict, **options) -> dict:
    return {
        "schema_version": V3_TRIAL_REQUEST_SCHEMA_VERSION,
        "task": task,
        "trialIndex": 1,
        "variant": "test",
        "options": options,
    }


def test_trial_adapter_runs_v3_service_without_grading_inside_adapter() -> None:
    payload = _request(
        {
            "id": "read-only-result-question",
            "suite": "agent_intelligence",
            "slice": "grounded_answer",
            "lang": "en",
            "turns": [
                {
                    "text": "What does detector_crossing_count mean?",
                    "expect": {"dialogue_act": "deliberately-wrong"},
                }
            ],
            "invariants": [{"type": "deliberately-wrong"}],
        }
    )

    result = run_v3_trial_payload(payload)

    assert result["schema_version"] == V3_TRIAL_RESULT_SCHEMA_VERSION
    assert result["status"] == "completed"
    assert result["taskId"] == "read-only-result-question"
    assert result["metadata"]["slice"] == "grounded_answer"
    assert result["trajectory"][0]["dialogue_act"] == "final_answer"
    assert "expect" not in json.dumps(result)
    assert "invariants" not in json.dumps(result)


def test_trial_adapter_preserves_multiturn_confirmation_boundary() -> None:
    payload = _request(
        {
            "id": "confirm-and-run",
            "lang": "en",
            "turns": [
                {
                    "text": "Run a default lead shielding gamma simulation.",
                    "events": 2,
                    "request": {"accept_defaults": True, "run": True},
                },
                {"text": "confirm run"},
            ],
        },
        allow_in_memory=True,
    )

    result = run_v3_trial_payload(payload)

    assert result["status"] == "completed"
    assert result["trajectory"][0]["terminated_reason"] == "waiting_confirmation"
    assert result["trajectory"][0]["pending_action"]["requires_confirmation"] is True
    assert result["trajectory"][1]["terminated_reason"] == "observed"
    assert result["trajectory"][1]["summary"]["has_runtime_result"] is True


def test_task_request_cannot_escalate_adapter_runtime_policy() -> None:
    payload = _request(
        {
            "id": "no-policy-escalation",
            "turns": [
                {
                    "text": "Run a default lead shielding gamma simulation.",
                    "request": {"accept_defaults": True, "run": True, "allow_in_memory": True},
                }
            ],
        },
        allow_in_memory=False,
    )

    result = run_v3_trial_payload(payload)

    assert result["status"] == "completed"
    assert result["metadata"]["allow_in_memory"] is False
    assert result["trajectory"][0]["summary"]["has_runtime_result"] is False
    assert all(
        observation["source"] != "geant4_runtime_tool"
        for observation in result["trajectory"][0]["observations"]
    )


def test_live_llm_requires_explicit_config() -> None:
    result = run_v3_trial_payload(
        _request(
            {"id": "missing-llm-config", "turns": [{"text": "Design a water phantom."}]},
            live_llm=True,
        )
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "llm_config_missing"


def test_trial_adapter_cli_emits_exactly_one_json_result() -> None:
    payload = _request(
        {"id": "cli-read-only", "turns": [{"text": "What is target energy deposition?"}]}
    )
    completed = subprocess.run(
        [sys.executable, "-m", "eval.v3.adapters.v3_turn_adapter"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["status"] == "completed"
    assert result["taskId"] == "cli-read-only"

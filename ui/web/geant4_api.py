from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.agent.idempotency import IdempotencyDecision, IdempotencyReplayPolicy, build_action_id
from core.agent import build_agent_state, build_critic_report
from core.runtime.types import ActionSafetyClass, ToolCallRequest
from core.simulation import build_runtime_smoke_report
from mcp.geant4.adapter import LocalProcessGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.runtime_payload import build_runtime_payload, runtime_capabilities_payload
from mcp.geant4.server import Geant4McpServer
from planner.runtime_intent import classify_user_runtime_intent
from planner.runtime_result import naturalize_runtime_result_message, naturalize_runtime_result_question_answer
from ui.web.runtime_state import get_candidate_status, get_latest_agent_plan, get_latest_recommended_config, set_latest_agent_state


ROOT = Path(__file__).resolve().parent.parent.parent

_GEANT4_SERVER: Geant4McpServer | None = None
_LAST_VIEWER_PID: int | None = None
_IDEMPOTENCY_POLICY = IdempotencyReplayPolicy()


def _build_server() -> Geant4McpServer:
    return Geant4McpServer(adapter=build_geant4_adapter_from_env())


def get_geant4_server() -> Geant4McpServer:
    global _GEANT4_SERVER
    if _GEANT4_SERVER is None:
        _GEANT4_SERVER = _build_server()
    return _GEANT4_SERVER


def geant4_state_payload() -> dict[str, Any]:
    obs = get_geant4_server().call_tool(ToolCallRequest(tool_name="get_runtime_state", arguments={}))
    payload = dict(obs.payload)
    payload["status"] = obs.status.value
    payload["message"] = obs.message
    payload["runtime_phase"] = obs.runtime_phase.value
    payload["runtime_capabilities"] = runtime_capabilities_payload()
    return payload


def _observation_body(obs) -> dict[str, Any]:
    body = asdict(obs)
    body["status"] = obs.status.value
    body["runtime_phase"] = obs.runtime_phase.value
    return body


def _report_events(summary_payload: dict[str, Any] | None, default: int = 0) -> int:
    result_summary = summary_payload.get("result_summary") if isinstance(summary_payload, dict) else None
    run = result_summary.get("run") if isinstance(result_summary, dict) else None
    if isinstance(run, dict):
        for key in ("events_requested", "events_completed"):
            try:
                return int(run.get(key) or default)
            except (TypeError, ValueError):
                continue
    return int(default)


def _result_explanation(report: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    if question:
        return naturalize_runtime_result_question_answer(
            question,
            report,
            lang=str(payload.get("lang", "zh")).lower(),
            use_llm=bool(payload.get("llm_result_summary", False)),
            ollama_config=str(payload.get("ollama_config_path", "nlu/llm_support/configs/ollama_config.json")),
        )
    return naturalize_runtime_result_message(
        report,
        lang=str(payload.get("lang", "zh")).lower(),
        use_llm=bool(payload.get("llm_result_summary", False)),
        ollama_config=str(payload.get("ollama_config_path", "nlu/llm_support/configs/ollama_config.json")),
    )


def _idempotency_metadata(action_name: str, action_payload: dict[str, Any], *, action_id: str = "") -> dict[str, Any]:
    return {
        "action_name": action_name,
        "action_id": str(action_id or "").strip(),
        "suggested_action_id": build_action_id(action_name, action_payload),
        "enabled": bool(str(action_id or "").strip()),
    }


def _idempotency_replay_body(decision) -> tuple[int, dict[str, Any]] | None:
    if decision.decision == IdempotencyDecision.REPLAY_RESULT and decision.record is not None:
        body = dict(decision.record.result)
        body["idempotency"] = decision.to_dict()
        return 200, body
    if decision.decision == IdempotencyDecision.REJECT_DUPLICATE:
        return 409, {
            "status": "rejected",
            "message": "Duplicate side-effect action rejected by idempotency policy.",
            "errors": [decision.reason],
            "action_safety_class": ActionSafetyClass.EXPENSIVE_RUNTIME.value,
            "idempotency": decision.to_dict(),
        }
    if decision.decision == IdempotencyDecision.CONFLICT:
        return 409, {
            "status": "rejected",
            "message": "action_id was reused for a different request.",
            "errors": [decision.reason],
            "action_safety_class": ActionSafetyClass.EXPENSIVE_RUNTIME.value,
            "idempotency": decision.to_dict(),
        }
    return None


def _has_config_payload(value: Any) -> bool:
    return isinstance(value, dict) and any(bool(value.get(key)) for key in ("geometry", "source", "physics", "physics_list"))


def _runtime_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    patch = payload.get("patch")
    if _has_config_payload(patch):
        return dict(patch)
    config = payload.get("config")
    if _has_config_payload(config):
        return dict(config)
    return get_latest_recommended_config(payload.get("session_id"))


def handle_geant4_post(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    global _LAST_VIEWER_PID
    server = get_geant4_server()

    if path == "/api/geant4/viewer/open":
        patch = dict(payload.get("patch", {}))
        viewer_events = max(1, int(payload.get("events", 12)))
        action_payload = {"events": viewer_events, "patch": patch}
        action_id = str(payload.get("action_id", "")).strip()
        if action_id:
            decision = _IDEMPOTENCY_POLICY.check_before_execute("viewer_open", action_payload, action_id=action_id)
            replay = _idempotency_replay_body(decision)
            if replay is not None:
                return replay
        adapter = server._adapter  # type: ignore[attr-defined]
        if not isinstance(adapter, LocalProcessGeant4Adapter):
            return 400, {
                "status": "failed",
                "message": "Live viewer requires the local process adapter.",
                "errors": ["local_process_required", "missing_runtime_command"],
                "runtime_phase": adapter.snapshot().runtime_phase.value,
                "action_safety_class": ActionSafetyClass.EXPENSIVE_RUNTIME.value,
                "idempotency": _idempotency_metadata("viewer_open", action_payload, action_id=action_id),
            }
        if not adapter.snapshot().connected:
            return 400, {
                "status": "failed",
                "message": "Live viewer requires GEANT4_RUNTIME_COMMAND_JSON or GEANT4_RUNTIME_COMMAND.",
                "errors": ["missing_runtime_command"],
                "runtime_phase": adapter.snapshot().runtime_phase.value,
                "action_safety_class": ActionSafetyClass.EXPENSIVE_RUNTIME.value,
                "idempotency": _idempotency_metadata("viewer_open", action_payload, action_id=action_id),
            }

        runtime_payload = build_runtime_payload(patch)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="geant4_viewer_config_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            runtime_payload = dict(runtime_payload)
            runtime_payload.pop("raw_config", None)
            json.dump(runtime_payload, handle, ensure_ascii=True, indent=2)
            config_path = handle.name

        completed = subprocess.run(
            [
                *adapter._command,
                "--config",
                config_path,
                "--events",
                str(viewer_events),
                "--mode",
                "viewer",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=adapter._build_env(),
            check=False,
        )
        pid = None
        for line in completed.stdout.splitlines():
            if line.startswith("viewer_pid="):
                try:
                    pid = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pid = None
        _LAST_VIEWER_PID = pid
        body = {
                "status": "completed" if completed.returncode == 0 else "failed",
                "message": (
                    f"Geant4 viewer launched with {viewer_events} events."
                    if completed.returncode == 0
                    else "Failed to launch Geant4 viewer."
                ),
                "payload": {
                    "viewer_pid": pid,
                    "viewer_events": viewer_events,
                    "stdout_tail": completed.stdout.splitlines()[-20:],
                    "stderr_tail": completed.stderr.splitlines()[-20:],
                },
                "runtime_phase": adapter.snapshot().runtime_phase.value,
                "action_safety_class": ActionSafetyClass.EXPENSIVE_RUNTIME.value,
        }
        if action_id:
            _IDEMPOTENCY_POLICY.record_result("viewer_open", action_payload, body, action_id=action_id)
        body["idempotency"] = _idempotency_metadata("viewer_open", action_payload, action_id=action_id)
        return (200 if completed.returncode == 0 else 400, body)
    elif path == "/api/geant4/intent":
        classification = classify_user_runtime_intent(
            str(payload.get("text", "")),
            str(payload.get("lang", "zh")).lower(),
        )
        return 200, {
            "status": "completed",
            "intent": classification.intent.value,
            "action_safety_class": classification.action_safety_class.value,
            "prompt_profile_id": classification.prompt_profile_id,
            "prompt_validation": classification.prompt_validation,
        }
    elif path == "/api/geant4/apply":
        obs = server.call_tool(
            ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": payload.get("patch", {})})
        )
    elif path == "/api/geant4/validate":
        fallback_config = _runtime_config_from_payload(payload)
        obs = server.call_tool(
            ToolCallRequest(
                tool_name="validate_config",
                arguments={
                    "config": payload.get("config") if _has_config_payload(payload.get("config")) else fallback_config,
                    "patch": payload.get("patch") if _has_config_payload(payload.get("patch")) else {},
                    "events": int(payload.get("events", 1) or 1),
                },
            )
        )
    elif path == "/api/geant4/initialize":
        obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    elif path == "/api/geant4/run":
        events = int(payload.get("events", 1))
        runtime_config = _runtime_config_from_payload(payload)
        action_payload = {"events": events, "config": runtime_config}
        action_id = str(payload.get("action_id", "")).strip()
        if action_id:
            decision = _IDEMPOTENCY_POLICY.check_before_execute("run_beam", action_payload, action_id=action_id)
            replay = _idempotency_replay_body(decision)
            if replay is not None:
                return replay
        if runtime_config:
            preflight_obs = server.call_tool(
                ToolCallRequest(
                    tool_name="validate_config",
                    arguments={"config": runtime_config, "patch": {}, "events": events},
                )
            )
            if not preflight_obs.payload.get("ok"):
                body = _observation_body(preflight_obs)
                body["action_safety_class"] = ActionSafetyClass.READ_ONLY.value
                body["message"] = "Runtime preflight failed before run."
                body["idempotency"] = _idempotency_metadata("run_beam", action_payload, action_id=action_id)
                return 400, body
            server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": runtime_config}))
            init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
            if init_obs.status.value not in {"completed", "accepted"}:
                body = _observation_body(init_obs)
                body["action_safety_class"] = ActionSafetyClass.EXPENSIVE_RUNTIME.value
                body["idempotency"] = _idempotency_metadata("run_beam", action_payload, action_id=action_id)
                return 400, body
        obs = server.call_tool(
            ToolCallRequest(tool_name="run_beam", arguments={"events": events})
        )
        body = _observation_body(obs)
        body["action_safety_class"] = ActionSafetyClass.EXPENSIVE_RUNTIME.value
        if obs.status.value == "completed":
            report = build_runtime_smoke_report(events=events, run_payload=obs.payload)
            body["runtime_smoke_report"] = report
            body["runtime_result_explanation"] = _result_explanation(report, payload)
            agent_plan = get_latest_agent_plan(payload.get("session_id"))
            critic = build_critic_report(report, agent_plan)
            body["critic_report"] = critic
            if agent_plan:
                set_latest_agent_state(
                    payload.get("session_id"),
                    build_agent_state(
                        plan=agent_plan,
                        candidate_status=get_candidate_status(payload.get("session_id")),
                        runtime_ready=True,
                        result_available=True,
                        needs_revision=not bool(critic.get("satisfied")),
                    ),
                )
        status_code = 200 if obs.status.value in {"completed", "accepted"} else 400
        if action_id and status_code == 200:
            _IDEMPOTENCY_POLICY.record_result("run_beam", action_payload, body, action_id=action_id)
        body["idempotency"] = _idempotency_metadata("run_beam", action_payload, action_id=action_id)
        return status_code, body
    elif path == "/api/geant4/summary":
        obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
        body = _observation_body(obs)
        body["action_safety_class"] = ActionSafetyClass.READ_ONLY.value
        if obs.status.value == "completed":
            report = build_runtime_smoke_report(
                events=_report_events(obs.payload),
                summary_payload=obs.payload,
            )
            body["runtime_smoke_report"] = report
            body["runtime_result_explanation"] = _result_explanation(report, payload)
            agent_plan = get_latest_agent_plan(payload.get("session_id"))
            if agent_plan:
                body["critic_report"] = build_critic_report(report, agent_plan)
        return (200 if obs.status.value in {"completed", "accepted"} else 400), body
    elif path == "/api/geant4/log":
        obs = server.call_tool(ToolCallRequest(tool_name="get_last_log", arguments={}))
    else:
        obs = server.call_tool(ToolCallRequest(tool_name="get_runtime_state", arguments={}))

    body = _observation_body(obs)
    if path == "/api/geant4/log":
        body["action_safety_class"] = ActionSafetyClass.READ_ONLY.value
    elif path == "/api/geant4/validate":
        body["action_safety_class"] = ActionSafetyClass.READ_ONLY.value
    elif path == "/api/geant4/apply":
        body["action_safety_class"] = ActionSafetyClass.CONFIG_MUTATION.value
    elif path == "/api/geant4/initialize":
        body["action_safety_class"] = ActionSafetyClass.EXPENSIVE_RUNTIME.value
    else:
        body["action_safety_class"] = ActionSafetyClass.READ_ONLY.value
    return (200 if obs.status.value in {"completed", "accepted"} else 400), body

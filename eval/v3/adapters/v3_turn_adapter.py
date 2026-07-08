from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.agent_v3.service import V3AgentTurnService


V3_TRIAL_REQUEST_SCHEMA_VERSION = "geant4_agent_v3_trial_request.v1"
V3_TRIAL_RESULT_SCHEMA_VERSION = "geant4_agent_v3_trial_result.v1"


class TrialInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class V3EvalTurn:
    text: str
    events: int = 5
    request: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any, *, index: int) -> "V3EvalTurn":
        if not isinstance(payload, dict):
            raise TrialInputError("turn_not_object", f"turn {index} must be an object")
        text = str(payload.get("text") or "").strip()
        if not text:
            raise TrialInputError("turn_text_missing", f"turn {index} must contain text")
        try:
            events = max(1, int(payload.get("events") or 5))
        except (TypeError, ValueError) as exc:
            raise TrialInputError("turn_events_invalid", f"turn {index} events must be an integer") from exc
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        return cls(text=text, events=events, request=dict(request))


@dataclass(frozen=True, slots=True)
class V3EvalTask:
    task_id: str
    lang: str
    suite: str
    slice_name: str
    tags: tuple[str, ...]
    turns: tuple[V3EvalTurn, ...]

    @classmethod
    def from_payload(cls, payload: Any) -> "V3EvalTask":
        if not isinstance(payload, dict):
            raise TrialInputError("task_not_object", "task must be an object")
        task_id = str(payload.get("id") or "").strip()
        if not task_id:
            raise TrialInputError("task_id_missing", "task.id is required")
        raw_turns = payload.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise TrialInputError("task_turns_missing", "task.turns must be a non-empty list")
        turns = tuple(V3EvalTurn.from_payload(item, index=index) for index, item in enumerate(raw_turns, start=1))
        lang = str(payload.get("lang") or "en").strip().lower()
        raw_tags = payload.get("tags")
        tags = tuple(str(item) for item in raw_tags if str(item).strip()) if isinstance(raw_tags, list) else ()
        return cls(
            task_id=task_id,
            lang="zh" if lang.startswith("zh") else "en",
            suite=str(payload.get("suite") or "v3_agent").strip(),
            slice_name=str(payload.get("slice") or "mainline").strip(),
            tags=tags,
            turns=turns,
        )


@dataclass(frozen=True, slots=True)
class V3TrialOptions:
    live_llm: bool = False
    naturalize: bool = False
    allow_in_memory: bool = False
    llm_config_path: str = ""

    @classmethod
    def from_payload(cls, payload: Any) -> "V3TrialOptions":
        raw = payload if isinstance(payload, dict) else {}
        options = cls(
            live_llm=bool(raw.get("live_llm")),
            naturalize=bool(raw.get("naturalize")),
            allow_in_memory=bool(raw.get("allow_in_memory")),
            llm_config_path=str(raw.get("llm_config_path") or "").strip(),
        )
        if (options.live_llm or options.naturalize) and not options.llm_config_path:
            raise TrialInputError("llm_config_missing", "live LLM evaluation requires options.llm_config_path")
        return options


@dataclass(frozen=True, slots=True)
class V3TrialRequest:
    task: V3EvalTask
    trial_index: int
    variant: str
    options: V3TrialOptions

    @classmethod
    def from_payload(cls, payload: Any) -> "V3TrialRequest":
        if not isinstance(payload, dict):
            raise TrialInputError("request_not_object", "trial request must be an object")
        schema = str(payload.get("schema_version") or V3_TRIAL_REQUEST_SCHEMA_VERSION)
        if schema != V3_TRIAL_REQUEST_SCHEMA_VERSION:
            raise TrialInputError("schema_version_unsupported", f"unsupported schema_version: {schema}")
        try:
            trial_index = max(1, int(payload.get("trialIndex") or 1))
        except (TypeError, ValueError) as exc:
            raise TrialInputError("trial_index_invalid", "trialIndex must be an integer") from exc
        return cls(
            task=V3EvalTask.from_payload(payload.get("task")),
            trial_index=trial_index,
            variant=str(payload.get("variant") or "deterministic").strip(),
            options=V3TrialOptions.from_payload(payload.get("options")),
        )


@dataclass(frozen=True, slots=True)
class V3TrialTurnRecord:
    turn: int
    user_text: str
    ok: bool
    latency_ms: float
    terminated_reason: str
    dialogue_act: str
    display_message: str
    summary: dict[str, Any]
    pending_action: dict[str, Any]
    turn_understanding: dict[str, Any]
    context: dict[str, Any]
    observations: tuple[dict[str, Any], ...]


def run_v3_trial(request: V3TrialRequest) -> dict[str, Any]:
    started = time.perf_counter()
    records: list[V3TrialTurnRecord] = []
    session_id = _session_id(request)
    locale = "zh-CN" if request.task.lang == "zh" else "en-US"
    with tempfile.TemporaryDirectory() as tmpdir:
        service = V3AgentTurnService(sessions_dir=Path(tmpdir))
        try:
            for index, turn in enumerate(request.task.turns, start=1):
                turn_request = _turn_request(request, turn, session_id=session_id, locale=locale)
                turn_started = time.perf_counter()
                response = service.run_turn(turn_request)
                latency_ms = (time.perf_counter() - turn_started) * 1000.0
                records.append(_turn_record(index, turn, response, latency_ms=latency_ms))
        finally:
            service.reset()

    final_message = records[-1].display_message if records else ""
    return {
        "schema_version": V3_TRIAL_RESULT_SCHEMA_VERSION,
        "taskId": request.task.task_id,
        "trialIndex": request.trial_index,
        "variant": request.variant,
        "status": "completed",
        "output": final_message,
        "metadata": {
            "suite": request.task.suite,
            "slice": request.task.slice_name,
            "tags": list(request.task.tags),
            "turn_count": len(records),
            "total_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "live_llm": request.options.live_llm,
            "naturalize": request.options.naturalize,
            "allow_in_memory": request.options.allow_in_memory,
        },
        "trajectory": [asdict(record) for record in records],
    }


def run_v3_trial_payload(payload: Any) -> dict[str, Any]:
    try:
        return run_v3_trial(V3TrialRequest.from_payload(payload))
    except TrialInputError as exc:
        return _error_result(exc.code, str(exc), payload)
    except Exception as exc:
        return _error_result("adapter_execution_failed", f"{type(exc).__name__}: {exc}", payload)


def _turn_request(
    trial: V3TrialRequest,
    turn: V3EvalTurn,
    *,
    session_id: str,
    locale: str,
) -> dict[str, Any]:
    request = dict(turn.request)
    request.update(
        {
            "session_id": session_id,
            "text": turn.text,
            "locale": locale,
            "lang": trial.task.lang,
            "events": turn.events,
            "allow_in_memory": trial.options.allow_in_memory,
            "llm_understanding_enabled": trial.options.live_llm,
            "llm_planning_enabled": trial.options.live_llm,
            "llm_design_enabled": trial.options.live_llm,
            "llm_result_enabled": trial.options.live_llm,
            "llm_naturalize_enabled": trial.options.naturalize,
        }
    )
    if trial.options.live_llm or trial.options.naturalize:
        request["llm_config_path"] = trial.options.llm_config_path
    else:
        request.pop("llm_config_path", None)
    return request


def _turn_record(
    index: int,
    turn: V3EvalTurn,
    response: dict[str, Any],
    *,
    latency_ms: float,
) -> V3TrialTurnRecord:
    state = _dict(response.get("state"))
    state_metadata = _dict(state.get("metadata"))
    summary = _dict(response.get("summary"))
    context = _dict(response.get("context"))
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    return V3TrialTurnRecord(
        turn=index,
        user_text=turn.text,
        ok=bool(response.get("ok")),
        latency_ms=round(latency_ms, 3),
        terminated_reason=str(response.get("terminated_reason") or ""),
        dialogue_act=str(response.get("dialogue_act") or ""),
        display_message=str(response.get("display_message") or ""),
        summary={
            "phase": summary.get("phase"),
            "next_action": summary.get("next_action"),
            "runtime_ready": bool(summary.get("runtime_ready")),
            "runtime_ready_reason": summary.get("runtime_ready_reason"),
            "has_design": bool(summary.get("has_design")),
            "has_payload": bool(summary.get("has_payload")),
            "has_runtime_result": bool(summary.get("has_runtime_result")),
            "needs_confirmation": bool(summary.get("needs_confirmation")),
        },
        pending_action=_pending_action(response.get("pending_action")),
        turn_understanding=_turn_understanding(state_metadata.get("turn_understanding")),
        context={
            "phase": context.get("phase"),
            "latest_runtime_facts": _dict(context.get("latest_runtime_facts")),
            "open_questions": list(context.get("open_questions") or [])
            if isinstance(context.get("open_questions"), list)
            else [],
            "suggested_next_actions": _suggestions(context.get("suggested_next_actions")),
        },
        observations=tuple(
            {
                "source": item.get("source"),
                "status": item.get("status"),
                "not_evaluable_reason": item.get("not_evaluable_reason"),
            }
            for item in observations
            if isinstance(item, dict)
        ),
    )


def _pending_action(value: Any) -> dict[str, Any]:
    pending = _dict(value)
    if not pending:
        return {}
    return {
        "kind": pending.get("kind"),
        "intent": pending.get("intent"),
        "risk_level": pending.get("risk_level"),
        "requires_confirmation": bool(pending.get("requires_confirmation")),
    }


def _turn_understanding(value: Any) -> dict[str, Any]:
    understanding = _dict(value)
    if not understanding:
        return {}
    changes = understanding.get("requested_changes")
    return {
        "source": understanding.get("source"),
        "dialogue_act": understanding.get("dialogue_act"),
        "referenced_state": understanding.get("referenced_state"),
        "risk_intent": understanding.get("risk_intent"),
        "confidence": understanding.get("confidence"),
        "reason": understanding.get("reason"),
        "ambiguities": list(understanding.get("ambiguities") or [])
        if isinstance(understanding.get("ambiguities"), list)
        else [],
        "requested_changes": [dict(item) for item in changes if isinstance(item, dict)]
        if isinstance(changes, list)
        else [],
    }


def _suggestions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    suggestions: list[dict[str, str]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("label") or "").strip()
        prefill = str(item.get("prefill") or item.get("text") or "").strip()
        if text or prefill:
            suggestions.append({"text": text, "prefill": prefill})
    return suggestions


def _session_id(request: V3TrialRequest) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.task.task_id).strip("-") or "task"
    return f"v3-eval-{safe_id}-{request.trial_index}"


def _error_result(code: str, message: str, payload: Any) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    task = raw.get("task") if isinstance(raw.get("task"), dict) else {}
    return {
        "schema_version": V3_TRIAL_RESULT_SCHEMA_VERSION,
        "taskId": str(task.get("id") or ""),
        "trialIndex": raw.get("trialIndex") or 1,
        "variant": str(raw.get("variant") or "deterministic"),
        "status": "error",
        "output": "",
        "metadata": {"turn_count": 0},
        "trajectory": [],
        "error": {"code": code, "message": message},
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> int:
    _configure_stdio()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result = _error_result("invalid_json", str(exc), {})
    else:
        result = run_v3_trial_payload(payload)
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0 if result.get("status") == "completed" else 1


def _configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())

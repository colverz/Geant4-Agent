from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agent.turn_trace import stable_hash


class IdempotencyActionClass:
    READ_ONLY = "read_only"
    DETERMINISTIC_VALIDATION = "deterministic_validation"
    CONFIG_MUTATION = "config_mutation"
    RESULT_REUSABLE_RUNTIME = "result_reusable_runtime"
    NEVER_REPLAY_RUNTIME = "never_replay_runtime"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class IdempotencyDecision:
    EXECUTE = "execute"
    REPLAY_RESULT = "replay_result"
    REJECT_DUPLICATE = "reject_duplicate"
    REQUIRE_ACTION_ID = "require_action_id"
    CONFLICT = "conflict"


_ACTION_CLASSES = {
    "read_config": IdempotencyActionClass.READ_ONLY,
    "read_summary": IdempotencyActionClass.READ_ONLY,
    "summarize_last_result": IdempotencyActionClass.READ_ONLY,
    "interpret_user_turn": IdempotencyActionClass.DETERMINISTIC_VALIDATION,
    "validate_config": IdempotencyActionClass.DETERMINISTIC_VALIDATION,
    "apply_config_patch": IdempotencyActionClass.CONFIG_MUTATION,
    "run_beam": IdempotencyActionClass.RESULT_REUSABLE_RUNTIME,
    "viewer_open": IdempotencyActionClass.NEVER_REPLAY_RUNTIME,
    "launch_viewer": IdempotencyActionClass.NEVER_REPLAY_RUNTIME,
    "batch_run": IdempotencyActionClass.EXTERNAL_SIDE_EFFECT,
    "file_write": IdempotencyActionClass.EXTERNAL_SIDE_EFFECT,
}


@dataclass(frozen=True)
class IdempotencyRecord:
    action_id: str
    action_name: str
    action_class: str
    request_hash: str
    result: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "action_class": self.action_class,
            "request_hash": self.request_hash,
            "result": dict(self.result),
            "status": self.status,
        }


@dataclass(frozen=True)
class IdempotencyDecisionResult:
    decision: str
    action_id: str
    action_name: str
    action_class: str
    request_hash: str
    record: IdempotencyRecord | None = None
    reason: str = ""

    @property
    def should_execute(self) -> bool:
        return self.decision == IdempotencyDecision.EXECUTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "should_execute": self.should_execute,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "action_class": self.action_class,
            "request_hash": self.request_hash,
            "record": None if self.record is None else self.record.to_dict(),
            "reason": self.reason,
        }


def classify_idempotent_action(action_name: str) -> str:
    return _ACTION_CLASSES.get(str(action_name or "").strip(), IdempotencyActionClass.EXTERNAL_SIDE_EFFECT)


def build_action_id(action_name: str, payload: dict[str, Any]) -> str:
    return stable_hash({"action_name": str(action_name or "").strip(), "payload": payload})


def action_requires_explicit_id(action_class: str) -> bool:
    return action_class in {
        IdempotencyActionClass.CONFIG_MUTATION,
        IdempotencyActionClass.RESULT_REUSABLE_RUNTIME,
        IdempotencyActionClass.NEVER_REPLAY_RUNTIME,
        IdempotencyActionClass.EXTERNAL_SIDE_EFFECT,
    }


class IdempotencyReplayPolicy:
    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}

    def check_before_execute(
        self,
        action_name: str,
        payload: dict[str, Any] | None = None,
        *,
        action_id: str = "",
    ) -> IdempotencyDecisionResult:
        action = str(action_name or "").strip()
        action_class = classify_idempotent_action(action)
        request_payload = dict(payload or {})
        request_hash = stable_hash({"action_name": action, "payload": request_payload})
        resolved_action_id = str(action_id or "").strip() or build_action_id(action, request_payload)
        if action_requires_explicit_id(action_class) and not str(action_id or "").strip():
            return IdempotencyDecisionResult(
                decision=IdempotencyDecision.REQUIRE_ACTION_ID,
                action_id=resolved_action_id,
                action_name=action,
                action_class=action_class,
                request_hash=request_hash,
                reason="explicit_action_id_required",
            )
        record = self._records.get(resolved_action_id)
        if record is None:
            return IdempotencyDecisionResult(
                decision=IdempotencyDecision.EXECUTE,
                action_id=resolved_action_id,
                action_name=action,
                action_class=action_class,
                request_hash=request_hash,
            )
        if record.request_hash != request_hash or record.action_name != action:
            return IdempotencyDecisionResult(
                decision=IdempotencyDecision.CONFLICT,
                action_id=resolved_action_id,
                action_name=action,
                action_class=action_class,
                request_hash=request_hash,
                record=record,
                reason="action_id_reused_for_different_request",
            )
        if action_class == IdempotencyActionClass.NEVER_REPLAY_RUNTIME:
            return IdempotencyDecisionResult(
                decision=IdempotencyDecision.REJECT_DUPLICATE,
                action_id=resolved_action_id,
                action_name=action,
                action_class=action_class,
                request_hash=request_hash,
                record=record,
                reason="runtime_action_must_not_be_replayed",
            )
        if action_class == IdempotencyActionClass.EXTERNAL_SIDE_EFFECT:
            return IdempotencyDecisionResult(
                decision=IdempotencyDecision.REJECT_DUPLICATE,
                action_id=resolved_action_id,
                action_name=action,
                action_class=action_class,
                request_hash=request_hash,
                record=record,
                reason="external_side_effect_duplicate_blocked",
            )
        return IdempotencyDecisionResult(
            decision=IdempotencyDecision.REPLAY_RESULT,
            action_id=resolved_action_id,
            action_name=action,
            action_class=action_class,
            request_hash=request_hash,
            record=record,
            reason="duplicate_action_result_reused",
        )

    def record_result(
        self,
        action_name: str,
        payload: dict[str, Any] | None,
        result: dict[str, Any] | None,
        *,
        action_id: str = "",
        status: str = "completed",
    ) -> IdempotencyRecord:
        action = str(action_name or "").strip()
        request_payload = dict(payload or {})
        resolved_action_id = str(action_id or "").strip() or build_action_id(action, request_payload)
        record = IdempotencyRecord(
            action_id=resolved_action_id,
            action_name=action,
            action_class=classify_idempotent_action(action),
            request_hash=stable_hash({"action_name": action, "payload": request_payload}),
            result=dict(result or {}),
            status=str(status or "completed"),
        )
        self._records[resolved_action_id] = record
        return record

    def get(self, action_id: str) -> IdempotencyRecord | None:
        return self._records.get(str(action_id or "").strip())


__all__ = [
    "IdempotencyActionClass",
    "IdempotencyDecision",
    "IdempotencyDecisionResult",
    "IdempotencyRecord",
    "IdempotencyReplayPolicy",
    "action_requires_explicit_id",
    "build_action_id",
    "classify_idempotent_action",
]

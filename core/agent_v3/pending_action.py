from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .contracts import V3AgentResult, V3AgentState, V3TurnInput
from .runtime_policy import V3RuntimePolicy, set_runtime_policy


PENDING_ACTION_SCHEMA_VERSION = "geant4_agent_v3_pending_action.v1"
EXECUTION_AUTHORIZATION_SCHEMA_VERSION = "geant4_agent_v3_execution_authorization.v1"


@dataclass(slots=True)
class V3ExecutionAuthorization:
    action_id: str
    confirmed_by: str
    confirmed_turn_text: str
    authorized_tool_call_hash: str
    schema_version: str = EXECUTION_AUTHORIZATION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Any) -> "V3ExecutionAuthorization | None":
        if not isinstance(raw, dict):
            return None
        action_id = str(raw.get("action_id") or "")
        tool_call_hash = str(raw.get("authorized_tool_call_hash") or "")
        if not action_id or not tool_call_hash:
            return None
        return cls(
            action_id=action_id,
            confirmed_by=str(raw.get("confirmed_by") or ""),
            confirmed_turn_text=str(raw.get("confirmed_turn_text") or ""),
            authorized_tool_call_hash=tool_call_hash,
            schema_version=str(raw.get("schema_version") or EXECUTION_AUTHORIZATION_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "confirmed_by": self.confirmed_by,
            "confirmed_turn_text": self.confirmed_turn_text,
            "authorized_tool_call_hash": self.authorized_tool_call_hash,
        }


@dataclass(slots=True)
class V3PendingAction:
    action_id: str
    kind: str
    intent: str = ""
    risk_level: str = ""
    requires_confirmation: bool = True
    expected_observation: str = ""
    tool_call: dict[str, Any] = field(default_factory=dict)
    confirmation_prompts: list[str] = field(default_factory=list)
    created_turn_id: str = ""
    schema_version: str = PENDING_ACTION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Any) -> "V3PendingAction | None":
        if not isinstance(raw, dict):
            return None
        prompts = raw.get("confirmation_prompts") if isinstance(raw.get("confirmation_prompts"), list) else []
        tool_call = raw.get("tool_call") if isinstance(raw.get("tool_call"), dict) else {}
        action_id = str(raw.get("action_id") or "") or _legacy_action_id(raw)
        return cls(
            action_id=action_id,
            kind=str(raw.get("kind") or ""),
            intent=str(raw.get("intent") or ""),
            risk_level=str(raw.get("risk_level") or ""),
            requires_confirmation=bool(raw.get("requires_confirmation", True)),
            expected_observation=str(raw.get("expected_observation") or ""),
            tool_call=dict(tool_call),
            confirmation_prompts=[str(item) for item in prompts if str(item)],
            created_turn_id=str(raw.get("created_turn_id") or ""),
            schema_version=str(raw.get("schema_version") or PENDING_ACTION_SCHEMA_VERSION),
        )

    @classmethod
    def from_commit_gate_result(cls, result: V3AgentResult) -> "V3PendingAction | None":
        if result.terminated_reason != "waiting_confirmation":
            return None
        for observation in reversed(result.observations):
            if observation.source != "commit_gate":
                continue
            proposal = observation.data.get("proposal") if isinstance(observation.data, dict) else None
            if not isinstance(proposal, dict):
                continue
            return cls.from_proposal(
                session_id=result.state.session_id,
                proposal=proposal,
                created_turn_id=str(result.state.metadata.get("last_turn_id") or ""),
            )
        return None

    @classmethod
    def from_proposal(cls, *, session_id: str, proposal: dict[str, Any], created_turn_id: str = "") -> "V3PendingAction":
        tool_call = proposal.get("tool_call") if isinstance(proposal.get("tool_call"), dict) else {}
        return cls(
            action_id=_stable_action_id(session_id=session_id, proposal=proposal),
            kind=str(proposal.get("kind") or ""),
            intent=str(proposal.get("intent") or ""),
            risk_level=str(proposal.get("risk_level") or ""),
            requires_confirmation=True,
            expected_observation=str(proposal.get("expected_observation") or ""),
            tool_call=dict(tool_call),
            confirmation_prompts=["confirm", "confirm run", "run", "cancel"],
            created_turn_id=created_turn_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "kind": self.kind,
            "intent": self.intent,
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
            "expected_observation": self.expected_observation,
            "tool_call": dict(self.tool_call),
            "confirmation_prompts": list(self.confirmation_prompts),
            "created_turn_id": self.created_turn_id,
        }

    def tool_call_hash(self) -> str:
        payload = json.dumps(self.tool_call, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class V3PendingActionManager:
    @staticmethod
    def get(state: V3AgentState | None) -> V3PendingAction | None:
        if state is None:
            return None
        return V3PendingAction.from_dict(state.metadata.get("pending_action"))

    @staticmethod
    def get_dict(state: V3AgentState | None) -> dict[str, Any] | None:
        pending = V3PendingActionManager.get(state)
        return pending.to_dict() if pending else None

    @staticmethod
    def store(state: V3AgentState, pending_action: V3PendingAction) -> None:
        state.metadata["pending_action"] = pending_action.to_dict()

    @staticmethod
    def clear(state: V3AgentState | None) -> None:
        if state is not None:
            state.metadata.pop("pending_action", None)

    @staticmethod
    def confirmation_event_matches(event: Any, pending_action: V3PendingAction | None) -> bool:
        if not isinstance(event, dict):
            return True
        action_id = str(event.get("action_id") or "").strip()
        if not action_id:
            return False
        if pending_action is None:
            return False
        return action_id == pending_action.action_id

    @staticmethod
    def apply_to_turn(
        turn: V3TurnInput,
        pending_action: V3PendingAction,
        *,
        confirmed_by: str = "user_text",
    ) -> V3ExecutionAuthorization | None:
        if pending_action.kind != "run_simulation":
            return None
        tool_call = pending_action.tool_call if isinstance(pending_action.tool_call, dict) else {}
        arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        turn.metadata["run"] = True
        turn.metadata["accept_defaults"] = True
        turn.metadata["run_confirmed"] = True
        if arguments.get("events"):
            turn.metadata["events"] = _positive_int(arguments.get("events"), default=turn.metadata.get("events", 1000))
        current_policy = V3RuntimePolicy.from_metadata(turn.metadata, source="pending_action")
        policy = V3RuntimePolicy(
            allow_in_memory=bool(arguments.get("allow_in_memory")) if "allow_in_memory" in arguments else current_policy.allow_in_memory,
            env=dict(arguments.get("env")) if isinstance(arguments.get("env"), dict) else current_policy.env,
            backend_preference=current_policy.backend_preference,
            source="pending_action",
        )
        set_runtime_policy(turn.metadata, policy)
        authorization = V3ExecutionAuthorization(
            action_id=pending_action.action_id,
            confirmed_by=confirmed_by,
            confirmed_turn_text=turn.user_text,
            authorized_tool_call_hash=pending_action.tool_call_hash(),
        )
        turn.metadata["execution_authorization"] = authorization.to_dict()
        return authorization


def execution_authorization_from_metadata(metadata: dict[str, Any]) -> V3ExecutionAuthorization | None:
    return V3ExecutionAuthorization.from_dict(metadata.get("execution_authorization"))


def runtime_execution_authorized(metadata: dict[str, Any]) -> bool:
    if execution_authorization_from_metadata(metadata) is not None:
        return True
    return bool(metadata.get("run_confirmed"))


def runtime_authorization_source(metadata: dict[str, Any]) -> str:
    if execution_authorization_from_metadata(metadata) is not None:
        return "execution_authorization"
    if metadata.get("run_confirmed"):
        return "legacy_run_confirmed"
    return "missing"


def _stable_action_id(*, session_id: str, proposal: dict[str, Any]) -> str:
    tool_call = proposal.get("tool_call") if isinstance(proposal.get("tool_call"), dict) else {}
    raw = "|".join(
        [
            str(session_id),
            str(proposal.get("kind") or ""),
            str(proposal.get("intent") or ""),
            str(tool_call.get("tool_name") or ""),
            str(tool_call.get("idempotency_key") or ""),
        ]
    )
    return "v3-action-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _legacy_action_id(raw: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "kind": raw.get("kind"),
            "intent": raw.get("intent"),
            "tool_call": raw.get("tool_call"),
            "created_turn_id": raw.get("created_turn_id"),
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return "v3-legacy-action-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)

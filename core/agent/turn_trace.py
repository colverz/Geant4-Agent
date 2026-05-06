from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.types import ActionSafetyClass
from .workflow_graph import WorkflowNode, WorkflowTerminalState


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class NluTurnTrace:
    schema_version: str = "nlu_turn_trace.v1"
    turn_id_before: int = 0
    turn_id_after: int = 0
    intent: str = "normal_chat"
    action_safety_class: ActionSafetyClass = ActionSafetyClass.READ_ONLY
    terminal_state: WorkflowTerminalState = WorkflowTerminalState.READ_ONLY_ANSWER
    node_sequence: list[WorkflowNode] = field(default_factory=list)
    prompt_profile_id: str = ""
    llm_used: bool = False
    fallback_reason: str | None = None
    candidate_patch_paths: list[str] = field(default_factory=list)
    confirmation_required: bool = False
    applied_paths: list[str] = field(default_factory=list)
    rejected_paths: list[str] = field(default_factory=list)
    context_pack_hash: str = ""
    patch_hash: str = ""
    grounding_status: str = "not_checked"
    interrupt_status: str = "none"
    idempotency_key: str = ""
    runtime_payload_ready: bool = False
    tool_calls_allowed: list[str] = field(default_factory=list)
    tool_calls_blocked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_safety_class"] = self.action_safety_class.value
        data["terminal_state"] = self.terminal_state.value
        data["node_sequence"] = [node.value for node in self.node_sequence]
        return data


__all__ = ["NluTurnTrace", "stable_hash"]

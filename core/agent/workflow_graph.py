from __future__ import annotations

from enum import Enum


class WorkflowNode(str, Enum):
    START = "start"
    ROUTE_INTENT = "route_intent"
    BUILD_CONTEXT = "build_context"
    INTERPRET = "interpret"
    CHECK_GROUNDING = "check_grounding"
    NORMALIZE_PATCH = "normalize_patch"
    VALIDATE = "validate"
    CONFIRMATION_POLICY = "confirmation_policy"
    WAIT_CONFIRMATION = "wait_confirmation"
    APPLY_SESSION = "apply_session"
    RUNTIME_GUARD = "runtime_guard"
    ANSWER = "answer"
    END = "end"


class WorkflowTerminalState(str, Enum):
    READ_ONLY_ANSWER = "read_only_answer"
    MUTATION_APPLIED = "mutation_applied"
    WAITING_CONFIRMATION = "waiting_confirmation"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    RUNTIME_ACTION_GUARDED = "runtime_action_guarded"
    ERROR = "error"


READ_ONLY_INTENTS = {"read_config", "read_summary", "normal_chat"}
RUNTIME_GUARDED_INTENTS = {"run_requested", "viewer_requested"}


def terminal_state_for_intent(
    intent: str,
    *,
    mutation_applied: bool = False,
    waiting_confirmation: bool = False,
    rejected: bool = False,
    unsupported: bool = False,
) -> WorkflowTerminalState:
    intent_key = str(intent or "normal_chat")
    if unsupported:
        return WorkflowTerminalState.UNSUPPORTED
    if rejected:
        return WorkflowTerminalState.REJECTED
    if intent_key in RUNTIME_GUARDED_INTENTS:
        return WorkflowTerminalState.RUNTIME_ACTION_GUARDED
    if waiting_confirmation:
        return WorkflowTerminalState.WAITING_CONFIRMATION
    if mutation_applied:
        return WorkflowTerminalState.MUTATION_APPLIED
    return WorkflowTerminalState.READ_ONLY_ANSWER


def graph_path_for_intent(
    intent: str,
    *,
    mutation_applied: bool = False,
    waiting_confirmation: bool = False,
    rejected: bool = False,
    unsupported: bool = False,
) -> list[WorkflowNode]:
    intent_key = str(intent or "normal_chat")
    path = [WorkflowNode.START, WorkflowNode.ROUTE_INTENT]
    if intent_key in RUNTIME_GUARDED_INTENTS:
        return [*path, WorkflowNode.RUNTIME_GUARD, WorkflowNode.ANSWER, WorkflowNode.END]
    if intent_key in READ_ONLY_INTENTS:
        return [*path, WorkflowNode.ANSWER, WorkflowNode.END]

    path.extend(
        [
            WorkflowNode.BUILD_CONTEXT,
            WorkflowNode.INTERPRET,
            WorkflowNode.CHECK_GROUNDING,
            WorkflowNode.NORMALIZE_PATCH,
            WorkflowNode.VALIDATE,
            WorkflowNode.CONFIRMATION_POLICY,
        ]
    )
    if unsupported:
        return [*path, WorkflowNode.ANSWER, WorkflowNode.END]
    if waiting_confirmation:
        return [*path, WorkflowNode.WAIT_CONFIRMATION, WorkflowNode.ANSWER, WorkflowNode.END]
    if rejected:
        return [*path, WorkflowNode.ANSWER, WorkflowNode.END]
    if mutation_applied:
        return [*path, WorkflowNode.APPLY_SESSION, WorkflowNode.ANSWER, WorkflowNode.END]
    return [*path, WorkflowNode.ANSWER, WorkflowNode.END]


def assert_path_invariants(intent: str, path: list[WorkflowNode]) -> None:
    nodes = set(path)
    intent_key = str(intent or "normal_chat")
    if intent_key in READ_ONLY_INTENTS and WorkflowNode.APPLY_SESSION in nodes:
        raise AssertionError(f"read-only intent reached apply_session: {intent_key}")
    if intent_key == "normal_chat" and WorkflowNode.VALIDATE in nodes:
        raise AssertionError("normal_chat reached validate")
    if intent_key in RUNTIME_GUARDED_INTENTS and WorkflowNode.APPLY_SESSION in nodes:
        raise AssertionError(f"runtime intent reached apply_session: {intent_key}")
    if intent_key == "config_mutation" and WorkflowNode.APPLY_SESSION in nodes:
        if path.index(WorkflowNode.VALIDATE) > path.index(WorkflowNode.APPLY_SESSION):
            raise AssertionError("config_mutation applied before validate")


__all__ = [
    "WorkflowNode",
    "WorkflowTerminalState",
    "assert_path_invariants",
    "graph_path_for_intent",
    "terminal_state_for_intent",
]

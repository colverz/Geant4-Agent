from __future__ import annotations

from dataclasses import asdict, dataclass

from core.runtime.types import ActionSafetyClass
from .workflow_graph import WorkflowNode, graph_path_for_intent


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    confidence: float
    safety_class: ActionSafetyClass
    requires_kb: bool
    allowed_next_nodes: list[WorkflowNode]
    prompt_profile_id: str = ""
    prompt_validation: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["safety_class"] = self.safety_class.value
        data["allowed_next_nodes"] = [node.value for node in self.allowed_next_nodes]
        data["prompt_validation"] = dict(self.prompt_validation or {})
        return data


def _requires_kb(intent: str) -> bool:
    return intent in {"config_mutation", "clarification_answer", "read_config", "read_summary"}


def route_user_turn(text: str, lang: str = "zh") -> IntentDecision:
    # Compatibility wrapper: keep the existing deterministic runtime-intent rules
    # while exposing a typed agent-router boundary for the new workflow.
    from planner.runtime_intent import classify_user_runtime_intent

    result = classify_user_runtime_intent(text, lang)
    intent = result.intent.value
    path = graph_path_for_intent(intent)
    next_nodes = path[1:-1]
    return IntentDecision(
        intent=intent,
        confidence=1.0 if (result.prompt_validation or {}).get("ok") else 0.0,
        safety_class=result.action_safety_class,
        requires_kb=_requires_kb(intent),
        allowed_next_nodes=next_nodes,
        prompt_profile_id=result.prompt_profile_id,
        prompt_validation=result.prompt_validation,
    )


__all__ = ["IntentDecision", "route_user_turn"]

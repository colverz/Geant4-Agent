from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class V3ActionKind(str, Enum):
    ASK_USER = "ask_user"
    CREATE_DESIGN = "create_design"
    COMPARE_DESIGNS = "compare_designs"
    DRAFT_SPEC = "draft_spec"
    COMMIT_PATCH = "commit_patch"
    RUN_SIMULATION = "run_simulation"
    EXPLAIN_RESULT = "explain_result"
    FINAL_ANSWER = "final_answer"


class V3ToolRiskLevel(IntEnum):
    READ_ONLY = 10
    DRAFT_ONLY = 20
    STATE_MUTATION = 30
    RUNTIME_EXECUTION = 40
    EXTERNAL_SIDE_EFFECT = 50

    @property
    def label(self) -> str:
        return {
            V3ToolRiskLevel.READ_ONLY: "read_only",
            V3ToolRiskLevel.DRAFT_ONLY: "draft_only",
            V3ToolRiskLevel.STATE_MUTATION: "state_mutation",
            V3ToolRiskLevel.RUNTIME_EXECUTION: "runtime_execution",
            V3ToolRiskLevel.EXTERNAL_SIDE_EFFECT: "external_side_effect",
        }[self]


class V3ObservationStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_EVALUABLE = "not_evaluable"


@dataclass(slots=True)
class V3TurnInput:
    session_id: str
    user_text: str
    locale: str = "zh-CN"
    attachments: list[dict[str, Any]] = field(default_factory=list)
    runtime_capability: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_text": self.user_text,
            "locale": self.locale,
            "attachments": list(self.attachments),
            "runtime_capability": dict(self.runtime_capability),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class V3ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    risk_level: V3ToolRiskLevel = V3ToolRiskLevel.READ_ONLY
    idempotency_key: str = ""
    timeout_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "risk_level": self.risk_level.label,
            "idempotency_key": self.idempotency_key,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(slots=True)
class V3ActionProposal:
    kind: V3ActionKind
    intent: str
    arguments: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    risk_level: V3ToolRiskLevel = V3ToolRiskLevel.READ_ONLY
    tool_call: V3ToolCall | None = None
    expected_observation: str = ""
    requires_confirmation: bool = False
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "intent": self.intent,
            "arguments": dict(self.arguments),
            "evidence": list(self.evidence),
            "risk_level": self.risk_level.label,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "expected_observation": self.expected_observation,
            "requires_confirmation": self.requires_confirmation,
            "confirmed": self.confirmed,
        }


@dataclass(slots=True)
class V3Observation:
    source: str
    status: V3ObservationStatus
    data: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    message: str = ""
    not_evaluable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status.value,
            "data": dict(self.data),
            "artifact_paths": list(self.artifact_paths),
            "message": self.message,
            "not_evaluable_reason": self.not_evaluable_reason,
        }


@dataclass(slots=True)
class V3AgentState:
    session_id: str
    goal: str = ""
    assumptions: list[str] = field(default_factory=list)
    active_plan: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    observations: list[V3Observation] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, observation: V3Observation) -> None:
        self.observations.append(observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "assumptions": list(self.assumptions),
            "active_plan": list(self.active_plan),
            "artifacts": dict(self.artifacts),
            "observations": [item.to_dict() for item in self.observations],
            "open_questions": list(self.open_questions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V3AgentState:
        raw_observations = data.get("observations")
        observations: list[V3Observation] = []
        if isinstance(raw_observations, list):
            for item in raw_observations:
                if not isinstance(item, dict):
                    continue
                observations.append(
                    V3Observation(
                        source=str(item.get("source") or ""),
                        status=V3ObservationStatus(str(item.get("status") or "ok")),
                        data=item.get("data") if isinstance(item.get("data"), dict) else {},
                        artifact_paths=[str(p) for p in item.get("artifact_paths") or []] if isinstance(item.get("artifact_paths"), list) else [],
                        message=str(item.get("message") or ""),
                        not_evaluable_reason=str(item.get("not_evaluable_reason") or ""),
                    )
                )
        return cls(
            session_id=str(data.get("session_id") or ""),
            goal=str(data.get("goal") or ""),
            assumptions=[str(item) for item in data.get("assumptions") or []] if isinstance(data.get("assumptions"), list) else [],
            active_plan=[str(item) for item in data.get("active_plan") or []] if isinstance(data.get("active_plan"), list) else [],
            artifacts=data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {},
            observations=observations,
            open_questions=[str(item) for item in data.get("open_questions") or []] if isinstance(data.get("open_questions"), list) else [],
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )


@dataclass(slots=True)
class V3Answer:
    message: str
    understanding: str = ""
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    next_options: list[str] = field(default_factory=list)
    runtime_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "understanding": self.understanding,
            "actions_taken": list(self.actions_taken),
            "evidence": list(self.evidence),
            "next_options": list(self.next_options),
            "runtime_status": self.runtime_status,
        }


@dataclass(slots=True)
class V3AgentResult:
    answer: V3Answer
    state: V3AgentState
    trace: list[dict[str, Any]]
    observations: list[V3Observation] = field(default_factory=list)
    terminated_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer.to_dict(),
            "state": self.state.to_dict(),
            "trace": list(self.trace),
            "observations": [item.to_dict() for item in self.observations],
            "terminated_reason": self.terminated_reason,
        }

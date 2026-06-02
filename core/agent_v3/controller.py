from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .contracts import (
    V3ActionKind,
    V3ActionProposal,
    V3AgentResult,
    V3AgentState,
    V3Answer,
    V3Observation,
    V3ObservationStatus,
    V3ToolRiskLevel,
    V3TurnInput,
)
from .tool_registry import ToolRegistry
from .trace import V3TraceEvent


class V3Reasoner(Protocol):
    def propose(self, turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
        ...


ConstraintReviewer = Callable[[V3ActionProposal, V3AgentState, ToolRegistry], V3Observation | None]


@dataclass(slots=True)
class AgentControllerConfig:
    max_steps: int = 8
    autonomous_risk_limit: V3ToolRiskLevel = V3ToolRiskLevel.DRAFT_ONLY


@dataclass(slots=True)
class AgentController:
    reasoner: V3Reasoner
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    config: AgentControllerConfig = field(default_factory=AgentControllerConfig)
    constraint_reviewer: ConstraintReviewer | None = None

    def run(self, turn: V3TurnInput, state: V3AgentState | None = None) -> V3AgentResult:
        agent_state = state or V3AgentState(session_id=turn.session_id, goal=turn.user_text)
        trace: list[V3TraceEvent] = [
            V3TraceEvent(
                step=0,
                phase="perceive",
                summary="received_user_turn",
                data={"session_id": turn.session_id, "locale": turn.locale},
            )
        ]
        observations: list[V3Observation] = []
        actions_taken: list[dict[str, object]] = []

        for step in range(1, self.config.max_steps + 1):
            proposal = self.reasoner.propose(turn, agent_state)
            trace.append(
                V3TraceEvent(
                    step=step,
                    phase="reason",
                    summary=proposal.intent,
                    data={"proposal": proposal.to_dict()},
                )
            )

            if proposal.kind == V3ActionKind.ASK_USER:
                question = str(proposal.arguments.get("question") or proposal.intent)
                agent_state.open_questions.append(question)
                answer = V3Answer(
                    message=question,
                    understanding=agent_state.goal,
                    actions_taken=actions_taken,
                    next_options=list(proposal.arguments.get("options") or []),
                )
                return self._result(answer, agent_state, trace, observations, "waiting_user")

            if proposal.kind == V3ActionKind.FINAL_ANSWER:
                answer = V3Answer(
                    message=str(proposal.arguments.get("message") or proposal.intent),
                    understanding=agent_state.goal,
                    actions_taken=actions_taken,
                    evidence=list(proposal.evidence),
                    runtime_status=self._latest_runtime_status(agent_state),
                )
                return self._result(answer, agent_state, trace, observations, "final_answer")

            review = self._review(proposal, agent_state)
            if review is not None:
                observations.append(review)
                agent_state.add_observation(review)
                trace.append(
                    V3TraceEvent(
                        step=step,
                        phase="review_constraints",
                        summary=review.status.value,
                        data={"observation": review.to_dict()},
                    )
                )
                if review.status == V3ObservationStatus.BLOCKED:
                    answer = V3Answer(
                        message=review.message or "当前行动被约束审查拦截。",
                        understanding=agent_state.goal,
                        actions_taken=actions_taken,
                        evidence=[review.to_dict()],
                        runtime_status=self._latest_runtime_status(agent_state),
                    )
                    return self._result(answer, agent_state, trace, observations, "blocked")

            gate = self._gate(proposal)
            if gate is not None:
                observations.append(gate)
                agent_state.add_observation(gate)
                trace.append(
                    V3TraceEvent(
                        step=step,
                        phase="commit_gate",
                        summary=gate.status.value,
                        data={"observation": gate.to_dict()},
                    )
                )
                answer = V3Answer(
                    message=gate.message or "这个行动需要确认后才能执行。",
                    understanding=agent_state.goal,
                    actions_taken=actions_taken,
                    evidence=[gate.to_dict()],
                    next_options=["确认执行", "调整方案", "取消行动"],
                    runtime_status=self._latest_runtime_status(agent_state),
                )
                return self._result(answer, agent_state, trace, observations, "waiting_confirmation")

            observation = self._act(proposal)
            observations.append(observation)
            agent_state.add_observation(observation)
            actions_taken.append({"kind": proposal.kind.value, "intent": proposal.intent})
            trace.append(
                V3TraceEvent(
                    step=step,
                    phase="observe",
                    summary=observation.status.value,
                    data={"observation": observation.to_dict()},
                )
            )

            if proposal.kind in {V3ActionKind.CREATE_DESIGN, V3ActionKind.DRAFT_SPEC}:
                artifact_id = str(proposal.arguments.get("artifact_id") or proposal.intent)
                agent_state.artifacts[artifact_id] = observation.data or dict(proposal.arguments)

            if proposal.kind in {V3ActionKind.RUN_SIMULATION, V3ActionKind.EXPLAIN_RESULT}:
                answer = V3Answer(
                    message=observation.message or "已完成行动并获得观察结果。",
                    understanding=agent_state.goal,
                    actions_taken=actions_taken,
                    evidence=[observation.to_dict()],
                    runtime_status=observation.status.value,
                )
                return self._result(answer, agent_state, trace, observations, "observed")

        answer = V3Answer(
            message="已达到本轮智能体步数预算，当前结果已保存到 trace。",
            understanding=agent_state.goal,
            actions_taken=actions_taken,
            evidence=[item.to_dict() for item in observations],
            runtime_status=self._latest_runtime_status(agent_state),
        )
        return self._result(answer, agent_state, trace, observations, "budget_exhausted")

    def _review(self, proposal: V3ActionProposal, state: V3AgentState) -> V3Observation | None:
        if self.constraint_reviewer is None:
            return None
        return self.constraint_reviewer(proposal, state, self.tools)

    def _gate(self, proposal: V3ActionProposal) -> V3Observation | None:
        risk = self._effective_risk(proposal)
        if risk <= self.config.autonomous_risk_limit and not proposal.requires_confirmation:
            return None
        if proposal.confirmed:
            return None
        return V3Observation(
            source="commit_gate",
            status=V3ObservationStatus.BLOCKED,
            data={"risk_level": risk.label, "proposal": proposal.to_dict()},
            message=f"行动 `{proposal.intent}` 的风险等级为 {risk.label}，需要确认后执行。",
        )

    def _effective_risk(self, proposal: V3ActionProposal) -> V3ToolRiskLevel:
        if proposal.tool_call is None:
            return proposal.risk_level
        return self.tools.effective_risk(proposal.tool_call)

    def _act(self, proposal: V3ActionProposal) -> V3Observation:
        if proposal.tool_call is not None:
            return self.tools.invoke(proposal.tool_call)
        return V3Observation(
            source=proposal.kind.value,
            status=V3ObservationStatus.OK,
            data=dict(proposal.arguments),
            message=str(proposal.arguments.get("message") or proposal.intent),
        )

    def _result(
        self,
        answer: V3Answer,
        state: V3AgentState,
        trace: list[V3TraceEvent],
        observations: list[V3Observation],
        reason: str,
    ) -> V3AgentResult:
        return V3AgentResult(
            answer=answer,
            state=state,
            trace=[item.to_dict() for item in trace],
            observations=list(observations),
            terminated_reason=reason,
        )

    def _latest_runtime_status(self, state: V3AgentState) -> str:
        if not state.observations:
            return ""
        return state.observations[-1].status.value

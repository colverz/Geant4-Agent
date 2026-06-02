from __future__ import annotations

import unittest

from core.agent_v3 import (
    AgentController,
    ToolRegistry,
    ToolSpec,
    V3ActionKind,
    V3ActionProposal,
    V3AgentState,
    V3Observation,
    V3ObservationStatus,
    V3ToolCall,
    V3ToolRiskLevel,
    V3TurnInput,
)


class QueueReasoner:
    def __init__(self, proposals: list[V3ActionProposal]) -> None:
        self._proposals = list(proposals)

    def propose(self, turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
        if self._proposals:
            return self._proposals.pop(0)
        return V3ActionProposal(
            kind=V3ActionKind.FINAL_ANSWER,
            intent="done",
            arguments={"message": "done"},
        )


class AgentV3ControllerTest(unittest.TestCase):
    def test_read_only_tool_runs_without_commit_gate(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="geant4_capability_tool",
                risk_level=V3ToolRiskLevel.READ_ONLY,
                handler=lambda call: V3Observation(
                    source=call.tool_name,
                    status=V3ObservationStatus.OK,
                    data={"geant4_available": True},
                    message="Geant4 runtime is available.",
                ),
            )
        )
        reasoner = QueueReasoner(
            [
                V3ActionProposal(
                    kind=V3ActionKind.CREATE_DESIGN,
                    intent="inspect_geant4_capability",
                    tool_call=V3ToolCall(tool_name="geant4_capability_tool"),
                ),
                V3ActionProposal(
                    kind=V3ActionKind.FINAL_ANSWER,
                    intent="answer",
                    arguments={"message": "可以继续设计屏蔽实验。"},
                ),
            ]
        )
        controller = AgentController(reasoner=reasoner, tools=registry)

        result = controller.run(V3TurnInput(session_id="s1", user_text="帮我设计屏蔽实验"))

        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertEqual(result.observations[0].data["geant4_available"], True)
        self.assertNotIn("commit_gate", [event["phase"] for event in result.trace])

    def test_runtime_tool_is_blocked_without_confirmation(self) -> None:
        reasoner = QueueReasoner(
            [
                V3ActionProposal(
                    kind=V3ActionKind.RUN_SIMULATION,
                    intent="run_geant4",
                    tool_call=V3ToolCall(
                        tool_name="geant4_runtime_tool",
                        risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                    ),
                    risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                )
            ]
        )
        controller = AgentController(reasoner=reasoner, tools=ToolRegistry())

        result = controller.run(V3TurnInput(session_id="s2", user_text="直接跑一下"))

        self.assertEqual(result.terminated_reason, "waiting_confirmation")
        self.assertEqual(result.observations[0].source, "commit_gate")
        self.assertIn("需要确认", result.answer.message)
        self.assertIn("commit_gate", [event["phase"] for event in result.trace])

    def test_gate_uses_registered_tool_risk_not_proposal_claim(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="geant4_runtime_tool",
                risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                handler=lambda call: V3Observation(source=call.tool_name, status=V3ObservationStatus.OK),
            )
        )
        reasoner = QueueReasoner(
            [
                V3ActionProposal(
                    kind=V3ActionKind.RUN_SIMULATION,
                    intent="run_geant4_with_wrong_risk",
                    tool_call=V3ToolCall(
                        tool_name="geant4_runtime_tool",
                        risk_level=V3ToolRiskLevel.READ_ONLY,
                    ),
                    risk_level=V3ToolRiskLevel.READ_ONLY,
                    requires_confirmation=False,
                )
            ]
        )
        controller = AgentController(reasoner=reasoner, tools=registry)

        result = controller.run(V3TurnInput(session_id="risk-gate", user_text="run now"))

        self.assertEqual(result.terminated_reason, "waiting_confirmation")
        self.assertEqual(result.observations[0].source, "commit_gate")
        self.assertEqual(result.observations[0].data["risk_level"], "runtime_execution")

    def test_confirmed_runtime_tool_returns_not_evaluable_when_unregistered(self) -> None:
        reasoner = QueueReasoner(
            [
                V3ActionProposal(
                    kind=V3ActionKind.RUN_SIMULATION,
                    intent="run_geant4",
                    tool_call=V3ToolCall(
                        tool_name="geant4_runtime_tool",
                        risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                    ),
                    risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                    confirmed=True,
                )
            ]
        )
        controller = AgentController(reasoner=reasoner, tools=ToolRegistry())

        result = controller.run(V3TurnInput(session_id="s3", user_text="确认运行"))

        self.assertEqual(result.terminated_reason, "observed")
        self.assertEqual(result.observations[0].status, V3ObservationStatus.NOT_EVALUABLE)
        self.assertEqual(result.observations[0].not_evaluable_reason, "tool_not_registered")
        self.assertEqual(result.answer.runtime_status, "not_evaluable")

    def test_constraint_reviewer_blocks_unrepairable_action_before_tool_call(self) -> None:
        def reviewer(proposal: V3ActionProposal, state: V3AgentState, tools: ToolRegistry) -> V3Observation:
            return V3Observation(
                source="constraint_review",
                status=V3ObservationStatus.BLOCKED,
                message="缺少 scoring 定义，不能提交为可运行配置。",
            )

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="simulation_spec_tool",
                risk_level=V3ToolRiskLevel.STATE_MUTATION,
                handler=lambda call: V3Observation(source=call.tool_name, status=V3ObservationStatus.OK),
            )
        )
        reasoner = QueueReasoner(
            [
                V3ActionProposal(
                    kind=V3ActionKind.COMMIT_PATCH,
                    intent="commit_spec",
                    tool_call=V3ToolCall(
                        tool_name="simulation_spec_tool",
                        risk_level=V3ToolRiskLevel.STATE_MUTATION,
                    ),
                    confirmed=True,
                )
            ]
        )
        controller = AgentController(reasoner=reasoner, tools=registry, constraint_reviewer=reviewer)

        result = controller.run(V3TurnInput(session_id="s4", user_text="接受方案"))

        self.assertEqual(result.terminated_reason, "blocked")
        self.assertEqual(result.observations[0].source, "constraint_review")
        self.assertIn("缺少 scoring", result.answer.message)


if __name__ == "__main__":
    unittest.main()

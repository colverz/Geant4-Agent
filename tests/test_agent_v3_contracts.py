from __future__ import annotations

import unittest

from core.agent_v3 import (
    V3ActionKind,
    V3ActionProposal,
    V3AgentState,
    V3Observation,
    V3ObservationStatus,
    V3ToolCall,
    V3ToolRiskLevel,
    V3TurnInput,
)


class AgentV3ContractsTest(unittest.TestCase):
    def test_turn_input_and_state_are_serializable(self) -> None:
        turn = V3TurnInput(
            session_id="s1",
            user_text="评估铅屏蔽 gamma 的效果",
            runtime_capability={"geant4": True},
        )
        state = V3AgentState(session_id=turn.session_id, goal=turn.user_text)
        state.assumptions.append("默认使用 662 keV gamma")
        state.add_observation(
            V3Observation(
                source="geant4_capability_tool",
                status=V3ObservationStatus.OK,
                data={"geant4": True},
            )
        )

        self.assertEqual(turn.to_dict()["runtime_capability"]["geant4"], True)
        self.assertEqual(state.to_dict()["observations"][0]["status"], "ok")
        self.assertEqual(state.to_dict()["assumptions"], ["默认使用 662 keV gamma"])

    def test_action_proposal_can_represent_tool_call_risk(self) -> None:
        proposal = V3ActionProposal(
            kind=V3ActionKind.RUN_SIMULATION,
            intent="run_geant4",
            tool_call=V3ToolCall(
                tool_name="geant4_runtime_tool",
                arguments={"events": 1000},
                risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                idempotency_key="run-1",
            ),
            risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
            requires_confirmation=True,
        )

        payload = proposal.to_dict()
        self.assertEqual(payload["kind"], "run_simulation")
        self.assertEqual(payload["risk_level"], "runtime_execution")
        self.assertEqual(payload["tool_call"]["tool_name"], "geant4_runtime_tool")
        self.assertTrue(payload["requires_confirmation"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from core.agent_v3.contracts import (
    V3ActionKind,
    V3ActionProposal,
    V3AgentState,
    V3Observation,
    V3ObservationStatus,
    V3ToolCall,
    V3ToolRiskLevel,
)
from core.agent_v3.proposal_critic import review_v3_proposal
from core.agent_v3.service import build_v3_agent_controller
from core.agent_v3.tools import (
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
    build_default_geant4_tool_registry,
)


def _runtime_proposal() -> V3ActionProposal:
    return V3ActionProposal(
        kind=V3ActionKind.RUN_SIMULATION,
        intent="run_geant4_runtime",
        tool_call=V3ToolCall(
            tool_name=GEANT4_RUNTIME_TOOL,
            arguments={"events": 3, "allow_in_memory": True},
            risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
        ),
        risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
        requires_confirmation=True,
    )


class V3ProposalCriticTest(unittest.TestCase):
    def test_blocks_runtime_without_payload(self) -> None:
        state = V3AgentState(session_id="critic-no-payload")

        review = review_v3_proposal(_runtime_proposal(), state)

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.source, "proposal_critic")
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "runtime_without_payload")

    def test_blocks_runtime_without_ok_preflight(self) -> None:
        state = V3AgentState(session_id="critic-no-preflight")
        state.add_observation(V3Observation(source=GEANT4_PAYLOAD_BUILDER_TOOL, status=V3ObservationStatus.OK))
        state.add_observation(
            V3Observation(source=GEANT4_RUNTIME_PREFLIGHT_TOOL, status=V3ObservationStatus.NOT_EVALUABLE)
        )

        review = review_v3_proposal(_runtime_proposal(), state)

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "runtime_without_preflight")

    def test_allows_runtime_after_payload_and_preflight_are_ok(self) -> None:
        state = V3AgentState(session_id="critic-ready")
        state.add_observation(V3Observation(source=GEANT4_PAYLOAD_BUILDER_TOOL, status=V3ObservationStatus.OK))
        state.add_observation(V3Observation(source=GEANT4_RUNTIME_PREFLIGHT_TOOL, status=V3ObservationStatus.OK))

        review = review_v3_proposal(_runtime_proposal(), state)

        self.assertIsNone(review)

    def test_blocks_unknown_tool_before_registry_invoke(self) -> None:
        state = V3AgentState(session_id="critic-unknown-tool")
        proposal = V3ActionProposal(
            kind=V3ActionKind.CREATE_DESIGN,
            intent="call_unknown_tool",
            tool_call=V3ToolCall(tool_name="not_a_real_v3_tool"),
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "unknown_tool")

    def test_blocks_schema_invalid_tool_arguments_with_repair_suggestions(self) -> None:
        state = V3AgentState(session_id="critic-schema-invalid")
        state.add_observation(V3Observation(source=GEANT4_PAYLOAD_BUILDER_TOOL, status=V3ObservationStatus.OK))
        state.add_observation(V3Observation(source=GEANT4_RUNTIME_PREFLIGHT_TOOL, status=V3ObservationStatus.OK))
        proposal = V3ActionProposal(
            kind=V3ActionKind.RUN_SIMULATION,
            intent="run_geant4_runtime",
            tool_call=V3ToolCall(
                tool_name=GEANT4_RUNTIME_TOOL,
                arguments={"events": "bad"},
                risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
            ),
            risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "tool_schema_invalid")
        self.assertIn("schema_errors", review.data)
        self.assertIn("repair_suggestions", review.data)

    def test_records_risk_mismatch_as_non_blocking_review(self) -> None:
        state = V3AgentState(session_id="critic-risk-mismatch")
        state.add_observation(V3Observation(source=GEANT4_PAYLOAD_BUILDER_TOOL, status=V3ObservationStatus.OK, data={}))
        proposal = V3ActionProposal(
            kind=V3ActionKind.DRAFT_SPEC,
            intent="preflight_with_wrong_risk",
            tool_call=V3ToolCall(
                tool_name=GEANT4_RUNTIME_PREFLIGHT_TOOL,
                arguments={"payload_builder_observation": {}, "events": 3, "allow_in_memory": True, "env": {}},
                risk_level=V3ToolRiskLevel.READ_ONLY,
            ),
            risk_level=V3ToolRiskLevel.READ_ONLY,
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.OK)
        self.assertEqual(review.data["reason"], "tool_contract_risk_mismatch")
        self.assertEqual(review.data["registered_risk"], "draft_only")

    def test_blocks_payload_builder_design_not_grounded_in_context(self) -> None:
        state = V3AgentState(session_id="critic-design-grounding")
        state.add_observation(
            V3Observation(
                source=GEANT4_DESIGN_TEMPLATE_TOOL,
                status=V3ObservationStatus.OK,
                data={"design": {"recommended_setup": {"material": "G4_Pb"}, "observables": ["detector_crossing_count"]}},
            )
        )
        proposal = V3ActionProposal(
            kind=V3ActionKind.DRAFT_SPEC,
            intent="draft_payload_from_hallucinated_design",
            tool_call=V3ToolCall(
                tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                arguments={"design": {"recommended_setup": {"material": "G4_WATER"}}, "events": 3},
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            ),
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "context_fact_not_grounded")
        self.assertIn("ungrounded_design_argument", review.data["grounding_errors"])

    def test_blocks_runtime_payload_not_grounded_in_latest_payload(self) -> None:
        state = V3AgentState(session_id="critic-payload-grounding")
        state.add_observation(
            V3Observation(
                source=GEANT4_PAYLOAD_BUILDER_TOOL,
                status=V3ObservationStatus.OK,
                data={"recommended_config": {"materials": {"selected_materials": ["G4_Pb"]}}},
            )
        )
        state.add_observation(V3Observation(source=GEANT4_RUNTIME_PREFLIGHT_TOOL, status=V3ObservationStatus.OK))
        proposal = V3ActionProposal(
            kind=V3ActionKind.RUN_SIMULATION,
            intent="run_hallucinated_payload",
            tool_call=V3ToolCall(
                tool_name=GEANT4_RUNTIME_TOOL,
                arguments={
                    "payload_builder_observation": {"recommended_config": {"materials": {"selected_materials": ["G4_WATER"]}}},
                    "events": 3,
                    "allow_in_memory": True,
                },
                risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
            ),
            risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
            confirmed=True,
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "context_fact_not_grounded")
        self.assertIn("ungrounded_payload_argument", review.data["grounding_errors"])

    def test_allows_grounded_payload_builder_design(self) -> None:
        design = {"recommended_setup": {"material": "G4_Pb"}, "observables": ["detector_crossing_count"]}
        state = V3AgentState(session_id="critic-grounded-design")
        state.add_observation(
            V3Observation(
                source=GEANT4_DESIGN_TEMPLATE_TOOL,
                status=V3ObservationStatus.OK,
                data={"design": design},
            )
        )
        proposal = V3ActionProposal(
            kind=V3ActionKind.DRAFT_SPEC,
            intent="draft_grounded_payload",
            tool_call=V3ToolCall(
                tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                arguments={"design": design, "events": 3},
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            ),
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNone(review)

    def test_blocks_internal_only_tool_argument(self) -> None:
        state = V3AgentState(session_id="critic-internal-arg")
        proposal = V3ActionProposal(
            kind=V3ActionKind.CREATE_DESIGN,
            intent="draft_design",
            tool_call=V3ToolCall(
                tool_name="geant4_design_template_tool",
                arguments={"goal": "gamma shielding", "run_confirmed": True},
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
            ),
            risk_level=V3ToolRiskLevel.DRAFT_ONLY,
        )

        review = review_v3_proposal(proposal, state, build_default_geant4_tool_registry())

        self.assertIsNotNone(review)
        assert review is not None
        self.assertEqual(review.status, V3ObservationStatus.BLOCKED)
        self.assertEqual(review.data["reason"], "internal_argument_not_allowed")

    def test_default_v3_controller_uses_proposal_critic(self) -> None:
        controller = build_v3_agent_controller()

        self.assertIs(controller.constraint_reviewer, review_v3_proposal)


if __name__ == "__main__":
    unittest.main()

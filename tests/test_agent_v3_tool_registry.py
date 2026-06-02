from __future__ import annotations

import unittest

from core.agent_v3 import ToolRegistry, ToolSpec, V3Observation, V3ObservationStatus, V3ToolCall, V3ToolRiskLevel


class V3ToolRegistryTest(unittest.TestCase):
    def test_invoke_rejects_missing_required_input(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="payload_builder",
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                input_schema={
                    "type": "object",
                    "required": ["design"],
                    "properties": {"design": {"type": "object"}},
                    "additionalProperties": False,
                },
                handler=lambda call: V3Observation(source=call.tool_name, status=V3ObservationStatus.OK),
            )
        )

        observation = registry.invoke(V3ToolCall(tool_name="payload_builder", arguments={}))

        self.assertEqual(observation.status, V3ObservationStatus.NOT_EVALUABLE)
        self.assertEqual(observation.not_evaluable_reason, "input_schema_validation_failed")
        self.assertIn("missing_required:design", observation.data["schema_errors"])

    def test_invoke_rejects_unexpected_input_property(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="preflight",
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                input_schema={
                    "type": "object",
                    "properties": {"events": {"type": "integer", "minimum": 1}},
                    "additionalProperties": False,
                },
                handler=lambda call: V3Observation(source=call.tool_name, status=V3ObservationStatus.OK),
            )
        )

        observation = registry.invoke(V3ToolCall(tool_name="preflight", arguments={"events": 3, "run_confirmed": True}))

        self.assertEqual(observation.status, V3ObservationStatus.NOT_EVALUABLE)
        self.assertIn("unexpected_property:run_confirmed", observation.data["schema_errors"])

    def test_invoke_rejects_wrong_type_and_minimum(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="preflight",
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                input_schema={
                    "type": "object",
                    "properties": {"events": {"type": "integer", "minimum": 1}},
                    "additionalProperties": False,
                },
                handler=lambda call: V3Observation(source=call.tool_name, status=V3ObservationStatus.OK),
            )
        )

        bad_type = registry.invoke(V3ToolCall(tool_name="preflight", arguments={"events": "3"}))
        too_small = registry.invoke(V3ToolCall(tool_name="preflight", arguments={"events": 0}))

        self.assertIn("type_mismatch:events:expected_integer", bad_type.data["schema_errors"])
        self.assertIn("minimum:events:1", too_small.data["schema_errors"])

    def test_valid_input_invokes_handler(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="preflight",
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                input_schema={
                    "type": "object",
                    "properties": {"events": {"type": "integer", "minimum": 1}},
                    "additionalProperties": False,
                },
                handler=lambda call: V3Observation(source=call.tool_name, status=V3ObservationStatus.OK, data={"events": call.arguments["events"]}),
            )
        )

        observation = registry.invoke(V3ToolCall(tool_name="preflight", arguments={"events": 3}))

        self.assertEqual(observation.status, V3ObservationStatus.OK)
        self.assertEqual(observation.data["events"], 3)


if __name__ == "__main__":
    unittest.main()

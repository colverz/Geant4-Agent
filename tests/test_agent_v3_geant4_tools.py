from __future__ import annotations

import unittest
from unittest.mock import patch

from core.agent_v3 import (
    AgentController,
    BasicGeant4Reasoner,
    V3ObservationStatus,
    V3ToolCall,
    V3ToolRiskLevel,
    V3TurnInput,
)
from core.agent_v3.tools import (
    GEANT4_CAPABILITY_TOOL,
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_LLM_DESIGN_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
    build_default_geant4_tool_registry,
)


class AgentV3Geant4ToolsTest(unittest.TestCase):
    def test_registered_tools_have_contract_metadata(self) -> None:
        registry = build_default_geant4_tool_registry()

        tools = {item["name"]: item for item in registry.list_tools()}

        self.assertIn(GEANT4_RUNTIME_TOOL, tools)
        for spec in tools.values():
            self.assertTrue(spec["description"])
            self.assertEqual(spec["input_schema"]["type"], "object")
            self.assertEqual(spec["output_schema"]["type"], "object")
            self.assertTrue(spec["idempotency_hint"])
        self.assertTrue(tools[GEANT4_RUNTIME_TOOL]["confirmation_required"])
        self.assertFalse(tools[GEANT4_PAYLOAD_BUILDER_TOOL]["confirmation_required"])

    def test_capability_tool_exposes_runtime_and_design_capabilities(self) -> None:
        registry = build_default_geant4_tool_registry()

        observation = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_CAPABILITY_TOOL,
                risk_level=V3ToolRiskLevel.READ_ONLY,
            )
        )

        self.assertEqual(observation.status, V3ObservationStatus.OK)
        self.assertIn("runtime_capabilities", observation.data)
        self.assertIn("design_capabilities", observation.data)
        self.assertIn("shielding", observation.data["supported_scenarios"])

    def test_design_template_tool_turns_chinese_shielding_goal_into_draft(self) -> None:
        registry = build_default_geant4_tool_registry()

        observation = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_DESIGN_TEMPLATE_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={
                    "goal": "我想评估铅屏蔽对 gamma 的透射效果。",
                    "artifact_id": "shielding_draft",
                },
            )
        )

        design = observation.data["design"]
        self.assertEqual(observation.status, V3ObservationStatus.OK)
        self.assertEqual(design["recommended_setup"]["material"], "G4_Pb")
        self.assertIn("detector_crossing_count", design["observables"])
        self.assertEqual(observation.data["artifact"]["artifact_id"], "shielding_draft")

    def test_llm_design_tool_uses_draft_only_boundary(self) -> None:
        registry = build_default_geant4_tool_registry()
        candidate = {
            "schema_version": "simulation_design_candidate.v1",
            "goal": "lead gamma shielding",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": "downstream silicon detector",
                "scoring": ["detector_crossing_count"],
            },
            "observables": ["detector_crossing_count"],
            "assumptions": ["1 MeV gamma beam"],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb"],
            "capability_check": {"supported": True},
            "next_action": "build_candidate_config",
        }

        with patch(
            "core.agent_v3.tools.geant4_tools.bridge_llm_design_candidate",
            return_value={
                "ok": True,
                "candidate": candidate,
                "prompt_profile_id": "test-profile",
                "prompt_validation": {"ok": True, "errors": []},
                "reference_pack": {"schema_version": "reference.v1"},
            },
        ):
            observation = registry.invoke(
                V3ToolCall(
                    tool_name=GEANT4_LLM_DESIGN_TOOL,
                    risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                    arguments={
                        "goal": "lead gamma shielding",
                        "llm_config_path": "fake.json",
                    },
                )
            )

        self.assertEqual(observation.status, V3ObservationStatus.OK)
        self.assertEqual(observation.source, GEANT4_LLM_DESIGN_TOOL)
        self.assertTrue(observation.data["llm"]["used"])
        self.assertTrue(observation.data["llm"]["ok"])
        self.assertEqual(observation.data["design"]["recommended_setup"]["material"], "G4_Pb")

    def test_llm_design_tool_preserves_explicit_goal_material(self) -> None:
        registry = build_default_geant4_tool_registry()
        candidate = {
            "schema_version": "simulation_design_candidate.v1",
            "goal": "100 MeV proton in water equivalent material",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": None,
                "scoring": ["target_edep"],
            },
            "observables": ["target_edep"],
            "assumptions": [],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb"],
            "capability_check": {"supported": True},
            "next_action": "build_candidate_config",
        }

        with patch(
            "core.agent_v3.tools.geant4_tools.bridge_llm_design_candidate",
            return_value={
                "ok": True,
                "candidate": candidate,
                "prompt_profile_id": "test-profile",
                "prompt_validation": {"ok": True, "errors": []},
                "reference_pack": {"schema_version": "reference.v1"},
            },
        ):
            observation = registry.invoke(
                V3ToolCall(
                    tool_name=GEANT4_LLM_DESIGN_TOOL,
                    risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                    arguments={
                        "goal": "Design a 100 MeV proton simulation in G4_WATER, design only.",
                        "llm_config_path": "fake.json",
                    },
                )
            )

        setup = observation.data["design"]["recommended_setup"]
        self.assertEqual(observation.status, V3ObservationStatus.OK)
        self.assertEqual(setup["material"], "G4_WATER")
        self.assertEqual(setup["target_material"], "G4_WATER")
        self.assertEqual(setup["source_energy_mev"], 100.0)
        self.assertEqual(setup["source_particle"], "proton")
        self.assertIn("materials:G4_WATER", observation.data["design"]["knowledge_references"])

    def test_basic_reasoner_can_use_llm_design_tool_when_enabled(self) -> None:
        candidate = {
            "schema_version": "simulation_design_candidate.v1",
            "goal": "lead gamma shielding",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": "downstream silicon detector",
                "scoring": ["detector_crossing_count"],
            },
            "observables": ["detector_crossing_count"],
            "assumptions": [],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb"],
            "capability_check": {"supported": True},
            "next_action": "build_candidate_config",
        }
        controller = AgentController(
            reasoner=BasicGeant4Reasoner(),
            tools=build_default_geant4_tool_registry(),
        )

        with patch(
            "core.agent_v3.tools.geant4_tools.bridge_llm_design_candidate",
            return_value={
                "ok": True,
                "candidate": candidate,
                "prompt_profile_id": "test-profile",
                "prompt_validation": {"ok": True, "errors": []},
                "reference_pack": {"schema_version": "reference.v1"},
            },
        ):
            result = controller.run(
                V3TurnInput(
                    session_id="v3-geant4-llm-design",
                    user_text="我想评估铅屏蔽对 gamma 的透射效果。",
                    metadata={"llm_design_enabled": True, "llm_config_path": "fake.json"},
                )
            )

        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertEqual([item.source for item in result.observations], [GEANT4_CAPABILITY_TOOL, GEANT4_LLM_DESIGN_TOOL])
        self.assertIn("G4_Pb", result.answer.message)

    def test_basic_reasoner_runs_capability_then_design_then_answer(self) -> None:
        controller = AgentController(
            reasoner=BasicGeant4Reasoner(),
            tools=build_default_geant4_tool_registry(),
        )

        result = controller.run(
            V3TurnInput(
                session_id="v3-geant4-design",
                user_text="我想评估铅屏蔽对 gamma 的透射效果。",
            )
        )

        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertIn("Geant4 design draft", result.answer.message)
        self.assertIn("G4_Pb", result.answer.message)
        self.assertEqual([item.source for item in result.observations], [GEANT4_CAPABILITY_TOOL, GEANT4_DESIGN_TEMPLATE_TOOL])
        self.assertIn("observe", [event["phase"] for event in result.trace])

    def test_payload_builder_turns_design_into_runtime_payload_draft(self) -> None:
        registry = build_default_geant4_tool_registry()
        design_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_DESIGN_TEMPLATE_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"goal": "1 MeV gamma lead shielding transmission with silicon detector"},
            )
        )

        payload_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"design": design_obs.data["design"], "events": 25},
            )
        )

        self.assertEqual(payload_obs.status, V3ObservationStatus.OK)
        self.assertEqual(payload_obs.data["recommended_config"]["materials"]["selected_materials"][0], "G4_Pb")
        self.assertEqual(payload_obs.data["simulation_spec"]["source"]["particle"], "gamma")
        self.assertEqual(payload_obs.data["simulation_spec"]["run"]["events"], 25)
        self.assertEqual(payload_obs.data["runtime_payload"]["schema_version"], "runtime_dsl.v1")
        self.assertTrue(payload_obs.data["runtime_payload"]["detector_enabled"])

    def test_basic_reasoner_accept_defaults_drafts_payload_without_runtime_execution(self) -> None:
        controller = AgentController(
            reasoner=BasicGeant4Reasoner(),
            tools=build_default_geant4_tool_registry(),
        )

        result = controller.run(
            V3TurnInput(
                session_id="v3-geant4-payload",
                user_text="我想评估铅屏蔽对 gamma 的透射效果，接受默认参数。",
                metadata={"accept_defaults": True, "events": 12},
            )
        )

        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertIn("runtime payload 草案", result.answer.message)
        self.assertIn("还没有执行 Geant4", result.answer.message)
        self.assertEqual(
            [item.source for item in result.observations],
            [GEANT4_CAPABILITY_TOOL, GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_PAYLOAD_BUILDER_TOOL],
        )

    def test_runtime_preflight_returns_not_evaluable_without_local_process_runtime(self) -> None:
        registry = build_default_geant4_tool_registry()
        design_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_DESIGN_TEMPLATE_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"goal": "1 MeV gamma lead shielding transmission with silicon detector"},
            )
        )
        payload_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"design": design_obs.data["design"], "events": 20},
            )
        )

        preflight_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_RUNTIME_PREFLIGHT_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"payload_builder_observation": payload_obs.data, "events": 20, "env": {}},
            )
        )

        self.assertEqual(preflight_obs.status, V3ObservationStatus.NOT_EVALUABLE)
        self.assertEqual(preflight_obs.not_evaluable_reason, "local_process_runtime_required")
        self.assertTrue(preflight_obs.data["config_ok"])
        self.assertEqual(preflight_obs.data["adapter"], "in_memory")

    def test_runtime_tool_refuses_in_memory_as_real_result(self) -> None:
        registry = build_default_geant4_tool_registry()
        design_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_DESIGN_TEMPLATE_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"goal": "1 MeV gamma lead shielding transmission with silicon detector"},
            )
        )
        payload_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                arguments={"design": design_obs.data["design"], "events": 20},
            )
        )

        runtime_obs = registry.invoke(
            V3ToolCall(
                tool_name=GEANT4_RUNTIME_TOOL,
                risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                arguments={"payload_builder_observation": payload_obs.data, "events": 20, "env": {}},
            )
        )

        self.assertEqual(runtime_obs.status, V3ObservationStatus.NOT_EVALUABLE)
        self.assertEqual(runtime_obs.not_evaluable_reason, "local_process_runtime_required")

    def test_basic_reasoner_run_stops_at_not_evaluable_preflight_without_runtime(self) -> None:
        controller = AgentController(
            reasoner=BasicGeant4Reasoner(),
            tools=build_default_geant4_tool_registry(),
        )

        result = controller.run(
            V3TurnInput(
                session_id="v3-geant4-run",
                user_text="我想评估铅屏蔽对 gamma 的透射效果并运行。",
                metadata={"run": True, "run_confirmed": True, "events": 8},
            )
        )

        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertIn("不能报告真实模拟结果", result.answer.message)
        self.assertEqual(
            [item.source for item in result.observations],
            [
                GEANT4_CAPABILITY_TOOL,
                GEANT4_DESIGN_TEMPLATE_TOOL,
                GEANT4_PAYLOAD_BUILDER_TOOL,
                GEANT4_RUNTIME_PREFLIGHT_TOOL,
            ],
        )


if __name__ == "__main__":
    unittest.main()

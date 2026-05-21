from __future__ import annotations

import unittest

from core.agent import build_agent_plan, build_agent_state, build_critic_report


class AgentPlanCriticTest(unittest.TestCase):
    def test_agent_plan_wraps_simulation_design_candidate(self) -> None:
        design = {
            "goal": "Measure gamma transmission through a lead shield.",
            "recommended_setup": {"geometry": "multi_layer_stack", "source": "beam"},
            "observables": ["plane_crossing_count", "transmission_factor"],
            "assumptions": ["Use downstream plane scoring."],
            "user_decisions_required": [],
            "capability_check": {"supported": True, "unsupported_capabilities": []},
            "next_action": "build_candidate_config",
        }
        config = {"geometry": {"structure": "multi_layer_stack"}}

        plan = build_agent_plan(design, recommended_config=config)
        state = build_agent_state(plan=plan, candidate_status={"status": "proposed"})

        self.assertEqual(plan["schema_version"], "agent_plan.v1")
        self.assertEqual(plan["selected_candidate_id"], "candidate_1")
        self.assertEqual(plan["candidate_set"][0]["recommended_config"], config)
        self.assertIn("scoring:plane_crossing_count", plan["required_runtime_capabilities"])
        self.assertIn("geometry:multi_layer_stack", plan["required_runtime_capabilities"])
        self.assertTrue(state["plan_proposed"])
        self.assertTrue(state["candidate_compiled"])
        self.assertFalse(state["config_committed"])

    def test_critic_reports_missing_metrics_without_claiming_success(self) -> None:
        plan = {
            "success_metrics": ["region_contrast", "detector_crossing_count"],
            "required_runtime_capabilities": ["scoring:region_contrast"],
        }
        report = {
            "ok": True,
            "key_metrics": {"detector_crossing_count": 3},
            "result_summary": {"scoring": {}},
        }

        critic = build_critic_report(report, plan)

        self.assertFalse(critic["satisfied"])
        self.assertFalse(critic["answerable"])
        self.assertEqual(critic["missing_metrics"], ["region_contrast"])
        self.assertEqual(critic["recommended_next_action"], "request_additional_scoring")

    def test_critic_marks_runtime_limitations_from_plan(self) -> None:
        plan = {
            "success_metrics": ["target_edep"],
            "required_runtime_capabilities": ["unsupported:cad_import"],
        }
        report = {
            "ok": True,
            "key_metrics": {"target_edep_total_mev": 1.2},
            "result_summary": {"scoring": {}},
        }

        critic = build_critic_report(report, plan)

        self.assertFalse(critic["satisfied"])
        self.assertTrue(critic["answerable"])
        self.assertEqual(critic["runtime_limitations"], ["cad_import"])
        self.assertEqual(critic["recommended_next_action"], "revise_plan")


if __name__ == "__main__":
    unittest.main()

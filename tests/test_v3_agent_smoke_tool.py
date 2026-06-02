from __future__ import annotations

import unittest

from tools.run_v3_agent_smoke import run_v3_agent_smoke


class V3AgentSmokeToolTest(unittest.TestCase):
    def test_v3_smoke_returns_design_answer_and_trace(self) -> None:
        result = run_v3_agent_smoke(text="我想评估铅屏蔽对 gamma 的透射效果。")

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema_version"], "geant4_agent_v3_smoke.v1")
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertIn("Geant4 design draft", result["answer"]["message"])
        self.assertGreaterEqual(len(result["trace"]), 3)
        self.assertEqual(result["observations"][0]["source"], "geant4_capability_tool")
        self.assertEqual(result["observations"][1]["source"], "geant4_design_template_tool")

    def test_v3_smoke_accept_defaults_returns_payload_draft(self) -> None:
        result = run_v3_agent_smoke(
            text="我想评估铅屏蔽对 gamma 的透射效果，接受默认参数。",
            accept_defaults=True,
            events=15,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertIn("runtime payload 草案", result["answer"]["message"])
        self.assertEqual(result["observations"][2]["source"], "geant4_payload_builder_tool")
        self.assertEqual(result["observations"][2]["data"]["simulation_spec"]["run"]["events"], 15)

    def test_v3_smoke_run_without_runtime_is_not_evaluable(self) -> None:
        result = run_v3_agent_smoke(
            text="我想评估铅屏蔽对 gamma 的透射效果并运行。",
            run=True,
            events=7,
        )

        self.assertTrue(result["ok"])
        self.assertIn("不能报告真实模拟结果", result["answer"]["message"])
        self.assertEqual(result["observations"][3]["source"], "geant4_runtime_preflight_tool")
        self.assertEqual(result["observations"][3]["status"], "not_evaluable")
        self.assertEqual(result["observations"][3]["not_evaluable_reason"], "local_process_runtime_required")


if __name__ == "__main__":
    unittest.main()

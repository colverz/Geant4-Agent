from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.evaluate_llm_scenario_parsing import evaluate_llm_scenario_parsing


class LlmScenarioParsingBenchmarkTest(unittest.TestCase):
    def test_offline_v2_parser_reaches_core_runtime_contract(self) -> None:
        report = evaluate_llm_scenario_parsing()

        self.assertEqual(report["mode"], "offline_v2")
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["passed"], report["total"])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertTrue(report["meets_threshold"])
        self.assertGreaterEqual(report["total"], 3)
        self.assertGreaterEqual(report["known_gap_count"], 1)
        for result in report["results"]:
            self.assertFalse(result["errors"])
            self.assertIn("trajectory", result)
            self.assertIn("runtime_payload_ready", result["trajectory"])

    def test_live_llm_mode_requires_explicit_config_path(self) -> None:
        report = evaluate_llm_scenario_parsing(live_llm=True, llm_config_path="")

        self.assertEqual(report["mode"], "live_llm")
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertFalse(report["meets_threshold"])
        self.assertEqual(report["failures"][0]["errors"], ["missing_llm_config_path"])

    def test_live_llm_mode_rejects_silent_fallback(self) -> None:
        casebank = [
            {
                "id": "fallback_probe",
                "prompt": "10 mm copper box; gamma 1 MeV along +z.",
                "parser_expected": {"is_complete": True},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "casebank.json"
            path.write_text(json.dumps(casebank), encoding="utf-8")
            with patch(
                "tools.evaluate_llm_scenario_parsing.process_turn",
                return_value={
                    "is_complete": True,
                    "config": {},
                    "llm_used": False,
                    "fallback_reason": "E_LLM_FRAME_CALL_FAILED",
                },
            ):
                report = evaluate_llm_scenario_parsing(path, live_llm=True, llm_config_path="dummy.json")

        self.assertEqual(report["mode"], "live_llm")
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertFalse(report["meets_threshold"])
        self.assertEqual(report["failures"][0]["errors"], ["live_llm_not_used:fallback='E_LLM_FRAME_CALL_FAILED'"])

    def test_max_cases_limits_low_cost_live_smoke_scope(self) -> None:
        report = evaluate_llm_scenario_parsing(max_cases=2)

        self.assertEqual(report["mode"], "offline_v2")
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["passed"], 2)
        self.assertEqual(report["accuracy"], 1.0)

    def test_min_accuracy_gate_fails_when_threshold_is_not_met(self) -> None:
        casebank = [
            {
                "id": "threshold_probe",
                "prompt": "10 mm copper box; gamma 1 MeV along +z.",
                "parser_expected": {"runtime": {"particle": "proton"}},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "casebank.json"
            path.write_text(json.dumps(casebank), encoding="utf-8")
            report = evaluate_llm_scenario_parsing(path, min_accuracy=1.0)

        self.assertEqual(report["total"], 1)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertFalse(report["meets_threshold"])

    def test_agent_expected_checks_trajectory_not_only_field_mapping(self) -> None:
        casebank = [
            {
                "id": "agent_trajectory_probe",
                "prompt": "10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm along +z; physics FTFP_BERT; output json.",
                "parser_expected": {},
                "agent_expected": {
                    "must_use_llm": True,
                    "forbid_fallback": True,
                    "must_be_complete": True,
                    "must_have_runtime_payload": True,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "casebank.json"
            path.write_text(json.dumps(casebank), encoding="utf-8")
            report = evaluate_llm_scenario_parsing(path)

        self.assertEqual(report["failed"], 1)
        errors = report["failures"][0]["errors"]
        self.assertTrue(any(error.startswith("agent.llm_used") for error in errors))

    def test_model_override_is_reported_and_restored(self) -> None:
        previous = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
        with patch.dict(os.environ, {}, clear=False):
            if previous is None:
                os.environ.pop("GEANT4_LLM_MODEL_OVERRIDE", None)
            report = evaluate_llm_scenario_parsing(max_cases=1, model_override="deepseek-v4-flash")

        self.assertEqual(report["model_override"], "deepseek-v4-flash")
        self.assertEqual(os.environ.get("GEANT4_LLM_MODEL_OVERRIDE"), previous)

    @unittest.skipUnless(
        os.environ.get("GEANT4_LLM_SCENARIO", "").strip().lower() in {"1", "true", "yes", "on"}
        and os.environ.get("GEANT4_LLM_CONFIG"),
        "live LLM scenario parsing is opt-in via GEANT4_LLM_SCENARIO=1 and GEANT4_LLM_CONFIG",
    )
    def test_live_llm_opt_in_reaches_core_runtime_contract(self) -> None:
        report = evaluate_llm_scenario_parsing(
            live_llm=True,
            llm_config_path=os.environ["GEANT4_LLM_CONFIG"],
        )

        self.assertEqual(report["mode"], "live_llm")
        self.assertEqual(report["failed"], 0)
        for result in report["results"]:
            self.assertTrue(result["llm_used"], result)


if __name__ == "__main__":
    unittest.main()

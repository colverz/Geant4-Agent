from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.evaluate_interpreter_v2_live import evaluate_interpreter_v2_live


class InterpreterV2LiveEvaluatorTest(unittest.TestCase):
    def test_requires_explicit_llm_config_path(self) -> None:
        report = evaluate_interpreter_v2_live(config_path="")

        self.assertFalse(report["ok"])
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["failures"][0]["errors"], ["missing_llm_config_path"])

    def test_evaluator_checks_expected_update_path_and_guarded_action(self) -> None:
        fake_result = mock.Mock()
        fake_result.ok = True
        fake_result.validation.errors = []
        fake_result.prompt_profile_id = "interpret_user_turn_en_v2_path_evidence"
        fake_result.fallback_reason = None
        fake_result.payload = {
            "turn_summary": {"intent": "modify", "focus": "mixed", "user_goal": "change and run", "requires_confirmation": False},
            "candidate_updates": [{"path": "source.energy_mev", "op": "set", "value": 10}],
            "ambiguities": [],
            "unsupported_requests": [],
            "guarded_actions": [{"action": "run_beam", "safety_class": "expensive_runtime", "requested": True}],
        }

        with mock.patch("tools.evaluate_interpreter_v2_live.run_interpreter_v2", return_value=fake_result):
            report = evaluate_interpreter_v2_live(config_path="dummy.json", max_cases=1)

        self.assertTrue(report["ok"], report["failures"])
        self.assertEqual(report["passed"], 1)

    def test_evaluator_reports_contract_mismatch(self) -> None:
        fake_result = mock.Mock()
        fake_result.ok = True
        fake_result.validation.errors = []
        fake_result.prompt_profile_id = "interpret_user_turn_en_v2_path_evidence"
        fake_result.fallback_reason = None
        fake_result.payload = {
            "turn_summary": {"intent": "modify", "focus": "mixed", "user_goal": "bad", "requires_confirmation": False},
            "candidate_updates": [{"path": "geometry.material", "op": "set", "value": "G4_Cu"}],
            "ambiguities": [],
            "unsupported_requests": [],
            "guarded_actions": [],
        }

        with mock.patch("tools.evaluate_interpreter_v2_live.run_interpreter_v2", return_value=fake_result):
            report = evaluate_interpreter_v2_live(config_path="dummy.json", max_cases=1)

        self.assertFalse(report["ok"])
        self.assertIn("missing_update_path:source.energy_mev", report["failures"][0]["errors"])
        self.assertIn("missing_guarded_action:run_beam", report["failures"][0]["errors"])

    def test_model_override_is_restored(self) -> None:
        previous = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
        fake_result = mock.Mock()
        fake_result.ok = True
        fake_result.validation.errors = []
        fake_result.prompt_profile_id = "interpret_user_turn_en_v2_path_evidence"
        fake_result.fallback_reason = None
        fake_result.payload = {
            "candidate_updates": [{"path": "source.energy_mev"}],
            "guarded_actions": [{"action": "run_beam"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            casebank = Path(tmpdir) / "casebank.json"
            casebank.write_text(
                '[{"id":"one","prompt":"Change source energy to 10 MeV and run 10 events now.","expect_update_path":"source.energy_mev","expect_guarded_action":"run_beam"}]',
                encoding="utf-8",
            )
            with mock.patch("tools.evaluate_interpreter_v2_live.run_interpreter_v2", return_value=fake_result):
                report = evaluate_interpreter_v2_live(
                    config_path="dummy.json",
                    casebank_path=casebank,
                    model_override="deepseek-v4-flash",
                )

        self.assertTrue(report["ok"], report["failures"])
        self.assertEqual(report["model_override"], "deepseek-v4-flash")
        self.assertEqual(os.environ.get("GEANT4_LLM_MODEL_OVERRIDE"), previous)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.evaluate_simulation_design_live import evaluate_simulation_design_live


class SimulationDesignLiveEvaluatorTest(unittest.TestCase):
    def test_live_evaluator_checks_llm_candidate_with_capability_boundary(self) -> None:
        casebank = [
            {
                "id": "lead",
                "goal": "Design a 1 MeV gamma transmission study through lead shielding.",
                "expected": {
                    "checked_next_action": "build_candidate_config",
                    "must_be_supported": True,
                    "must_include_observables": ["detector_crossing_count"],
                    "must_reference_any": ["materials:G4_Pb"],
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(json.dumps(casebank), encoding="utf-8")
            llm_payload = {
                "schema_version": "geant4_agent_simulation_design_candidate.v1",
                "goal": casebank[0]["goal"],
                "recommended_setup": {
                    "geometry": "single_box",
                    "material": "G4_Pb",
                    "source": "beam",
                    "detector": {"enabled": True, "material": "G4_Si"},
                },
                "observables": ["detector_crossing_count", "target_edep"],
                "assumptions": [],
                "simplifications": [],
                "unsupported_capabilities": [],
                "user_decisions_required": [],
                "knowledge_references": ["materials:G4_Pb", "scoring:detector_crossing_count"],
                "capability_check": {},
                "next_action": "build_candidate_config",
            }

            with mock.patch(
                "tools.evaluate_simulation_design_live.ollama_client.chat",
                return_value={"response": json.dumps(llm_payload)},
            ):
                report = evaluate_simulation_design_live(
                    casebank=path,
                    live_llm=True,
                    llm_config="dummy.json",
                    model_override="deepseek-v4-flash",
                )

        self.assertTrue(report["ok"])
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["case_results"][0]["candidate"]["checked_next_action"], "build_candidate_config")
        self.assertEqual(report["stage_summary"]["simulation_design"]["supported_count"], 1)

    def test_live_evaluator_fails_when_unsupported_is_leaked_as_supported(self) -> None:
        casebank = [
            {
                "id": "depth-dose",
                "goal": "Design water phantom depth-binned dose scoring.",
                "expected": {
                    "checked_next_action": "unsupported_capability",
                    "must_be_supported": False,
                    "must_include_unsupported": ["depth_binned_scoring"],
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(json.dumps(casebank), encoding="utf-8")
            llm_payload = {
                "schema_version": "geant4_agent_simulation_design_candidate.v1",
                "goal": casebank[0]["goal"],
                "recommended_setup": {"geometry": "single_box", "material": "G4_WATER", "source": "beam"},
                "observables": ["target_edep"],
                "assumptions": [],
                "simplifications": [],
                "unsupported_capabilities": [],
                "user_decisions_required": [],
                "knowledge_references": ["materials:G4_WATER"],
                "capability_check": {},
                "next_action": "build_candidate_config",
            }

            with mock.patch(
                "tools.evaluate_simulation_design_live.ollama_client.chat",
                return_value={"response": json.dumps(llm_payload)},
            ):
                report = evaluate_simulation_design_live(
                    casebank=path,
                    live_llm=True,
                    llm_config="dummy.json",
                    model_override="deepseek-v4-flash",
                )

        self.assertFalse(report["ok"])
        self.assertEqual(report["failed"], 1)
        self.assertTrue(any("checked_next_action" in error for error in report["case_results"][0]["errors"]))


if __name__ == "__main__":
    unittest.main()

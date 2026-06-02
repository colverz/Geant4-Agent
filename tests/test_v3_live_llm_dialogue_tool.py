from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.agent_v3.response_naturalizer import V3ResponseNaturalizationResult
from tools.run_v3_live_llm_dialogue import run_v3_live_llm_dialogue


class V3LiveLlmDialogueToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._sessions_dir = str(Path(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_live_dialogue_skips_without_explicit_live_flag(self) -> None:
        result = run_v3_live_llm_dialogue(sessions_dir=self._sessions_dir,live_llm=False, llm_config="dummy.json")

        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skip_reason"], "live_llm_not_enabled")

    def test_live_dialogue_exposes_raw_dialogue_and_llm_response(self) -> None:
        llm_payload = {
            "schema_version": "geant4_agent_simulation_design_candidate.v1",
            "goal": "lead gamma shielding",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": {"enabled": True, "material": "G4_Si"},
                "scoring": ["detector_crossing_count"],
            },
            "observables": ["detector_crossing_count"],
            "assumptions": ["1 MeV gamma beam"],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb", "scoring:detector_crossing_count"],
            "capability_check": {},
            "next_action": "build_candidate_config",
        }

        with patch(
            "core.agent.simulation_design_llm.ollama_client.chat",
            return_value={"response": json.dumps(llm_payload, ensure_ascii=False)},
        ):
            result = run_v3_live_llm_dialogue(sessions_dir=self._sessions_dir,
                live_llm=True,
                llm_config="dummy.json",
                texts=["Design a 1 MeV gamma lead shielding setup, do not run."],
                include_full_responses=False,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["skipped"])
        self.assertNotIn("responses", result)
        self.assertEqual(result["raw_dialogue"][0]["role"], "user")
        self.assertEqual(result["raw_dialogue"][1]["role"], "agent")
        self.assertIn("raw_message", result["raw_dialogue"][1])
        self.assertEqual(result["raw_dialogue"][1]["dialogue_act"], "design_presented")
        self.assertIn("metrics", result)
        self.assertEqual(result["metrics"]["turn_count"], 1)
        self.assertEqual(result["metrics"]["dialogue_quality"]["reported_count"], 1)
        self.assertGreaterEqual(result["metrics"]["dialogue_quality"]["ok_count"], 0)
        self.assertFalse(result["metrics"]["naturalization"]["enabled"])
        self.assertEqual(result["llm_raw_responses"][0]["source"], "geant4_llm_design_tool")
        self.assertIn("G4_Pb", result["llm_raw_responses"][0]["raw_response"])

    def test_live_dialogue_records_only_current_turn_llm_raw_response(self) -> None:
        llm_payload = {
            "schema_version": "geant4_agent_simulation_design_candidate.v1",
            "goal": "lead gamma shielding",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": {"enabled": True, "material": "G4_Si"},
                "scoring": ["detector_crossing_count"],
            },
            "observables": ["detector_crossing_count"],
            "assumptions": [],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb"],
            "capability_check": {},
            "next_action": "build_candidate_config",
        }

        with patch(
            "core.agent.simulation_design_llm.ollama_client.chat",
            return_value={"response": json.dumps(llm_payload, ensure_ascii=False)},
        ):
            result = run_v3_live_llm_dialogue(sessions_dir=self._sessions_dir,
                live_llm=True,
                llm_config="dummy.json",
                texts=[
                    "Design a 1 MeV gamma lead shielding setup, do not run.",
                    "Change the previous setup to 2 MeV.",
                ],
                include_full_responses=False,
            )

        self.assertEqual(len(result["llm_raw_responses"]), 1)
        self.assertEqual(result["llm_raw_responses"][0]["turn"], "1")

    def test_live_dialogue_resets_reused_session_before_running(self) -> None:
        llm_payload = {
            "schema_version": "geant4_agent_simulation_design_candidate.v1",
            "goal": "lead gamma shielding",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": {"enabled": True, "material": "G4_Si"},
                "scoring": ["detector_crossing_count"],
            },
            "observables": ["detector_crossing_count"],
            "assumptions": [],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb"],
            "capability_check": {},
            "next_action": "build_candidate_config",
        }

        with patch(
            "core.agent.simulation_design_llm.ollama_client.chat",
            return_value={"response": json.dumps(llm_payload, ensure_ascii=False)},
        ):
            run_v3_live_llm_dialogue(
                sessions_dir=self._sessions_dir,
                live_llm=True,
                llm_config="dummy.json",
                session_id="reused-live-session",
                texts=[
                    "Design a 1 MeV gamma lead shielding setup, do not run.",
                    "Change the previous setup to 2 MeV.",
                ],
                include_full_responses=False,
            )
            result = run_v3_live_llm_dialogue(
                sessions_dir=self._sessions_dir,
                live_llm=True,
                llm_config="dummy.json",
                session_id="reused-live-session",
                texts=["Design a 1 MeV gamma lead shielding setup, do not run."],
                include_full_responses=False,
            )

        self.assertEqual(result["raw_dialogue"][1]["dialogue_act"], "design_presented")
        self.assertIn("1 MeV", result["raw_dialogue"][1]["content"])
        self.assertNotIn("2 MeV", result["raw_dialogue"][1]["content"])

    def test_live_dialogue_metrics_include_naturalization_success_and_fallbacks(self) -> None:
        llm_payload = {
            "schema_version": "geant4_agent_simulation_design_candidate.v1",
            "goal": "lead gamma shielding",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": {"enabled": True, "material": "G4_Si"},
                "scoring": ["detector_crossing_count"],
            },
            "observables": ["detector_crossing_count"],
            "assumptions": [],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb"],
            "capability_check": {},
            "next_action": "build_candidate_config",
        }
        naturalization_results = [
            V3ResponseNaturalizationResult(ok=True, used_llm=True, display_message="Naturalized first response."),
            V3ResponseNaturalizationResult(ok=False, used_llm=True, display_message="unsafe", fallback_reason="material_conflict"),
        ]

        with patch(
            "core.agent.simulation_design_llm.ollama_client.chat",
            return_value={"response": json.dumps(llm_payload, ensure_ascii=False)},
        ), patch(
            "core.agent_v3.service.V3ResponseNaturalizer.naturalize",
            side_effect=naturalization_results,
        ):
            result = run_v3_live_llm_dialogue(
                sessions_dir=self._sessions_dir,
                live_llm=True,
                llm_config="dummy.json",
                texts=[
                    "Design a 1 MeV gamma lead shielding setup, do not run.",
                    "Change the previous setup to 2 MeV.",
                ],
                naturalize=True,
                include_full_responses=False,
            )

        metrics = result["metrics"]
        self.assertTrue(metrics["naturalization"]["enabled"])
        self.assertEqual(metrics["naturalization"]["reported_count"], 2)
        self.assertEqual(metrics["naturalization"]["ok_count"], 1)
        self.assertEqual(metrics["naturalization"]["fallback_count"], 1)
        self.assertEqual(metrics["naturalization"]["fallback_reasons"], {"material_conflict": 1})
        self.assertEqual(metrics["naturalization"]["fallback_categories"], {"safety_rejected": 1})
        self.assertEqual(result["raw_dialogue"][1]["naturalization"]["ok"], True)
        self.assertEqual(result["raw_dialogue"][3]["naturalization"]["fallback_reason"], "material_conflict")
        self.assertEqual(result["raw_dialogue"][3]["naturalization"]["fallback_category"], "safety_rejected")


if __name__ == "__main__":
    unittest.main()

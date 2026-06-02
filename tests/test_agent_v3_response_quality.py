from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.agent_v3.response_naturalizer import V3ResponseNaturalizationResult
from core.agent_v3.response_quality import evaluate_v3_response_quality
from core.agent_v3.service import V3AgentTurnService


class V3ResponseQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._sessions_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_service(self) -> V3AgentTurnService:
        return V3AgentTurnService(sessions_dir=self._sessions_dir)

    def test_quality_report_accepts_actionable_dialogue(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "display_message": "I did not start Geant4 because preflight has not passed. Next step: run preflight again.",
            "dialogue_act": "blocked",
            "evidence_used": [{"source": "proposal_critic", "status": "blocked"}],
            "dialogue": {
                "answer_parts": [
                    {"kind": "summary", "text": "I did not start Geant4 because preflight has not passed."},
                    {"kind": "evidence", "items": [{"source": "proposal_critic", "status": "blocked"}]},
                    {"kind": "next_step", "items": [{"text": "Run preflight again", "prefill": "run runtime preflight again for the current payload"}]},
                ],
                "next_suggestions": [
                    {"text": "Run preflight again", "prefill": "run runtime preflight again for the current payload"}
                ]
            },
            "observations": [{"source": "proposal_critic", "status": "blocked"}],
        }

        report = evaluate_v3_response_quality(response)

        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["score"], 0.8)
        self.assertEqual(report["warnings"], [])

    def test_quality_report_flags_trace_leak_and_missing_next_step(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "display_message": "Blocked. See raw trace and state.metadata for details.",
            "dialogue_act": "blocked",
            "evidence_used": [],
            "dialogue": {"next_suggestions": []},
            "observations": [],
        }

        report = evaluate_v3_response_quality(response)

        self.assertFalse(report["ok"])
        self.assertIn("response_leaks_internal_trace_or_metadata", report["warnings"])
        self.assertIn("response_missing_next_step", report["warnings"])
        self.assertIn("response_missing_structured_answer_parts", report["warnings"])

    def test_service_attaches_dialogue_quality_to_turn_and_state(self) -> None:
        service = self._make_service()

        result = service.run_turn(
            {
                "session_id": "quality-service",
                "text": "design a 1 MeV gamma lead shielding setup, do not run",
                "events": 3,
            }
        )

        self.assertIn("dialogue_quality", result)
        self.assertTrue(result["dialogue_quality"]["ok"])
        self.assertTrue(result["dialogue_quality"]["checks"]["has_structured_answer_parts"])
        self.assertEqual(
            result["state"]["metadata"]["last_dialogue_quality"]["schema_version"],
            "geant4_agent_v3_response_quality.v1",
        )

    def test_service_can_apply_optional_grounded_naturalization(self) -> None:
        service = self._make_service()
        naturalized = V3ResponseNaturalizationResult(
            ok=True,
            used_llm=True,
            display_message="I drafted a concise Geant4 design and kept the next step explicit.",
        )

        with patch("core.agent_v3.service.V3ResponseNaturalizer.naturalize", return_value=naturalized):
            result = service.run_turn(
                {
                    "session_id": "quality-naturalized",
                    "text": "design a 1 MeV gamma lead shielding setup, do not run",
                    "events": 3,
                    "llm_config_path": "fake.json",
                    "llm_naturalize_enabled": True,
                }
            )

        self.assertEqual(result["display_message"], naturalized.display_message)
        self.assertEqual(result["answer"]["display_message"], naturalized.display_message)
        self.assertTrue(result["naturalization"]["ok"])
        self.assertTrue(result["dialogue_quality"]["ok"])

    def test_service_keeps_original_message_when_naturalization_is_rejected(self) -> None:
        service = self._make_service()
        rejected = V3ResponseNaturalizationResult(
            ok=False,
            used_llm=True,
            display_message="unsafe replacement",
            fallback_reason="material_conflict",
        )

        with patch("core.agent_v3.service.V3ResponseNaturalizer.naturalize", return_value=rejected):
            result = service.run_turn(
                {
                    "session_id": "quality-naturalized-reject",
                    "text": "design a 1 MeV gamma lead shielding setup, do not run",
                    "events": 3,
                    "llm_config_path": "fake.json",
                    "llm_naturalize_enabled": True,
                }
            )

        self.assertNotEqual(result["display_message"], rejected.display_message)
        self.assertEqual(result["naturalization"]["fallback_reason"], "material_conflict")


if __name__ == "__main__":
    unittest.main()

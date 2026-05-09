from __future__ import annotations

import os
import unittest
from unittest import mock

from core.interpreter import run_interpreter, run_interpreter_v2


class InterpreterRunnerTests(unittest.TestCase):
    def test_runner_builds_prompt_and_parses_response(self) -> None:
        fake_json = """
        {
          "turn_summary": {
            "intent": "set",
            "focus": "mixed",
            "scope": "full_request",
            "user_goal": "Create a copper target and define a gamma point source.",
            "explicit_domains": ["geometry", "source"],
            "uncertain_domains": []
          },
          "geometry_candidate": {
            "kind_candidate": "box",
            "material_candidate": "G4_Cu",
            "dimension_hints": {"size_triplet_mm": [10, 20, 30]},
            "placement_relation": null,
            "confidence": 0.92,
            "ambiguities": [],
            "evidence_spans": [{"text": "10 mm x 20 mm x 30 mm", "role": "dimensions"}]
          },
          "source_candidate": {
            "source_type_candidate": "point",
            "particle_candidate": "gamma",
            "energy_candidate_mev": 1.0,
            "position_mode": "absolute",
            "position_hint": {"position_mm": [0, 0, -20]},
            "direction_mode": "explicit_vector",
            "direction_hint": {"direction_vec": [0, 0, 1]},
            "confidence": 0.95,
            "ambiguities": [],
            "evidence_spans": [{"text": "along +z", "role": "direction"}]
          }
        }
        """

        with mock.patch("core.interpreter.runner.chat", return_value={"response": fake_json}) as fake_chat:
            result = run_interpreter(
                "10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm along +z",
                "phase=geometry source=missing",
                temperature=0.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.parsed.geometry_candidate.kind_candidate, "box")
        self.assertEqual(result.parsed.source_candidate.source_type_candidate, "point")
        called_prompt = fake_chat.call_args.args[0]
        self.assertIn('"turn_summary"', called_prompt)
        self.assertIn("Do not output final config paths.", called_prompt)

    def test_runner_reports_parse_failure(self) -> None:
        with mock.patch("core.interpreter.runner.chat", return_value={"response": "not json"}):
            result = run_interpreter("copper box", "phase=geometry", temperature=0.0)
        self.assertFalse(result.ok)
        self.assertEqual(result.fallback_reason, "json_parse_failed")

    def test_v2_runner_calls_llm_and_validates_path_evidence_contract(self) -> None:
        fake_json = """
        {
          "turn_summary": {
            "intent": "modify",
            "focus": "mixed",
            "user_goal": "change source energy and request a run",
            "requires_confirmation": false
          },
          "candidate_updates": [
            {
              "path": "source.energy_mev",
              "op": "set",
              "value": 10,
              "confidence": 0.9,
              "evidence": [{"text": "10 MeV", "source": "user", "role": "energy"}],
              "requires_confirmation": true
            }
          ],
          "ambiguities": [],
          "unsupported_requests": [],
          "guarded_actions": [
            {
              "action": "run_beam",
              "safety_class": "expensive_runtime",
              "requested": true,
              "reason": "user asked to run"
            }
          ]
        }
        """

        with mock.patch("core.interpreter.runner.chat", return_value={"response": fake_json}) as fake_chat:
            result = run_interpreter_v2(
                "Change source energy to 10 MeV and run 10 events now.",
                "source.energy_mev=1",
                temperature=0.0,
            )

        self.assertTrue(result.ok, result.validation.errors)
        self.assertEqual(result.prompt_profile_id, "interpret_user_turn_en_v2_path_evidence")
        self.assertEqual(result.payload["candidate_updates"][0]["path"], "source.energy_mev")
        called_prompt = fake_chat.call_args.args[0]
        self.assertIn('"candidate_updates"', called_prompt)
        self.assertIn('"guarded_actions"', called_prompt)

    def test_v2_runner_rejects_non_json_llm_output(self) -> None:
        with mock.patch("core.interpreter.runner.chat", return_value={"response": "not json"}):
            result = run_interpreter_v2("Change source energy to 10 MeV.", "", temperature=0.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.fallback_reason, "json_parse_failed")
        self.assertIn("not_json", result.validation.errors)

    def test_v2_runner_rejects_llm_output_that_fails_grounding(self) -> None:
        fake_json = """
        {
          "turn_summary": {
            "intent": "modify",
            "focus": "source",
            "user_goal": "change source energy",
            "requires_confirmation": false
          },
          "candidate_updates": [
            {
              "path": "source.energy_mev",
              "op": "set",
              "value": 99,
              "confidence": 0.9,
              "evidence": [{"text": "10 MeV", "source": "user", "role": "energy"}],
              "requires_confirmation": false
            }
          ],
          "ambiguities": [],
          "unsupported_requests": [],
          "guarded_actions": []
        }
        """

        with mock.patch("core.interpreter.runner.chat", return_value={"response": fake_json}):
            result = run_interpreter_v2("Change source energy to 10 MeV.", "", temperature=0.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.fallback_reason, "validation_failed")
        self.assertIn("ungrounded_numeric_value:candidate_updates[0].value", result.validation.errors)

    @unittest.skipUnless(
        os.environ.get("GEANT4_INTERPRETER_V2_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}
        and os.environ.get("GEANT4_LLM_CONFIG"),
        "live interpreter v2 LLM smoke is opt-in via GEANT4_INTERPRETER_V2_LIVE=1 and GEANT4_LLM_CONFIG",
    )
    def test_live_llm_v2_opt_in_outputs_valid_path_evidence_contract(self) -> None:
        result = run_interpreter_v2(
            "Change source energy to 10 MeV and run 10 events now.",
            "source.energy_mev=1",
            config_path=os.environ["GEANT4_LLM_CONFIG"],
            temperature=0.0,
        )

        self.assertTrue(result.ok, result.validation.errors)
        self.assertTrue(result.payload.get("candidate_updates"))
        self.assertTrue(result.payload.get("guarded_actions"))


if __name__ == "__main__":
    unittest.main()

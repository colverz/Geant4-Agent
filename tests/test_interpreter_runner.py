from __future__ import annotations

import unittest
from unittest import mock

from core.interpreter import run_interpreter


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


if __name__ == "__main__":
    unittest.main()

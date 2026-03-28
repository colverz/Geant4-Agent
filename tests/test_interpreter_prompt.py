from __future__ import annotations

import unittest

from core.interpreter import build_interpreter_prompt, detect_prompt_language, parse_interpreter_response


class InterpreterPromptTests(unittest.TestCase):
    def test_prompt_mentions_candidate_only_boundary(self) -> None:
        prompt = build_interpreter_prompt(
            "10 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm along +z",
            "phase=geometry source=missing",
        )
        self.assertIn("Do not output final config paths.", prompt)
        self.assertIn('"turn_summary"', prompt)
        self.assertIn('"geometry_candidate"', prompt)
        self.assertIn('"source_candidate"', prompt)
        self.assertIn("Geometry and source are interpreted candidates only", prompt)

    def test_prompt_switches_to_chinese_template(self) -> None:
        prompt = build_interpreter_prompt(
            "做一个铜靶，10 mm 见方，gamma点源1 MeV，位于(0,0,-20) mm，朝+z方向。",
            "phase=geometry source=missing",
        )
        self.assertIn("请解释这轮 Geant4 配置请求真正表达的意思", prompt)
        self.assertIn("不要输出最终 config path", prompt)
        self.assertIn('用户: "10 mm x 20 mm x 30 mm 铜盒靶"', prompt)

    def test_language_detector_distinguishes_en_zh_and_mixed(self) -> None:
        self.assertEqual(detect_prompt_language("copper box target"), "en")
        self.assertEqual(detect_prompt_language("铜盒靶"), "zh")
        self.assertEqual(detect_prompt_language("10 mm copper box 铜靶"), "mixed")

    def test_parser_builds_candidate_objects(self) -> None:
        raw = """
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
        result = parse_interpreter_response(raw)
        self.assertTrue(result.ok)
        self.assertEqual(result.turn_summary.intent, "set")
        self.assertEqual(result.geometry_candidate.kind_candidate, "box")
        self.assertEqual(result.geometry_candidate.dimension_hints["size_triplet_mm"], [10, 20, 30])
        self.assertEqual(result.source_candidate.source_type_candidate, "point")
        self.assertEqual(result.source_candidate.direction_hint["direction_vec"], [0, 0, 1])

    def test_parser_reports_json_failure(self) -> None:
        result = parse_interpreter_response("not json")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "json_parse_failed")


if __name__ == "__main__":
    unittest.main()

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
        self.assertIn("Bind geometry_candidate to the target/object being built.", prompt)

    def test_prompt_switches_to_chinese_template(self) -> None:
        prompt = build_interpreter_prompt(
            "\u505a\u4e00\u4e2a\u94dc\u9776\uff0c10 mm \u89c1\u65b9\uff0cgamma\u70b9\u6e901 MeV\uff0c\u4f4d\u4e8e(0,0,-20) mm\uff0c\u671d+z\u65b9\u5411\u3002",
            "phase=geometry source=missing",
        )
        self.assertIn("\u8bf7\u89e3\u91ca\u8fd9\u8f6e Geant4 \u914d\u7f6e\u8bf7\u6c42\u771f\u6b63\u8868\u8fbe\u7684\u610f\u601d", prompt)
        self.assertIn("\u4e0d\u8981\u8f93\u51fa\u6700\u7ec8 config path", prompt)
        self.assertIn("\u201c10 mm \u89c1\u65b9\u9776\u201d", prompt)
        self.assertIn("\u5982\u679c\u540c\u4e00\u53e5\u91cc\u65e2\u6709\u9776\u53c8\u6709\u6e90", prompt)
        self.assertIn("\u201c\u94dc\u9776\u201d", prompt)
        self.assertIn("geometry_candidate.material_candidate = \"G4_Cu\"", prompt)

    def test_prompt_mentions_material_plus_target_binding_rule(self) -> None:
        prompt = build_interpreter_prompt(
            "copper target with gamma source",
            "phase=geometry source=missing",
        )
        self.assertIn("treat that material as geometry_candidate.material_candidate", prompt)
        self.assertIn('User: "copper target"', prompt)

    def test_language_detector_distinguishes_en_zh_and_mixed(self) -> None:
        self.assertEqual(detect_prompt_language("copper box target"), "en")
        self.assertEqual(detect_prompt_language("\u94dc\u76d2\u9776"), "zh")
        self.assertEqual(detect_prompt_language("10 mm copper box \u94dc\u9776"), "mixed")

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

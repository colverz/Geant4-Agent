from __future__ import annotations

import unittest

from ui.web.server import solve, step


class SmokeNoOllamaTest(unittest.TestCase):
    def test_solve_without_ollama(self) -> None:
        out = solve(
            {
                "text": "ring of 12 modules, radius 40 mm, module 8x10x2 mm, clearance 1 mm",
                "min_confidence": 0.6,
                "normalize_input": False,
                "llm_fill_missing": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", out)
        self.assertIn("structure", out)
        self.assertIn("synthesis", out)

    def test_step_without_ollama(self) -> None:
        out = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", out)
        self.assertIn("missing_fields", out)
        self.assertIn("assistant_message", out)
        self.assertIn("phase", out)
        self.assertIn("phase_title", out)
        self.assertIn("asked_fields", out)
        self.assertIn("asked_fields_friendly", out)
        self.assertFalse(out.get("is_complete", False))

    def test_schema_closure_progress(self) -> None:
        first = step(
            {
                "text": "ring of 12 modules, radius 40 mm, module 8x10x2 mm, clearance 1 mm, material G4_Cu, gamma source, FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", first)
        sid = first["session_id"]
        m1 = set(first.get("missing_fields", []))
        self.assertIn(first.get("phase"), {"source_kinematics", "geometry_params", "materials", "source_core", "physics", "output", "geometry_core"})
        self.assertIn("source.energy_MeV", m1)
        self.assertIn("source.position", m1)
        self.assertIn("source.direction", m1)

        second = step(
            {
                "session_id": sid,
                "text": "energy 2 MeV, position (0,0,-100), direction (0,0,1)",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", second)
        m2 = set(second.get("missing_fields", []))
        self.assertNotIn("source.energy_MeV", m2)
        self.assertNotIn("source.position", m2)
        self.assertNotIn("source.direction", m2)
        self.assertIn("delta_paths", second)

    def test_non_english_requires_normalization(self) -> None:
        out = step(
            {
                "text": "我想要一个1m的铜立方体和gamma源",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
            }
        )
        self.assertTrue(out.get("requires_llm_normalization", False))
        self.assertEqual(out.get("phase"), "normalization")


if __name__ == "__main__":
    unittest.main()

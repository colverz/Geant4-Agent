from __future__ import annotations

import unittest

from core.orchestrator.session_manager import process_turn, reset_session


class V2RealPromptRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session("v2-real-prompt")

    def _run(self, text: str) -> dict:
        return process_turn(
            {
                "session_id": "v2-real-prompt",
                "text": text,
                "llm_router": False,
                "llm_question": False,
                "normalize_input": True,
                "geometry_pipeline": "v2",
                "source_pipeline": "v2",
                "enable_compare": False,
            },
            ollama_config_path="",
        )

    def test_complete_box_point_prompt_is_runtime_ready_in_v2(self) -> None:
        out = self._run(
            "10 mm x 20 mm x 30 mm copper box target; "
            "gamma point source 1 MeV at (0,0,-20) mm along +z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["particle"], "gamma")
        self.assertTrue(out["slot_debug"]["geometry_v2"]["runtime_ready"])
        self.assertTrue(out["slot_debug"]["source_v2"]["runtime_ready"])

    def test_cylinder_beam_prompt_keeps_geometry_and_asks_for_direction(self) -> None:
        out = self._run(
            "copper cylinder radius 15 mm half length 40 mm; "
            "gamma beam 5 MeV from (0,0,-250) mm; "
            "physics Shielding; output json."
        )
        self.assertFalse(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_tubs")
        self.assertNotEqual(out["config"]["source"].get("type"), "beam")
        self.assertIn("source.direction", out["missing_fields"])
        self.assertIn("source.direction", out["asked_fields"])
        self.assertTrue(out["slot_debug"]["geometry_v2"]["runtime_ready"])
        self.assertFalse(out["slot_debug"]["source_v2"]["runtime_ready"])

    def test_box_side_shorthand_prompt_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm copper box target; "
            "gamma point source 1 MeV at (0,0,-20) mm along +z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["geometry"]["params"]["module_x"], 10.0)
        self.assertEqual(out["config"]["geometry"]["params"]["module_y"], 10.0)
        self.assertEqual(out["config"]["geometry"]["params"]["module_z"], 10.0)

    def test_relative_target_center_source_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma point source 1 MeV 20 mm outside the target center along -z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["position"]["value"], [0.0, 0.0, -20.0])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_cylinder_shorthand_prompt_is_understood_by_v2(self) -> None:
        out = self._run(
            "radius 5 mm, height 20 mm copper cylinder; "
            "gamma point source 1 MeV at (0,0,-20) mm along +z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_tubs")
        self.assertEqual(out["config"]["geometry"]["params"]["child_rmax"], 5.0)
        self.assertEqual(out["config"]["geometry"]["params"]["child_hz"], 10.0)

    def test_in_front_of_target_source_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma point source 1 MeV 5 mm in front of the target along -z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["position"]["value"], [0.0, 0.0, -5.0])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_toward_target_center_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma point source 1 MeV at (0,0,-20) mm toward target center; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_cylinder_diameter_shorthand_is_understood_by_v2(self) -> None:
        out = self._run(
            "20 mm diameter copper cylinder with half length 40 mm; "
            "gamma point source 1 MeV at (0,0,-20) mm along +z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_tubs")
        self.assertEqual(out["config"]["geometry"]["params"]["child_rmax"], 10.0)
        self.assertEqual(out["config"]["geometry"]["params"]["child_hz"], 40.0)

    def test_target_surface_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma point source 1 MeV 5 mm from the target surface along -z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["position"]["value"], [0.0, 0.0, -5.0])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_toward_target_face_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma point source 1 MeV at (0,0,-20) mm toward target face along -z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_thick_slab_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "2 mm thick lead slab; "
            "gamma point source 1 MeV at (0,0,-20) mm along +z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["geometry"]["params"]["module_z"], 2.0)

    def test_target_surface_normal_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma beam 1 MeV at (0,0,-20) mm toward target surface normal along -z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_chinese_square_target_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm 见方铜靶；"
            "gamma 点源 1 MeV，位于 (0,0,-20) mm，沿 +z 方向；"
            "物理列表 FTFP_BERT；输出 json。"
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["geometry"]["params"]["module_x"], 10.0)
        self.assertEqual(out["config"]["geometry"]["params"]["module_y"], 10.0)
        self.assertEqual(out["config"]["geometry"]["params"]["module_z"], 10.0)

    def test_chinese_thick_slab_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "厚 2 mm 的铅板；"
            "gamma 点源 1 MeV，位于 (0,0,-20) mm，沿 +z 方向；"
            "物理列表 FTFP_BERT；输出 json。"
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["geometry"]["params"]["module_z"], 2.0)

    def test_chinese_front_of_target_phrase_is_understood_by_v2(self) -> None:
        out = self._run(
            "10 mm x 10 mm x 10 mm copper box target; "
            "gamma point source 1 MeV 距靶面前方 5 mm，朝靶面法线方向，沿 -z； "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["source"]["position"]["value"], [0.0, 0.0, -5.0])
        self.assertEqual(out["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

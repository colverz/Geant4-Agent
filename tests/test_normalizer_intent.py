from __future__ import annotations

import unittest
from unittest import mock

from core.orchestrator.types import Intent
from nlu.llm.normalizer import infer_user_turn_controls, normalize_user_turn


class NormalizerIntentTest(unittest.TestCase):
    def test_payload_with_question_mark_stays_set(self) -> None:
        controls = infer_user_turn_controls(
            "Gamma point source? energy 1 MeV, position (0,0,0), direction +z."
        )
        self.assertEqual(controls["intent"], Intent.SET)
        self.assertIn("source.energy", controls["target_paths"])
        self.assertIn("source.position", controls["target_paths"])
        self.assertIn("source.direction", controls["target_paths"])

    def test_semantic_question_is_question_intent(self) -> None:
        controls = infer_user_turn_controls("Why choose this physics list?")
        self.assertEqual(controls["intent"], Intent.QUESTION)

    def test_chinese_payload_targets_geometry_family_and_dimensions(self) -> None:
        controls = infer_user_turn_controls(
            "\u6211\u8981\u4e00\u4e2a\u94dc\u7acb\u65b9\u4f53\uff0c\u5c3a\u5bf8 1m x 1m x 1m\uff0c\u70b9\u6e90 gamma\uff0c\u80fd\u91cf 1 MeV\u3002"
        )
        self.assertEqual(controls["intent"], Intent.SET)
        self.assertIn("geometry.structure", controls["target_paths"])
        self.assertIn("geometry.params.module_x", controls["target_paths"])
        self.assertIn("geometry.params.module_y", controls["target_paths"])
        self.assertIn("geometry.params.module_z", controls["target_paths"])
        self.assertIn("materials.selected_materials", controls["target_paths"])

    def test_normalize_user_turn_can_skip_llm_call(self) -> None:
        with mock.patch("nlu.llm.normalizer.chat", side_effect=RuntimeError("should not call")) as mocked_chat:
            out = normalize_user_turn(
                "set source energy 1 MeV",
                context_summary="",
                config_path="nlu/llm_support/configs/ollama_config.json",
                enable_llm=False,
            )
        mocked_chat.assert_not_called()
        self.assertEqual(out["normalized_text"], "set source energy 1 MeV")
        self.assertEqual(out["intent"], Intent.SET)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import unittest
from unittest import mock

from planner.agent import ask_missing, naturalize_response


class DialogueNaturalizationTest(unittest.TestCase):
    def test_ask_missing_uses_context_for_prompt(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Could you share the source energy in MeV?"}) as mocked_chat:
            out = ask_missing(
                ["source.energy"],
                lang="en",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
                recent_user_text="I already set the geometry.",
                confirmed_items=["geometry structure", "material"],
            )
        self.assertEqual(out, "Could you share the source energy in MeV?")
        prompt_text = mocked_chat.call_args.args[0]
        self.assertIn("Latest user input: I already set the geometry.", prompt_text)
        self.assertIn("Confirmed context: geometry structure, material", prompt_text)

    def test_ask_missing_rejects_internal_field_names(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Please provide: source.type"}):
            out = ask_missing(
                ["source.type"],
                lang="en",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "I still need the source type, for example point, beam, or isotropic.")

    def test_ask_missing_uses_family_aware_fallback_for_graph_paths(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Please provide: geometry.ask.ring.radius"}):
            out = ask_missing(
                ["geometry.ask.ring.radius"],
                lang="en",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "To complete the ring geometry, I still need the ring radius.")

    def test_ask_missing_uses_source_fallback_for_internal_field_response(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Please provide: source.energy"}):
            out = ask_missing(
                ["source.energy"],
                lang="en",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "I still need the source energy in MeV.")

    def test_ask_missing_uses_material_fallback_for_internal_field_response(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Please provide: materials.volume_material_map"}):
            out = ask_missing(
                ["materials.volume_material_map"],
                lang="en",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "I still need the material assignment, meaning which volume uses which material.")

    def test_ask_missing_uses_zh_fallback_for_internal_field_response(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Please provide: source.energy"}):
            out = ask_missing(
                ["source.energy"],
                lang="zh",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u80fd\u91cf\uff0c\u6309 MeV \u7ed9\u51fa\u5373\u53ef\u3002")

    def test_naturalize_response_uses_llm_reply(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Naturalized response."}) as mocked_chat:
            out = naturalize_response(
                "Base response.",
                lang="en",
                action="answer_status",
                updated_paths=["source.energy"],
                missing_fields=["output.path"],
                asked_fields=[],
                overwrite_preview=[],
                dialogue_summary={"updated_fields": ["source energy"]},
                raw_dialogue=[{"role": "user", "content": "set source energy to 1 MeV"}],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "Naturalized response.")
        mocked_chat.assert_called_once()

    def test_naturalize_response_falls_back_on_error(self) -> None:
        with mock.patch("planner.agent.chat", side_effect=RuntimeError("network down")):
            out = naturalize_response(
                "Base response.",
                lang="en",
                action="answer_status",
                updated_paths=[],
                missing_fields=[],
                asked_fields=[],
                overwrite_preview=[],
                dialogue_summary={},
                raw_dialogue=[],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "Base response.")

    def test_naturalize_response_prompt_contains_overwrite_context(self) -> None:
        with mock.patch(
            "planner.agent.chat",
            return_value={
                "response": "Please confirm the overwrite, or say keep existing to keep the current value."
            },
        ) as mocked_chat:
            out = naturalize_response(
                "This would overwrite an already confirmed value.",
                lang="en",
                action="confirm_overwrite",
                updated_paths=[],
                missing_fields=["output.path"],
                asked_fields=[],
                overwrite_preview=[{"path": "materials.selected_materials", "old": "G4_Cu", "new": "G4_Al"}],
                dialogue_summary={"pending_fields": ["output path"]},
                raw_dialogue=[{"role": "user", "content": "change material to aluminum"}],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(
            out,
            "Please confirm the overwrite, or say keep existing to keep the current value.",
        )
        prompt_text = mocked_chat.call_args.args[0]
        self.assertIn('"action": "confirm_overwrite"', prompt_text)
        self.assertIn('"overwrite_preview"', prompt_text)

    def test_ask_missing_rejects_language_mismatch(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Please share the source energy in MeV."}):
            out = ask_missing(
                ["source.energy"],
                lang="zh",
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "\u8fd8\u9700\u8981\u786e\u8ba4\u6e90\u80fd\u91cf\uff0c\u6309 MeV \u7ed9\u51fa\u5373\u53ef\u3002")

    def test_naturalize_response_rejects_internal_field_output(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "Updated source.energy and output.path."}):
            out = naturalize_response(
                "Base response.",
                lang="en",
                action="answer_status",
                updated_paths=["source.energy"],
                missing_fields=[],
                asked_fields=[],
                overwrite_preview=[],
                dialogue_summary={},
                raw_dialogue=[],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "Base response.")

    def test_naturalize_response_rejects_internal_geometry_terms(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "I updated geometry parameter module_x."}):
            out = naturalize_response(
                "Base response.",
                lang="en",
                action="confirm_update",
                updated_paths=["geometry.params.module_x"],
                missing_fields=[],
                asked_fields=[],
                overwrite_preview=[],
                dialogue_summary={},
                raw_dialogue=[],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "Base response.")

    def test_naturalize_response_rejects_language_mismatch(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "\u6211\u5df2\u7ecf\u66f4\u65b0\u597d\u4e86\u3002"}):
            out = naturalize_response(
                "Base response.",
                lang="en",
                action="answer_status",
                updated_paths=[],
                missing_fields=[],
                asked_fields=[],
                overwrite_preview=[],
                dialogue_summary={},
                raw_dialogue=[],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertEqual(out, "Base response.")

    def test_naturalize_response_rejects_invalid_confirm_overwrite_copy(self) -> None:
        with mock.patch("planner.agent.chat", return_value={"response": "I can change the value for you."}):
            out = naturalize_response(
                "This would overwrite an already confirmed value. Pending change: output format: csv -> root. Reply 'confirm' to apply it, or say 'keep existing' to keep the current value.",
                lang="en",
                action="confirm_overwrite",
                updated_paths=[],
                missing_fields=[],
                asked_fields=[],
                overwrite_preview=[{"path": "output.format", "old": "csv", "new": "root"}],
                dialogue_summary={},
                raw_dialogue=[],
                ollama_config="nlu/llm_support/configs/ollama_config.json",
                temperature=1.0,
            )
        self.assertIn("Reply 'confirm' to apply it", out)


if __name__ == "__main__":
    unittest.main()


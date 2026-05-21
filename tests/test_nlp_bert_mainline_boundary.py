from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import nlu.bert.extractor as legacy_extractor
import nlu.runtime_extractor as runtime_extractor
from nlu.runtime_semantic import extract_runtime_semantic_frame
from ui.web.strict_api import handle_strict_step


class NlpBertMainlineBoundaryTest(unittest.TestCase):
    def test_legacy_bert_extractor_path_is_thin_runtime_wrapper(self) -> None:
        self.assertIs(
            legacy_extractor.extract_candidates_from_normalized_text,
            runtime_extractor.extract_candidates_from_normalized_text,
        )

    def test_active_code_does_not_import_legacy_bert_extractor(self) -> None:
        roots = [Path("core"), Path("tools"), Path("ui"), Path("nlu")]
        offenders: list[str] = []
        for root in roots:
            for path in root.rglob("*.py"):
                normalized = path.as_posix()
                if normalized == "nlu/bert/extractor.py":
                    continue
                text = path.read_text(encoding="utf-8")
                if "nlu.bert.extractor" in text:
                    offenders.append(normalized)
        self.assertEqual(offenders, [])

    def test_runtime_semantic_can_bypass_nlp_bert_model_prior(self) -> None:
        with (
            patch("nlu.runtime_semantic.extract_params") as extract_params,
            patch("nlu.runtime_semantic._pick_structure_model") as pick_structure_model,
            patch("nlu.runtime_semantic.predict_structure") as predict_structure,
        ):
            frame, debug = extract_runtime_semantic_frame(
                "10 mm x 20 mm x 30 mm copper box with 1 MeV gamma source",
                enable_nlp_bert_prior=False,
                apply_autofix=True,
            )

        extract_params.assert_not_called()
        pick_structure_model.assert_not_called()
        predict_structure.assert_not_called()
        self.assertEqual(debug["inference_backend"], "runtime_semantic_rules")
        self.assertFalse(debug["nlp_bert_model_prior_enabled"])
        self.assertIn(frame.geometry.structure, {"single_box", "box", ""})

    def test_strict_web_defaults_v2_only_for_llm_mainline(self) -> None:
        captured_payloads: list[dict] = []

        def fake_process_turn(*, payload: dict, **_: object) -> dict:
            captured_payloads.append(dict(payload))
            return {"ok": True, "payload": dict(payload)}

        with (
            patch(
                "ui.web.strict_api._load_session_manager",
                return_value=(lambda _: [], fake_process_turn, lambda _: None, lambda *_: {}),
            ),
            patch("ui.web.runtime_state.get_ollama_config_path", return_value=""),
        ):
            llm_result = handle_strict_step({"text": "configure a copper box", "llm_router": True})
            fallback_result = handle_strict_step(
                {
                    "text": "configure a copper box",
                    "llm_router": False,
                    "normalize_input": False,
                }
            )

        self.assertEqual(llm_result["payload"]["geometry_pipeline"], "v2")
        self.assertEqual(llm_result["payload"]["source_pipeline"], "v2")
        self.assertNotIn("geometry_pipeline", fallback_result["payload"])
        self.assertNotIn("source_pipeline", fallback_result["payload"])
        self.assertEqual(len(captured_payloads), 2)


if __name__ == "__main__":
    unittest.main()

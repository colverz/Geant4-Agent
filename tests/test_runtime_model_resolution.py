from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nlu.runtime_components import infer as runtime_infer
from nlu.runtime_components import model_preflight


class RuntimeModelResolutionTest(unittest.TestCase):
    def test_require_local_model_dir_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "ner"
            with self.assertRaises(RuntimeError) as ctx:
                runtime_infer._require_local_model_dir(missing, label="NER")
        self.assertIn("Missing local NER model", str(ctx.exception))

    def test_default_ner_model_raises_when_default_directory_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(runtime_infer, "MODELS_DIR", Path(tmp)):
                with self.assertRaises(RuntimeError) as ctx:
                    runtime_infer._default_ner_model()
        self.assertIn("Train it locally first", str(ctx.exception))

    def test_validate_model_dir_reports_missing_tokenizer_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "m"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            report = model_preflight.validate_model_dir(model_dir, label="test")
        self.assertFalse(report["ok"])
        self.assertIn("tokenizer_assets", report["missing_files"])

    def test_validate_model_dir_accepts_config_and_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "m"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
            report = model_preflight.validate_model_dir(model_dir, label="test")
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()

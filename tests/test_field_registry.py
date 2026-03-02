from __future__ import annotations

import unittest

from core.config.field_registry import canonical_field_path, friendly_labels, missing_field_question


class FieldRegistryTest(unittest.TestCase):
    def test_canonicalizes_legacy_paths(self) -> None:
        self.assertEqual(canonical_field_path("source.energy_MeV"), "source.energy")
        self.assertEqual(canonical_field_path("physics_list.name"), "physics.physics_list")

    def test_friendly_labels_handle_legacy_aliases(self) -> None:
        labels = friendly_labels(["source.energy_MeV", "physics_list.name"], "en")
        self.assertEqual(labels, ["source energy", "physics list"])

    def test_missing_field_question_uses_shared_registry(self) -> None:
        self.assertEqual(
            missing_field_question("physics_list.name", "zh"),
            "请提供物理列表名称（例如 FTFP_BERT 或 QBBC）。",
        )


if __name__ == "__main__":
    unittest.main()

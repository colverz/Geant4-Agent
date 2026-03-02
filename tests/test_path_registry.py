from __future__ import annotations

import unittest

from core.config.path_registry import canonical_field_path, field_matches_pattern


class PathRegistryTest(unittest.TestCase):
    def test_canonicalizes_known_aliases(self) -> None:
        self.assertEqual(canonical_field_path("source.energy_MeV"), "source.energy")
        self.assertEqual(canonical_field_path("physics_list.name"), "physics.physics_list")

    def test_matches_legacy_and_canonical_patterns(self) -> None:
        self.assertTrue(field_matches_pattern("source.energy", "source.energy_MeV"))
        self.assertTrue(field_matches_pattern("physics.physics_list", "physics_list."))
        self.assertFalse(field_matches_pattern("source.direction", "source.energy_MeV"))


if __name__ == "__main__":
    unittest.main()

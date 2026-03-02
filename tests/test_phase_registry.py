from __future__ import annotations

import unittest

from core.config.phase_registry import phase_title, select_phase_fields


class PhaseRegistryTest(unittest.TestCase):
    def test_select_phase_fields_handles_legacy_aliases(self) -> None:
        phase, fields = select_phase_fields(["source.energy_MeV", "source.direction"])
        self.assertEqual(phase, "source_kinematics")
        self.assertEqual(fields, ["source.energy_MeV", "source.direction"])

    def test_phase_title_covers_strict_and_legacy_phases(self) -> None:
        self.assertEqual(phase_title("physics", "en"), "Physics")
        self.assertEqual(phase_title("source_kinematics", "zh"), "\u6e90\u9879\u8fd0\u52a8\u5b66")


if __name__ == "__main__":
    unittest.main()

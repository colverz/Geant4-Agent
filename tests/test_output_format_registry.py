from __future__ import annotations

import unittest

from core.config.output_format_registry import (
    accepted_output_formats,
    canonical_output_format,
    official_output_formats,
    project_output_extensions,
)


class OutputFormatRegistryTest(unittest.TestCase):
    def test_registry_exposes_official_and_project_formats(self) -> None:
        self.assertEqual(official_output_formats(), ("csv", "hdf5", "root", "xml"))
        self.assertEqual(project_output_extensions(), ("json",))
        self.assertEqual(
            accepted_output_formats(),
            ("csv", "hdf5", "json", "root", "xml"),
        )

    def test_registry_canonicalizes_h5_alias(self) -> None:
        self.assertEqual(canonical_output_format("h5"), "hdf5")
        self.assertEqual(canonical_output_format("ROOT"), "root")
        self.assertIsNone(canonical_output_format("ascii"))


if __name__ == "__main__":
    unittest.main()

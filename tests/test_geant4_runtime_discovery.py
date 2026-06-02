from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcp.geant4.runtime_discovery import DEFAULT_LOCAL_RUNTIME, discover_local_geant4_runtime


class Geant4RuntimeDiscoveryTest(unittest.TestCase):
    def test_discovery_reports_not_found_for_empty_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = discover_local_geant4_runtime(repo_root=Path(tmpdir))

        self.assertFalse(report.found)
        self.assertEqual(report.source, "not_found")
        self.assertEqual(report.command, [])
        self.assertEqual(report.env(), {})

    def test_discovery_finds_repo_local_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exe = root / DEFAULT_LOCAL_RUNTIME
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_text("fake exe", encoding="utf-8")

            report = discover_local_geant4_runtime(repo_root=root, geant4_root="X:/Geant4")

        self.assertTrue(report.found)
        self.assertEqual(report.source, "repo_default_local_app")
        self.assertEqual(report.command, [str(exe.resolve())])
        self.assertEqual(report.env()["GEANT4_ROOT"], "X:/Geant4")
        self.assertIn("GEANT4_RUNTIME_COMMAND_JSON", report.env())


if __name__ == "__main__":
    unittest.main()

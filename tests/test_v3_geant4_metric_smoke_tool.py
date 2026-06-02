from __future__ import annotations

import unittest
from unittest.mock import patch

from mcp.geant4.adapter import InMemoryGeant4Adapter
from tools.run_v3_geant4_metric_smoke import _metric_cases, run_v3_geant4_metric_smoke


class V3Geant4MetricSmokeToolTest(unittest.TestCase):
    def test_metric_cases_cover_edep_detector_and_plane_scoring(self) -> None:
        cases = _metric_cases(7)
        case_ids = {case["id"] for case in cases}

        self.assertIn("target_edep_copper_1mev", case_ids)
        self.assertIn("detector_crossing_vacuum_gamma", case_ids)
        self.assertIn("plane_crossing_vacuum_gamma", case_ids)
        detector_case = next(case for case in cases if case["id"] == "detector_crossing_vacuum_gamma")
        plane_case = next(case for case in cases if case["id"] == "plane_crossing_vacuum_gamma")
        self.assertTrue(detector_case["config"]["scoring"]["detector_crossings"])
        self.assertTrue(plane_case["config"]["scoring"]["plane_crossings"])
        self.assertEqual(detector_case["config"]["run"]["events"], 7)

    def test_metric_smoke_can_skip_without_local_process_runtime(self) -> None:
        with patch(
            "tools.run_v3_geant4_metric_smoke.build_geant4_adapter_from_env",
            return_value=InMemoryGeant4Adapter(),
        ):
            result = run_v3_geant4_metric_smoke(auto_discover_runtime=False, require_runtime=False)

        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skip_reason"], "local_process_runtime_required")


if __name__ == "__main__":
    unittest.main()

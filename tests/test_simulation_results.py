from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from core.simulation import load_simulation_result, simulation_result_from_dict


class SimulationResultTest(unittest.TestCase):
    def test_simulation_result_from_dict_keeps_scoring_fields(self) -> None:
        result = simulation_result_from_dict(
            {
                "run_ok": True,
                "events_requested": 4,
                "events_completed": 4,
                "geometry_structure": "single_box",
                "material": "G4_Cu",
                "particle": "gamma",
                "source_type": "point",
                "source_position_mm": [0, 0, -20],
                "source_direction": [0, 0, 1],
                "physics_list": "FTFP_BERT",
                "events": 4,
                "mode": "batch",
                "scoring": {
                    "target_edep_enabled": True,
                    "target_edep_total_mev": 2.7,
                    "target_edep_mean_mev_per_event": 0.675,
                    "target_hit_events": 3,
                    "target_step_count": 18,
                    "target_track_entries": 4,
                },
            }
        )
        self.assertTrue(result.run_ok)
        self.assertEqual(result.geometry_structure, "single_box")
        self.assertEqual(result.source_position_mm, (0.0, 0.0, -20.0))
        self.assertEqual(result.scoring.target_hit_events, 3)
        self.assertEqual(result.scoring.target_step_count, 18)
        self.assertEqual(result.scoring.volume_stats["Target"]["track_entries"], 4)

    def test_load_simulation_result_reads_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "run_summary.json"
            summary_path.write_text(
                """{
  "run_ok": true,
  "events_requested": 2,
  "events_completed": 2,
  "geometry_structure": "single_tubs",
  "material": "G4_W",
  "particle": "proton",
  "source_type": "beam",
  "source_position_mm": [0, 0, -250],
  "source_direction": [0, 0, 1],
  "physics_list": "QGSP_BERT",
  "events": 2,
  "mode": "batch",
  "scoring": {
    "target_edep_enabled": true,
    "target_edep_total_mev": 1.25,
    "target_edep_mean_mev_per_event": 0.625,
    "target_hit_events": 2,
    "target_step_count": 9,
    "target_track_entries": 2,
    "volume_stats": {
      "Target": {
        "edep_total_mev": 1.25,
        "edep_mean_mev_per_event": 0.625,
        "hit_events": 2,
        "step_count": 9,
        "track_entries": 2
      }
    }
  }
}""",
                encoding="utf-8",
            )
            result = load_simulation_result(summary_path)
        self.assertEqual(result.geometry_structure, "single_tubs")
        self.assertEqual(result.source_type, "beam")
        self.assertAlmostEqual(result.scoring.target_edep_total_mev, 1.25)
        self.assertEqual(result.scoring.volume_stats["Target"]["step_count"], 9)

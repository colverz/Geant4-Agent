from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from core.simulation import (
    SIMULATION_RESULT_SCHEMA_VERSION,
    derive_role_stats,
    load_simulation_result,
    simulation_result_from_dict,
)


class SimulationResultTest(unittest.TestCase):
    def test_simulation_result_from_dict_keeps_scoring_fields(self) -> None:
        result = simulation_result_from_dict(
            {
                "run_ok": True,
                "events_requested": 4,
                "events_completed": 4,
                "schema_version": SIMULATION_RESULT_SCHEMA_VERSION,
                "geometry_structure": "single_box",
                "material": "G4_Cu",
                "particle": "gamma",
                "source_type": "point",
                "payload_sha256": "abc123",
                "geant4_version": "geant4-test",
                "source_position_mm": [0, 0, -20],
                "source_direction": [0, 0, 1],
                "physics_list": "FTFP_BERT",
                "events": 4,
                "mode": "batch",
                "detector": {
                    "enabled": True,
                    "volume_name": "Detector",
                    "material": "G4_Si",
                    "position_mm": [0, 0, 100],
                    "size_mm": [20, 20, 2],
                },
                "scoring": {
                    "target_edep_enabled": True,
                    "target_edep_total_mev": 2.7,
                    "target_edep_mean_mev_per_event": 0.675,
                    "target_hit_events": 3,
                    "target_step_count": 18,
                    "target_track_entries": 4,
                    "role_stats": {
                        "target": {
                            "edep_total_mev": 2.7,
                            "edep_mean_mev_per_event": 0.675,
                            "hit_events": 3,
                            "step_count": 18,
                            "track_entries": 4
                        }
                    }
                },
            }
        )
        self.assertTrue(result.run_ok)
        self.assertEqual(result.schema_version, SIMULATION_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.geometry_structure, "single_box")
        self.assertEqual(result.payload_sha256, "abc123")
        self.assertEqual(result.geant4_version, "geant4-test")
        self.assertEqual(result.source_position_mm, (0.0, 0.0, -20.0))
        self.assertTrue(result.detector.enabled)
        self.assertEqual(result.detector.volume_name, "Detector")
        self.assertEqual(result.scoring.target_hit_events, 3)
        self.assertEqual(result.scoring.target_step_count, 18)
        self.assertEqual(result.scoring.volume_stats["Target"]["track_entries"], 4)
        self.assertEqual(result.scoring.role_stats["target"]["track_entries"], 4)

    def test_load_simulation_result_reads_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "run_summary.json"
            summary_path.write_text(
                """{
  "run_ok": true,
  "events_requested": 2,
  "events_completed": 2,
  "schema_version": "2026-04-12.v1",
  "geometry_structure": "single_tubs",
  "material": "G4_W",
  "particle": "proton",
  "source_type": "beam",
  "payload_sha256": "def456",
  "geant4_version": "geant4-test",
  "source_position_mm": [0, 0, -250],
  "source_direction": [0, 0, 1],
  "physics_list": "QGSP_BERT",
  "events": 2,
  "mode": "batch",
  "detector": {
    "enabled": true,
    "volume_name": "Detector",
    "material": "G4_Si",
    "position_mm": [0, 0, 50],
    "size_mm": [15, 15, 2]
  },
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
    },
    "role_stats": {
      "target": {
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
        self.assertEqual(result.schema_version, SIMULATION_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.payload_sha256, "def456")
        self.assertEqual(result.source_type, "beam")
        self.assertTrue(result.detector.enabled)
        self.assertAlmostEqual(result.scoring.target_edep_total_mev, 1.25)
        self.assertEqual(result.scoring.volume_stats["Target"]["step_count"], 9)
        self.assertEqual(result.scoring.role_stats["target"]["step_count"], 9)

    def test_derive_role_stats_aggregates_named_volumes(self) -> None:
        role_stats = derive_role_stats(
            {
                "target_core": {
                    "edep_total_mev": 1.0,
                    "edep_mean_mev_per_event": 0.5,
                    "hit_events": 2,
                    "step_count": 10,
                    "track_entries": 2,
                },
                "target_shell": {
                    "edep_total_mev": 0.5,
                    "edep_mean_mev_per_event": 0.25,
                    "hit_events": 1,
                    "step_count": 6,
                    "track_entries": 1,
                },
            },
            {"target": ["target_core", "target_shell"]},
        )
        self.assertAlmostEqual(role_stats["target"]["edep_total_mev"], 1.5)
        self.assertEqual(role_stats["target"]["step_count"], 16)

    def test_derive_role_stats_keeps_detector_role(self) -> None:
        role_stats = derive_role_stats(
            {
                "Target": {
                    "edep_total_mev": 1.0,
                    "edep_mean_mev_per_event": 0.5,
                    "hit_events": 2,
                    "step_count": 8,
                    "track_entries": 2,
                },
                "Detector": {
                    "edep_total_mev": 0.3,
                    "edep_mean_mev_per_event": 0.15,
                    "hit_events": 1,
                    "step_count": 4,
                    "track_entries": 1,
                },
            },
            {"target": ["Target"], "detector": ["Detector"]},
        )
        self.assertAlmostEqual(role_stats["detector"]["edep_total_mev"], 0.3)
        self.assertEqual(role_stats["detector"]["track_entries"], 1)

    def test_missing_schema_version_uses_current_default(self) -> None:
        result = simulation_result_from_dict(
            {
                "run_ok": True,
                "events_requested": 1,
                "events_completed": 1,
                "scoring": {},
            }
        )
        self.assertEqual(result.schema_version, SIMULATION_RESULT_SCHEMA_VERSION)

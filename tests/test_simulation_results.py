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
                "source_spot_radius_mm": 0.0,
                "source_divergence_half_angle_deg": 0.0,
                "source_spot_profile": "uniform_disk",
                "source_spot_sigma_mm": 0.0,
                "source_divergence_profile": "uniform_cone",
                "source_divergence_sigma_deg": 0.0,
                "source_primary_count": 4,
                "source_sampled_position_mean_mm": [0, 0, -20],
                "source_sampled_position_rms_mm": [0, 0, 0],
                "source_sampled_direction_mean": [0, 0, 1],
                "source_sampled_direction_rms": [0, 0, 0],
                "payload_sha256": "abc123",
                "geant4_version": "geant4-test",
                "run_seed": 20260414,
                "run_manifest": {
                    "bridge": "simulation_bridge",
                    "geometry_root_volume": "Target",
                    "detector_enabled": True,
                    "detector_volume_name": "Detector",
                    "scoring_plane_name": "CheckPlane",
                    "scoring_plane_z_mm": 25.0,
                    "scoring_volume_names": ["Target", "Detector"],
                    "scoring_roles": {"target": ["Target"], "detector": ["Detector"]},
                },
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
                    "detector_crossings_enabled": True,
                    "plane_crossings_enabled": True,
                    "plane_crossing_name": "CheckPlane",
                    "plane_crossing_z_mm": 25.0,
                    "plane_crossing_count": 5,
                    "plane_crossing_events": 4,
                    "plane_crossing_forward_count": 5,
                    "plane_crossing_forward_events": 4,
                    "plane_crossing_reverse_count": 0,
                    "plane_crossing_reverse_events": 0,
                    "plane_crossing_mean_per_event": 1.25,
                    "plane_crossing_particle_counts": {"gamma": 5},
                    "plane_crossing_particle_events": {"gamma": 4},
                    "detector_crossing_count": 2,
                    "detector_crossing_events": 2,
                    "detector_crossing_mean_per_event": 0.5,
                    "detector_crossing_particle_counts": {"gamma": 2},
                    "detector_crossing_particle_events": {"gamma": 2},
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
                            "crossing_events": 2,
                            "crossing_count": 2,
                            "crossing_mean_per_event": 0.5,
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
        self.assertEqual(result.run_seed, 20260414)
        self.assertEqual(result.source_spot_radius_mm, 0.0)
        self.assertEqual(result.source_divergence_half_angle_deg, 0.0)
        self.assertEqual(result.source_spot_profile, "uniform_disk")
        self.assertEqual(result.source_spot_sigma_mm, 0.0)
        self.assertEqual(result.source_divergence_profile, "uniform_cone")
        self.assertEqual(result.source_divergence_sigma_deg, 0.0)
        self.assertEqual(result.source_model.spot_profile, "uniform_disk")
        self.assertEqual(result.source_model.spot_radius_mm, 0.0)
        self.assertEqual(result.source_model.divergence_profile, "uniform_cone")
        self.assertEqual(result.source_model.divergence_half_angle_deg, 0.0)
        self.assertEqual(result.source_primary_count, 4)
        self.assertEqual(result.source_sampled_position_mean_mm, (0.0, 0.0, -20.0))
        self.assertEqual(result.source_sampled_position_rms_mm, (0.0, 0.0, 0.0))
        self.assertEqual(result.source_sampled_direction_mean, (0.0, 0.0, 1.0))
        self.assertEqual(result.source_sampled_direction_rms, (0.0, 0.0, 0.0))
        self.assertEqual(result.source_sampling.primary_count, 4)
        self.assertEqual(result.source_sampling.sampled_direction_mean, (0.0, 0.0, 1.0))
        self.assertEqual(result.run_manifest["geometry_root_volume"], "Target")
        self.assertEqual(result.scoring.plane_crossing_name, "CheckPlane")
        self.assertEqual(result.scoring.plane_crossing_count, 5)
        self.assertEqual(result.scoring.plane_crossing_forward_count, 5)
        self.assertEqual(result.scoring.plane_crossing_reverse_count, 0)
        self.assertAlmostEqual(result.scoring.plane_crossing_mean_per_event, 1.25)
        self.assertEqual(result.scoring.plane_crossing_particle_counts["gamma"], 5)
        self.assertEqual(result.scoring.plane_crossing_particle_events["gamma"], 4)
        self.assertEqual(result.source_position_mm, (0.0, 0.0, -20.0))
        self.assertTrue(result.detector.enabled)
        self.assertEqual(result.detector.volume_name, "Detector")
        self.assertEqual(result.scoring.detector_crossing_count, 2)
        self.assertEqual(result.scoring.detector_crossing_events, 2)
        self.assertAlmostEqual(result.scoring.detector_crossing_mean_per_event, 0.5)
        self.assertEqual(result.scoring.detector_crossing_particle_counts["gamma"], 2)
        self.assertEqual(result.scoring.detector_crossing_particle_events["gamma"], 2)
        self.assertEqual(result.scoring.target_hit_events, 3)
        self.assertEqual(result.scoring.target_step_count, 18)
        self.assertEqual(result.scoring.volume_stats["Target"]["track_entries"], 4)
        self.assertEqual(result.scoring.role_stats["target"]["track_entries"], 4)
        self.assertAlmostEqual(result.scoring.role_stats["target"]["crossing_mean_per_event"], 0.5)

    def test_load_simulation_result_reads_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "run_summary.json"
            summary_path.write_text(
                """{
  "run_ok": true,
  "events_requested": 2,
  "events_completed": 2,
  "schema_version": "2026-04-14.v7",
  "geometry_structure": "single_tubs",
  "material": "G4_W",
  "particle": "proton",
  "source_type": "beam",
  "source_spot_radius_mm": 2.0,
  "source_divergence_half_angle_deg": 1.0,
  "source_spot_profile": "gaussian",
  "source_spot_sigma_mm": 0.75,
  "source_divergence_profile": "gaussian",
  "source_divergence_sigma_deg": 0.25,
  "source_primary_count": 2,
  "source_sampled_position_mean_mm": [0.1, -0.2, -250],
  "source_sampled_position_rms_mm": [1.2, 0.8, 0.0],
  "source_sampled_direction_mean": [0.01, 0.02, 0.999],
  "source_sampled_direction_rms": [0.02, 0.02, 0.001],
  "payload_sha256": "def456",
  "geant4_version": "geant4-test",
  "run_seed": 2718,
  "run_manifest": {
    "bridge": "simulation_bridge",
    "geometry_root_volume": "Target",
    "detector_enabled": true,
    "detector_volume_name": "Detector",
    "scoring_plane_name": "DetectorPlane",
    "scoring_plane_z_mm": 40.0,
    "scoring_volume_names": ["Target", "Detector"],
    "scoring_roles": {"target": ["Target"], "detector": ["Detector"]}
  },
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
    "detector_crossings_enabled": true,
    "plane_crossings_enabled": true,
    "plane_crossing_name": "DetectorPlane",
    "plane_crossing_z_mm": 40.0,
    "plane_crossing_count": 2,
    "plane_crossing_events": 2,
    "plane_crossing_forward_count": 2,
    "plane_crossing_forward_events": 2,
    "plane_crossing_reverse_count": 0,
    "plane_crossing_reverse_events": 0,
    "plane_crossing_mean_per_event": 1.0,
    "plane_crossing_particle_counts": {"proton": 2},
    "plane_crossing_particle_events": {"proton": 2},
    "detector_crossing_count": 1,
    "detector_crossing_events": 1,
    "detector_crossing_mean_per_event": 0.5,
    "detector_crossing_particle_counts": {"proton": 1},
    "detector_crossing_particle_events": {"proton": 1},
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
        "crossing_events": 0,
        "crossing_count": 0,
        "crossing_mean_per_event": 0.0,
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
        self.assertEqual(result.source_spot_radius_mm, 2.0)
        self.assertEqual(result.source_divergence_half_angle_deg, 1.0)
        self.assertEqual(result.source_spot_profile, "gaussian")
        self.assertEqual(result.source_spot_sigma_mm, 0.75)
        self.assertEqual(result.source_divergence_profile, "gaussian")
        self.assertEqual(result.source_divergence_sigma_deg, 0.25)
        self.assertEqual(result.source_model.spot_profile, "gaussian")
        self.assertEqual(result.source_model.spot_sigma_mm, 0.75)
        self.assertEqual(result.source_model.divergence_profile, "gaussian")
        self.assertEqual(result.source_model.divergence_sigma_deg, 0.25)
        self.assertEqual(result.source_primary_count, 2)
        self.assertEqual(result.source_sampled_position_mean_mm, (0.1, -0.2, -250.0))
        self.assertEqual(result.source_sampled_position_rms_mm, (1.2, 0.8, 0.0))
        self.assertEqual(result.source_sampled_direction_mean, (0.01, 0.02, 0.999))
        self.assertEqual(result.source_sampled_direction_rms, (0.02, 0.02, 0.001))
        self.assertEqual(result.source_sampling.primary_count, 2)
        self.assertEqual(result.source_sampling.sampled_position_rms_mm, (1.2, 0.8, 0.0))
        payload = result.to_payload()
        self.assertEqual(payload["source_model"]["spot_profile"], "gaussian")
        self.assertEqual(payload["source_sampling"]["primary_count"], 2)
        self.assertEqual(payload["source_spot_profile"], "gaussian")
        self.assertEqual(payload["source_primary_count"], 2)
        self.assertEqual(result.run_seed, 2718)
        self.assertEqual(result.run_manifest["detector_volume_name"], "Detector")
        self.assertEqual(result.scoring.plane_crossing_name, "DetectorPlane")
        self.assertEqual(result.scoring.plane_crossing_events, 2)
        self.assertEqual(result.scoring.plane_crossing_forward_events, 2)
        self.assertAlmostEqual(result.scoring.plane_crossing_mean_per_event, 1.0)
        self.assertEqual(result.scoring.plane_crossing_particle_counts["proton"], 2)
        self.assertEqual(result.scoring.plane_crossing_particle_events["proton"], 2)
        self.assertTrue(result.detector.enabled)
        self.assertEqual(result.scoring.detector_crossing_count, 1)
        self.assertAlmostEqual(result.scoring.detector_crossing_mean_per_event, 0.5)
        self.assertEqual(result.scoring.detector_crossing_particle_counts["proton"], 1)
        self.assertEqual(result.scoring.detector_crossing_particle_events["proton"], 1)
        self.assertAlmostEqual(result.scoring.target_edep_total_mev, 1.25)
        self.assertEqual(result.scoring.volume_stats["Target"]["step_count"], 9)
        self.assertEqual(result.scoring.role_stats["target"]["step_count"], 9)
        self.assertAlmostEqual(result.scoring.role_stats["target"]["crossing_mean_per_event"], 0.0)

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
                "crossing_events": 2,
                "crossing_count": 2,
                "crossing_mean_per_event": 1.0,
                "step_count": 8,
                "track_entries": 2,
            },
            "Detector": {
                "edep_total_mev": 0.3,
                "edep_mean_mev_per_event": 0.15,
                "hit_events": 1,
                "crossing_events": 1,
                "crossing_count": 1,
                "crossing_mean_per_event": 0.5,
                "step_count": 4,
                "track_entries": 1,
            },
        },
        {"target": ["Target"], "detector": ["Detector"]},
    )
        self.assertAlmostEqual(role_stats["detector"]["edep_total_mev"], 0.3)
        self.assertEqual(role_stats["detector"]["track_entries"], 1)
        self.assertEqual(role_stats["detector"]["crossing_count"], 1)
        self.assertAlmostEqual(role_stats["target"]["crossing_mean_per_event"], 1.0)
        self.assertAlmostEqual(role_stats["detector"]["crossing_mean_per_event"], 0.5)

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

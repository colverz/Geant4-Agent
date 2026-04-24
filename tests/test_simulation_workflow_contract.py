from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from core.simulation import (
    RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
    SIMULATION_RESULT_SCHEMA_VERSION,
    build_runtime_smoke_report,
    build_simulation_spec,
    simulation_result_from_dict,
)
from mcp.geant4.adapter import _load_run_summary_payload
from mcp.geant4.runtime_payload import build_runtime_payload


def _representative_config() -> dict:
    return {
        "geometry": {
            "structure": "single_box",
            "root_name": "Target",
            "size_triplet_mm": [10.0, 20.0, 30.0],
        },
        "materials": {
            "selected_materials": ["G4_Cu"],
            "volume_material_map": {"Target": "G4_Cu"},
        },
        "simulation": {
            "detector": {
                "enabled": True,
                "name": "Detector",
                "material": "G4_Si",
                "position": {"type": "vector", "value": [0.0, 0.0, 50.0]},
                "size_triplet_mm": [12.0, 12.0, 1.5],
            },
            "run": {"seed": 2468},
        },
        "source": {
            "type": "beam",
            "particle": "gamma",
            "energy": 1.0,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
            "spot_radius_mm": 2.5,
            "spot_profile": "gaussian",
            "spot_sigma_mm": 0.75,
            "divergence_half_angle_deg": 1.25,
            "divergence_profile": "gaussian",
            "divergence_sigma_deg": 0.35,
        },
        "physics_list": {"name": "FTFP_BERT"},
        "scoring": {
            "target_edep": True,
            "detector_crossings": True,
            "plane_crossings": True,
            "plane": {"name": "MidPlane", "z_mm": 15.0},
        },
    }


def _run_summary_from_payload(runtime_payload: dict) -> dict:
    return {
        "run_ok": True,
        "events_requested": 3,
        "events_completed": 3,
        "schema_version": SIMULATION_RESULT_SCHEMA_VERSION,
        "geometry_structure": runtime_payload["structure"],
        "material": runtime_payload["material"],
        "particle": runtime_payload["particle"],
        "source_type": runtime_payload["source_type"],
        "source_spot_radius_mm": runtime_payload["source_spot_radius_mm"],
        "source_divergence_half_angle_deg": runtime_payload["source_divergence_half_angle_deg"],
        "source_spot_profile": runtime_payload["source_spot_profile"],
        "source_spot_sigma_mm": runtime_payload["source_spot_sigma_mm"],
        "source_divergence_profile": runtime_payload["source_divergence_profile"],
        "source_divergence_sigma_deg": runtime_payload["source_divergence_sigma_deg"],
        "source_primary_count": 3,
        "source_sampled_position_mean_mm": [0.0, 0.0, -20.0],
        "source_sampled_position_rms_mm": [0.5, 0.25, 0.0],
        "source_sampled_direction_mean": [0.0, 0.0, 1.0],
        "source_sampled_direction_rms": [0.01, 0.01, 0.0],
        "source_position_mm": [
            runtime_payload["position"]["x"],
            runtime_payload["position"]["y"],
            runtime_payload["position"]["z"],
        ],
        "source_direction": [
            runtime_payload["direction"]["x"],
            runtime_payload["direction"]["y"],
            runtime_payload["direction"]["z"],
        ],
        "physics_list": runtime_payload["physics_list"],
        "events": 3,
        "mode": "batch",
        "run_seed": runtime_payload["run"]["seed"],
        "run_manifest": runtime_payload["run_manifest"],
        "detector": {
            "enabled": True,
            "volume_name": runtime_payload["detector_name"],
            "material": runtime_payload["detector_material"],
            "position_mm": [
                runtime_payload["detector_position"]["x"],
                runtime_payload["detector_position"]["y"],
                runtime_payload["detector_position"]["z"],
            ],
            "size_mm": [
                runtime_payload["detector_size_x"],
                runtime_payload["detector_size_y"],
                runtime_payload["detector_size_z"],
            ],
        },
        "scoring": {
            "target_edep_enabled": True,
            "detector_crossings_enabled": True,
            "plane_crossings_enabled": True,
            "plane_crossing_name": "MidPlane",
            "plane_crossing_z_mm": 15.0,
            "plane_crossing_count": 2,
            "plane_crossing_events": 2,
            "plane_crossing_forward_count": 2,
            "plane_crossing_forward_events": 2,
            "plane_crossing_reverse_count": 0,
            "plane_crossing_reverse_events": 0,
            "plane_crossing_mean_per_event": 2.0 / 3.0,
            "plane_crossing_particle_counts": {"gamma": 2},
            "plane_crossing_particle_events": {"gamma": 2},
            "detector_crossing_count": 1,
            "detector_crossing_events": 1,
            "detector_crossing_mean_per_event": 1.0 / 3.0,
            "detector_crossing_particle_counts": {"gamma": 1},
            "detector_crossing_particle_events": {"gamma": 1},
            "target_edep_total_mev": 0.8,
            "target_edep_mean_mev_per_event": 0.8 / 3.0,
            "target_hit_events": 2,
            "target_step_count": 9,
            "target_track_entries": 3,
            "volume_stats": {
                "Target": {
                    "edep_total_mev": 0.8,
                    "edep_mean_mev_per_event": 0.8 / 3.0,
                    "hit_events": 2,
                    "crossing_events": 2,
                    "crossing_count": 2,
                    "crossing_mean_per_event": 2.0 / 3.0,
                    "step_count": 9,
                    "track_entries": 3,
                },
                "Detector": {
                    "edep_total_mev": 0.05,
                    "edep_mean_mev_per_event": 0.05 / 3.0,
                    "hit_events": 1,
                    "crossing_events": 1,
                    "crossing_count": 1,
                    "crossing_mean_per_event": 1.0 / 3.0,
                    "step_count": 4,
                    "track_entries": 1,
                },
            },
        },
    }


class SimulationWorkflowContractTest(unittest.TestCase):
    def test_config_to_runtime_summary_result_and_mcp_payload_contract(self) -> None:
        spec = build_simulation_spec(_representative_config(), events=3, mode="batch")
        runtime_payload = build_runtime_payload(spec)
        run_summary = _run_summary_from_payload(runtime_payload)

        result = simulation_result_from_dict(run_summary)
        self.assertEqual(result.scoring.plane_crossing.count, 2)
        self.assertEqual(result.scoring.plane_crossing_particle_counts["gamma"], 2)
        self.assertEqual(result.scoring.volume_stats["Detector"].track_entries, 1)

        payload = result.to_payload()
        self.assertEqual(payload["scoring"]["plane_crossing_count"], 2)
        self.assertEqual(payload["source_model"]["spot_profile"], "gaussian")
        self.assertEqual(payload["source_spot_profile"], "gaussian")
        self.assertEqual(payload["result_summary"]["run"]["completion_fraction"], 1.0)
        self.assertEqual(payload["result_summary"]["configuration"]["source_type"], "beam")
        self.assertEqual(payload["result_summary"]["scoring"]["target"]["target_hit_events"], 2)
        self.assertEqual(payload["result_summary"]["scoring"]["plane_crossing"]["plane_crossing_count"], 2)
        self.assertEqual(payload["result_summary"]["scoring"]["volumes"]["Detector"]["track_entries"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "run_summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
            mcp_payload = _load_run_summary_payload(
                [f"artifact_dir={artifact_dir}"],
                [],
                runtime_payload,
            )

        self.assertIsNotNone(mcp_payload)
        self.assertEqual(mcp_payload["scoring"]["plane_crossing_count"], 2)
        self.assertEqual(mcp_payload["scoring"]["role_stats"]["target"]["track_entries"], 3)
        self.assertEqual(mcp_payload["scoring"]["role_stats"]["detector"]["track_entries"], 1)
        self.assertEqual(mcp_payload["result_summary"]["scoring"]["roles"]["target"]["track_entries"], 3)
        self.assertEqual(mcp_payload["result_summary"]["scoring"]["roles"]["detector"]["track_entries"], 1)
        self.assertIn("run_summary_path", mcp_payload)

        smoke_report = build_runtime_smoke_report(
            events=3,
            run_payload={"simulation_result": mcp_payload, "result_summary": mcp_payload["result_summary"]},
            summary_payload={
                "result_summary": mcp_payload["result_summary"],
                "artifact_dir": mcp_payload["artifact_dir"],
                "run_summary_path": mcp_payload["run_summary_path"],
            },
        )
        self.assertEqual(smoke_report["schema_version"], RUNTIME_SMOKE_REPORT_SCHEMA_VERSION)
        self.assertTrue(smoke_report["ok"])
        self.assertEqual(smoke_report["events_requested"], 3)
        self.assertEqual(smoke_report["events_completed"], 3)
        self.assertEqual(smoke_report["configuration"]["geometry_structure"], "single_box")
        self.assertEqual(smoke_report["configuration"]["particle"], "gamma")
        self.assertEqual(smoke_report["key_metrics"]["target_edep_total_mev"], 0.8)
        self.assertEqual(smoke_report["key_metrics"]["plane_crossing_count"], 2)
        self.assertEqual(smoke_report["run_summary_path"], mcp_payload["run_summary_path"])

    def test_runtime_payload_covers_cpp_consumed_schema_keys(self) -> None:
        runtime_payload = build_runtime_payload(build_simulation_spec(_representative_config(), events=3))
        main_cc = Path("runtime/geant4_local_app/main.cc").read_text(encoding="utf-8")

        top_level_keys = {
            "structure",
            "material",
            "root_volume_name",
            "particle",
            "source_type",
            "physics_list",
            "energy",
            "source_spot_radius_mm",
            "source_divergence_half_angle_deg",
            "source_spot_profile",
            "source_spot_sigma_mm",
            "source_divergence_profile",
            "source_divergence_sigma_deg",
            "position",
            "direction",
            "size_x",
            "size_y",
            "size_z",
            "detector_enabled",
            "detector_name",
            "detector_material",
            "detector_position",
            "detector_size_x",
            "detector_size_y",
            "detector_size_z",
            "run",
            "scoring",
        }
        missing_payload_keys = sorted(top_level_keys - set(runtime_payload.keys()))
        self.assertEqual(missing_payload_keys, [])
        for key in top_level_keys:
            self.assertIn(f'"{key}"', main_cc)
        self.assertIn('"payload_sha256"', main_cc)

        for key in (
            "type",
            "particle",
            "energy_mev",
            "position_mm",
            "direction_vec",
            "spot_radius_mm",
            "divergence_half_angle_deg",
            "spot_profile",
            "spot_sigma_mm",
            "divergence_profile",
            "divergence_sigma_deg",
        ):
            self.assertIn(key, runtime_payload["source"])
            self.assertIn(f'"{key}"', main_cc)

        for key in ("target_edep", "detector_crossings", "plane_crossings", "plane", "volume_names", "volume_roles"):
            self.assertIn(key, runtime_payload["scoring"])
            self.assertIn(f'"{key}"', main_cc)


if __name__ == "__main__":
    unittest.main()

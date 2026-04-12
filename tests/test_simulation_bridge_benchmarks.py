from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from core.simulation import SIMULATION_RESULT_SCHEMA_VERSION
from mcp.geant4.runtime_payload import build_runtime_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "mcp" / "geant4" / "local_wrapper.py"
RUNTIME_EXE = REPO_ROOT / "runtime" / "geant4_local_app" / "build" / "Release" / "geant4_local_app.exe"


@unittest.skipUnless(RUNTIME_EXE.exists(), "Geant4 local runtime executable is not available")
class SimulationBridgeBenchmarkTest(unittest.TestCase):
    def _run_wrapper(self, config: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.json"
            artifact_dir = tmp / "artifacts"
            runtime_payload = build_runtime_payload(config)
            runtime_payload.pop("raw_config", None)
            config_path.write_text(json.dumps(runtime_payload, ensure_ascii=True, indent=2), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER),
                    "--config",
                    str(config_path),
                    "--events",
                    "2",
                    "--mode",
                    "batch",
                    "--artifact-dir",
                    str(artifact_dir),
                ],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"wrapper failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
            )
            summary_path = artifact_dir / "run_summary.json"
            self.assertTrue(summary_path.exists(), msg=f"missing run_summary.json\nSTDOUT:\n{completed.stdout}")
            return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_target_only_box_batch_returns_stable_schema(self) -> None:
        summary = self._run_wrapper(
            {
                "geometry": {
                    "structure": "single_box",
                    "root_name": "Target",
                    "params": {"module_x": 10.0, "module_y": 10.0, "module_z": 10.0},
                },
                "materials": {"selected_materials": ["G4_Cu"]},
                "source": {
                    "type": "point",
                    "particle": "gamma",
                    "energy": 1.0,
                    "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
                    "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
                },
                "physics": {"physics_list": "FTFP_BERT"},
                "scoring": {"target_edep": True},
            }
        )
        self.assertEqual(summary["schema_version"], SIMULATION_RESULT_SCHEMA_VERSION)
        self.assertTrue(summary["run_ok"])
        self.assertEqual(summary["geometry_structure"], "single_box")
        self.assertIn("Target", summary["scoring"]["volume_stats"])

    def test_target_and_detector_batch_returns_both_volume_stats(self) -> None:
        summary = self._run_wrapper(
            {
                "geometry": {
                    "structure": "single_box",
                    "root_name": "Target",
                    "params": {"module_x": 10.0, "module_y": 10.0, "module_z": 10.0},
                },
                "materials": {"selected_materials": ["G4_Cu"]},
                "simulation": {
                    "detector": {
                        "enabled": True,
                        "name": "Detector",
                        "material": "G4_Si",
                        "position": {"type": "vector", "value": [0.0, 0.0, 40.0]},
                        "size_triplet_mm": [20.0, 20.0, 2.0],
                    }
                },
                "source": {
                    "type": "point",
                    "particle": "gamma",
                    "energy": 1.0,
                    "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
                    "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
                },
                "physics": {"physics_list": "FTFP_BERT"},
                "scoring": {"target_edep": True},
            }
        )
        self.assertEqual(summary["schema_version"], SIMULATION_RESULT_SCHEMA_VERSION)
        self.assertTrue(summary["detector"]["enabled"])
        self.assertEqual(summary["detector"]["volume_name"], "Detector")
        self.assertIn("Target", summary["scoring"]["volume_stats"])
        self.assertIn("Detector", summary["scoring"]["volume_stats"])


if __name__ == "__main__":
    unittest.main()

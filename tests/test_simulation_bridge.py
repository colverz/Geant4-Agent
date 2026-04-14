from __future__ import annotations

import unittest

from core.simulation import build_simulation_spec
from mcp.geant4.runtime_payload import build_runtime_payload


class SimulationBridgeTest(unittest.TestCase):
    def test_build_simulation_spec_from_box_config(self) -> None:
        spec = build_simulation_spec(
            {
                "geometry": {
                    "structure": "single_box",
                    "root_name": "target",
                    "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
                },
                "simulation": {
                    "detector": {
                        "enabled": True,
                        "name": "Detector",
                        "material": "G4_Si",
                        "position": {"type": "vector", "value": [0.0, 0.0, 120.0]},
                        "size_triplet_mm": [25.0, 25.0, 3.0],
                    }
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
                "run": {"seed": 20260414},
            },
            events=25,
            mode="batch",
        )
        self.assertEqual(spec.geometry.structure, "single_box")
        self.assertEqual(spec.geometry.material, "G4_Cu")
        self.assertEqual(spec.geometry.root_volume_name, "target")
        self.assertEqual(spec.geometry.size_x_mm, 10.0)
        self.assertEqual(spec.source.source_type, "point")
        self.assertEqual(spec.source.position_mm, (0.0, 0.0, -20.0))
        self.assertEqual(spec.run.events, 25)
        self.assertEqual(spec.run.seed, 20260414)
        self.assertTrue(spec.scoring.target_edep)
        self.assertTrue(spec.scoring.detector_crossings)
        self.assertEqual(spec.scoring.volume_roles["target"], ("target",))
        self.assertEqual(spec.scoring.volume_roles["detector"], ("Detector",))
        self.assertIsNotNone(spec.detector)
        self.assertEqual(spec.detector.volume_name, "Detector")
        self.assertEqual(spec.detector.position_mm, (0.0, 0.0, 120.0))

    def test_runtime_payload_keeps_nested_bridge_sections(self) -> None:
        payload = build_runtime_payload(
            {
                "geometry": {
                    "structure": "single_tubs",
                    "root_name": "target",
                    "params": {"child_rmax": 5.0, "child_hz": 40.0},
                },
                "simulation": {
                    "detector": {
                        "enabled": True,
                        "name": "Detector",
                        "material": "G4_Si",
                        "position": {"type": "vector", "value": [0.0, 0.0, 60.0]},
                        "size_triplet_mm": [15.0, 15.0, 2.0],
                    }
                },
                "materials": {"selected_materials": ["G4_W"]},
                "source": {
                    "type": "beam",
                    "particle": "proton",
                    "energy": 250.0,
                    "position": {"type": "vector", "value": [0.0, 0.0, -250.0]},
                    "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
                },
                "physics": {"physics_list": "QGSP_BERT"},
                "run": {"seed": 4242},
                "scoring": {
                    "target_edep": True,
                    "plane_crossings": True,
                    "plane": {"name": "BeamCheck", "z_mm": 25.0},
                },
            }
        )
        self.assertEqual(payload["geometry"]["structure"], "single_tubs")
        self.assertEqual(payload["geometry"]["root_volume_name"], "target")
        self.assertEqual(payload["source"]["type"], "beam")
        self.assertEqual(payload["physics"]["list"], "QGSP_BERT")
        self.assertEqual(payload["run"]["seed"], 4242)
        self.assertEqual(payload["run_manifest"]["bridge"], "simulation_bridge")
        self.assertEqual(payload["run_manifest"]["geometry_root_volume"], "target")
        self.assertEqual(payload["run_manifest"]["detector_volume_name"], "Detector")
        self.assertEqual(payload["run_manifest"]["scoring_plane_name"], "BeamCheck")
        self.assertTrue(payload["scoring"]["target_edep"])
        self.assertTrue(payload["scoring"]["detector_crossings"])
        self.assertTrue(payload["scoring"]["plane_crossings"])
        self.assertEqual(payload["scoring"]["plane"]["name"], "BeamCheck")
        self.assertEqual(payload["scoring"]["plane"]["z_mm"], 25.0)
        self.assertEqual(payload["scoring"]["volume_names"], ["target", "Detector"])
        self.assertEqual(payload["scoring"]["volume_roles"]["target"], ["target"])
        self.assertEqual(payload["scoring"]["volume_roles"]["detector"], ["Detector"])
        self.assertTrue(payload["detector"]["enabled"])
        self.assertEqual(payload["detector_name"], "Detector")
        self.assertEqual(payload["detector_material"], "G4_Si")
        self.assertEqual(payload["detector_position"]["z"], 60.0)
        self.assertEqual(payload["radius"], 5.0)
        self.assertEqual(payload["half_length"], 40.0)


if __name__ == "__main__":
    unittest.main()

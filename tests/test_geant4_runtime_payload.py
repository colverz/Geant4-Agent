from __future__ import annotations

import unittest

from mcp.geant4.runtime_payload import build_runtime_payload


class Geant4RuntimePayloadTest(unittest.TestCase):
    def test_box_payload_maps_nested_config(self) -> None:
        payload = build_runtime_payload(
            {
                "geometry": {
                    "structure": "single_box",
                    "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
                },
                "materials": {"selected_materials": ["G4_Cu"]},
                "source": {
                    "particle": "gamma",
                    "type": "point",
                    "energy": 1000.0,
                    "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
                    "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
                },
                "physics": {"physics_list": "FTFP_BERT"},
            }
        )
        self.assertEqual(payload["structure"], "single_box")
        self.assertEqual(payload["material"], "G4_Cu")
        self.assertEqual(payload["particle"], "gamma")
        self.assertEqual(payload["physics_list"], "FTFP_BERT")
        self.assertEqual(payload["root_volume_name"], "Target")
        self.assertTrue(payload["scoring"]["detector_crossings"])
        self.assertEqual(payload["size_x"], 10.0)
        self.assertEqual(payload["size_y"], 20.0)
        self.assertEqual(payload["size_z"], 30.0)
        self.assertEqual(payload["position"]["z"], -20.0)
        self.assertEqual(payload["direction"]["z"], 1.0)

    def test_tubs_payload_prefers_volume_material_map(self) -> None:
        payload = build_runtime_payload(
            {
                "geometry": {
                    "structure": "single_tubs",
                    "root_name": "target",
                    "params": {"child_rmax": 5.0, "child_hz": 40.0},
                },
                "materials": {
                    "selected_materials": ["G4_Al"],
                    "volume_material_map": {"target": "G4_W"},
                },
                "simulation": {
                    "detector": {
                        "enabled": True,
                        "name": "Detector",
                        "material": "G4_Si",
                        "position": {"type": "vector", "value": [0.0, 0.0, 50.0]},
                        "size_triplet_mm": [12.0, 12.0, 1.5],
                    }
                },
                "source": {"particle": "proton", "energy": 250.0},
                "physics_list": {"name": "QGSP_BERT"},
            }
        )
        self.assertEqual(payload["structure"], "single_tubs")
        self.assertEqual(payload["material"], "G4_W")
        self.assertEqual(payload["root_volume_name"], "target")
        self.assertEqual(payload["radius"], 5.0)
        self.assertEqual(payload["half_length"], 40.0)
        self.assertEqual(payload["physics_list"], "QGSP_BERT")
        self.assertTrue(payload["scoring"]["detector_crossings"])
        self.assertTrue(payload["detector_enabled"])
        self.assertEqual(payload["detector_name"], "Detector")
        self.assertEqual(payload["detector_size_z"], 1.5)


if __name__ == "__main__":
    unittest.main()

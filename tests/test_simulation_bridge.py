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
                    "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
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
            },
            events=25,
            mode="batch",
        )
        self.assertEqual(spec.geometry.structure, "single_box")
        self.assertEqual(spec.geometry.material, "G4_Cu")
        self.assertEqual(spec.geometry.size_x_mm, 10.0)
        self.assertEqual(spec.source.source_type, "point")
        self.assertEqual(spec.source.position_mm, (0.0, 0.0, -20.0))
        self.assertEqual(spec.run.events, 25)
        self.assertTrue(spec.scoring.target_edep)

    def test_runtime_payload_keeps_nested_bridge_sections(self) -> None:
        payload = build_runtime_payload(
            {
                "geometry": {
                    "structure": "single_tubs",
                    "params": {"child_rmax": 5.0, "child_hz": 40.0},
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
            }
        )
        self.assertEqual(payload["geometry"]["structure"], "single_tubs")
        self.assertEqual(payload["source"]["type"], "beam")
        self.assertEqual(payload["physics"]["list"], "QGSP_BERT")
        self.assertTrue(payload["scoring"]["target_edep"])
        self.assertEqual(payload["scoring"]["volume_names"], ["Target"])
        self.assertEqual(payload["radius"], 5.0)
        self.assertEqual(payload["half_length"], 40.0)


if __name__ == "__main__":
    unittest.main()

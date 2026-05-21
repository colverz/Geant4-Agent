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
                    "spot_radius_mm": 0.5,
                    "divergence_half_angle_deg": 0.25,
                    "spot_profile": "gaussian",
                    "spot_sigma_mm": 0.2,
                    "divergence_profile": "gaussian",
                    "divergence_sigma_deg": 0.1,
                },
                "physics": {"physics_list": "FTFP_BERT"},
                "run": {"seed": 31415},
                "scoring": {"plane_crossings": True, "plane": {"name": "SourcePlane", "z_mm": -10.0}},
            }
        )
        self.assertEqual(payload["structure"], "single_box")
        self.assertEqual(payload["material"], "G4_Cu")
        self.assertEqual(payload["particle"], "gamma")
        self.assertEqual(payload["physics_list"], "FTFP_BERT")
        self.assertEqual(payload["run"]["seed"], 31415)
        self.assertEqual(payload["run_manifest"]["geometry_root_volume"], "Target")
        self.assertEqual(payload["run_manifest"]["scoring_plane_name"], "SourcePlane")
        self.assertEqual(payload["root_volume_name"], "Target")
        self.assertTrue(payload["scoring"]["detector_crossings"])
        self.assertTrue(payload["scoring"]["plane_crossings"])
        self.assertEqual(payload["scoring"]["plane"]["z_mm"], -10.0)
        self.assertEqual(payload["size_x"], 10.0)
        self.assertEqual(payload["size_y"], 20.0)
        self.assertEqual(payload["size_z"], 30.0)
        self.assertEqual(payload["position"]["z"], -20.0)
        self.assertEqual(payload["direction"]["z"], 1.0)
        self.assertEqual(payload["source"]["spot_radius_mm"], 0.5)
        self.assertEqual(payload["source"]["divergence_half_angle_deg"], 0.25)
        self.assertEqual(payload["source"]["spot_profile"], "gaussian")
        self.assertEqual(payload["source"]["spot_sigma_mm"], 0.2)
        self.assertEqual(payload["source"]["divergence_profile"], "gaussian")
        self.assertEqual(payload["source"]["divergence_sigma_deg"], 0.1)
        self.assertEqual(payload["source_spot_radius_mm"], 0.5)
        self.assertEqual(payload["source_divergence_half_angle_deg"], 0.25)
        self.assertEqual(payload["source_spot_profile"], "gaussian")
        self.assertEqual(payload["source_spot_sigma_mm"], 0.2)
        self.assertEqual(payload["source_divergence_profile"], "gaussian")
        self.assertEqual(payload["source_divergence_sigma_deg"], 0.1)

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
                    },
                    "run": {"seed": 2718},
                },
                "source": {
                    "particle": "proton",
                    "type": "beam",
                    "energy": 250.0,
                    "spot_radius_mm": 2.0,
                    "divergence_half_angle_deg": 1.0,
                    "spot_profile": "uniform_disk",
                    "divergence_profile": "uniform_cone",
                },
                "physics_list": {"name": "QGSP_BERT"},
                "scoring": {"plane_crossings": True, "plane": {"name": "DetectorPlane", "z_mm": 40.0}},
            }
        )
        self.assertEqual(payload["structure"], "single_tubs")
        self.assertEqual(payload["material"], "G4_W")
        self.assertEqual(payload["root_volume_name"], "target")
        self.assertEqual(payload["radius"], 5.0)
        self.assertEqual(payload["half_length"], 40.0)
        self.assertEqual(payload["physics_list"], "QGSP_BERT")
        self.assertEqual(payload["run"]["seed"], 2718)
        self.assertEqual(payload["run_manifest"]["detector_volume_name"], "Detector")
        self.assertEqual(payload["run_manifest"]["scoring_plane_name"], "DetectorPlane")
        self.assertTrue(payload["scoring"]["detector_crossings"])
        self.assertTrue(payload["scoring"]["plane_crossings"])
        self.assertTrue(payload["detector_enabled"])
        self.assertEqual(payload["detector_name"], "Detector")
        self.assertEqual(payload["detector_size_z"], 1.5)
        self.assertEqual(payload["source"]["type"], "beam")
        self.assertEqual(payload["source"]["spot_radius_mm"], 2.0)
        self.assertEqual(payload["source"]["divergence_half_angle_deg"], 1.0)
        self.assertEqual(payload["source"]["spot_profile"], "uniform_disk")
        self.assertEqual(payload["source"]["divergence_profile"], "uniform_cone")

    def test_runtime_dsl_payload_preserves_multi_volume_geometry(self) -> None:
        payload = build_runtime_payload(
            {
                "geometry": {
                    "structure": "multi_layer_stack",
                    "root_name": "ShieldStack",
                    "size_triplet_mm": [120.0, 120.0, 0.0],
                    "layers": [
                        {"name": "LeadLayer", "material": "G4_Pb", "thickness_mm": 10.0, "role": "shield"},
                        {"name": "PolyLayer", "material": "G4_POLYETHYLENE", "thickness_mm": 30.0, "role": "shield"},
                    ],
                },
                "source": {
                    "type": "beam",
                    "particle": "gamma",
                    "energy": 1.25,
                    "position": [0.0, 0.0, -100.0],
                    "direction": [0.0, 0.0, 1.0],
                },
                "physics": {"physics_list": "FTFP_BERT"},
                "scoring": {
                    "volume_roles": {"region_a": ["LeadLayer"], "region_b": ["PolyLayer"]},
                    "derived_metrics": ["region_contrast"],
                },
            }
        )

        self.assertEqual(payload["schema_version"], "runtime_dsl.v1")
        self.assertEqual(payload["geometry"]["structure"], "multi_layer_stack")
        self.assertEqual([v["name"] for v in payload["geometry"]["volumes"]], ["LeadLayer", "PolyLayer"])
        self.assertEqual(payload["geometry"]["volumes"][0]["size_mm"], [120.0, 120.0, 10.0])
        self.assertEqual(payload["geometry"]["volumes"][1]["position_mm"][2], 5.0)
        self.assertEqual(payload["geometry"]["roles"]["shield"], ["LeadLayer", "PolyLayer"])
        self.assertEqual(payload["scoring"]["volume_roles"]["region_a"], ["LeadLayer"])
        self.assertEqual(payload["scoring"]["volume_roles"]["region_b"], ["PolyLayer"])
        self.assertIn("LeadLayer", payload["scoring"]["volume_names"])
        self.assertIn("PolyLayer", payload["run_manifest"]["geometry_volume_names"])
        self.assertEqual(payload["runtime_capabilities"]["paired_run_support"], True)
        self.assertIn("depth_bins", payload["runtime_capabilities"]["scoring_types"])

    def test_step_wedge_generates_scoring_regions_without_flat_payload_regression(self) -> None:
        payload = build_runtime_payload(
            {
                "geometry": {
                    "structure": "step_wedge",
                    "root_name": "SteelWedge",
                    "params": {"module_y": 80.0},
                    "steps": [
                        {"name": "ThinStep", "width_mm": 20.0, "thickness_mm": 5.0, "material": "G4_Fe"},
                        {"name": "ThickStep", "width_mm": 20.0, "thickness_mm": 20.0, "material": "G4_Fe"},
                    ],
                },
                "source": {"type": "isotropic", "particle": "gamma", "energy": 1.0},
                "physics_list": {"name": "FTFP_BERT"},
            }
        )

        self.assertEqual(payload["source_type"], "isotropic")
        self.assertEqual(payload["structure"], "step_wedge")
        self.assertEqual([v["shape"] for v in payload["geometry"]["volumes"]], ["box", "box"])
        self.assertEqual(payload["geometry"]["volumes"][0]["role"], "region_a")
        self.assertEqual(payload["geometry"]["volumes"][1]["role"], "region_b")
        self.assertEqual(payload["scoring"]["volume_roles"]["target"], ["ThinStep", "ThickStep"])
        self.assertIn("ThinStep", payload["scoring"]["volume_names"])


if __name__ == "__main__":
    unittest.main()

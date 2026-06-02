from __future__ import annotations

import unittest

from core.agent_v3.geometry_validator import validate_and_normalize, _normalize_volume


class GeometryValidatorTest(unittest.TestCase):
    def test_returns_fallback_for_missing_volumes(self):
        cfg, warns = validate_and_normalize({})
        self.assertIn("missing", warns[0])
        self.assertEqual(cfg["volumes"][0]["name"], "Target")

    def test_accepts_valid_sphere(self):
        spec = {"volumes": [{"name": "phantom", "shape": "sphere", "material": "G4_WATER", "dimensions": {"radius_mm": 150}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["volumes"][0]["name"], "phantom")
        self.assertEqual(cfg["volumes"][0]["shape"], "sphere")
        self.assertEqual(cfg["volumes"][0]["dimensions"]["radius_mm"], 150.0)
        self.assertEqual(cfg["material"], "G4_Galactic")  # default env

    def test_accepts_valid_box(self):
        spec = {"volumes": [{"name": "shield", "shape": "box", "material": "G4_Pb", "dimensions": {"size_x_mm": 100, "size_y_mm": 100, "size_z_mm": 10}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["volumes"][0]["dimensions"]["size_x_mm"], 100.0)

    def test_accepts_valid_cylinder(self):
        spec = {"volumes": [{"name": "pipe", "shape": "tubs", "material": "G4_Cu", "dimensions": {"radius_mm": 25, "half_length_mm": 100}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["volumes"][0]["shape"], "tubs")
        self.assertEqual(cfg["volumes"][0]["dimensions"]["half_length_mm"], 100.0)

    def test_fills_missing_dimensions_with_defaults(self):
        spec = {"volumes": [{"name": "target", "shape": "sphere", "material": "G4_WATER", "dimensions": {}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["volumes"][0]["dimensions"]["radius_mm"], 50.0)  # sphere default

    def test_corrects_negative_dimensions(self):
        spec = {"volumes": [{"name": "target", "shape": "box", "material": "G4_Pb", "dimensions": {"size_x_mm": -5, "size_y_mm": 10, "size_z_mm": 10}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["volumes"][0]["dimensions"]["size_x_mm"], 10.0)  # corrected to default
        self.assertTrue(any("non-positive" in w for w in warns))

    def test_warns_on_unknown_material(self):
        spec = {"volumes": [{"name": "t", "shape": "box", "material": "G4_UNOBTAINIUM", "dimensions": {"size_x_mm": 10, "size_y_mm": 10, "size_z_mm": 10}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertTrue(any("not in known G4" in w for w in warns))

    def test_accepts_unknown_shape_with_box_defaults(self):
        spec = {"volumes": [{"name": "t", "shape": "exotic", "material": "G4_WATER", "dimensions": {"size_x_mm": 10, "size_y_mm": 10, "size_z_mm": 10}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["volumes"][0]["shape"], "exotic")  # passes through

    def test_multi_volume_layers(self):
        spec = {"volumes": [
            {"name": "layer1", "shape": "box", "material": "G4_Pb", "dimensions": {"size_x_mm": 100, "size_y_mm": 100, "size_z_mm": 5}, "position_mm": [0, 0, -30]},
            {"name": "layer2", "shape": "box", "material": "G4_POLYETHYLENE", "dimensions": {"size_x_mm": 100, "size_y_mm": 100, "size_z_mm": 20}, "position_mm": [0, 0, -5]},
            {"name": "layer3", "shape": "box", "material": "G4_Pb", "dimensions": {"size_x_mm": 100, "size_y_mm": 100, "size_z_mm": 5}, "position_mm": [0, 0, 20]},
        ]}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(len(cfg["volumes"]), 3)
        self.assertEqual(cfg["volumes"][1]["material"], "G4_POLYETHYLENE")

    def test_world_auto_sized_larger_than_volumes(self):
        spec = {"volumes": [{"name": "big", "shape": "box", "material": "G4_WATER", "dimensions": {"size_x_mm": 500, "size_y_mm": 500, "size_z_mm": 500}}]}
        cfg, warns = validate_and_normalize(spec)
        self.assertGreater(cfg["size_mm"][0], 500)  # world should be bigger

    def test_environment_material_respected(self):
        spec = {"volumes": [{"name": "t", "shape": "box", "material": "G4_WATER", "dimensions": {"size_x_mm": 10, "size_y_mm": 10, "size_z_mm": 10}}],
                "environment": {"material": "G4_AIR"}}
        cfg, warns = validate_and_normalize(spec)
        self.assertEqual(cfg["material"], "G4_AIR")

    def test_normalize_volume_handles_position(self):
        vol, w = _normalize_volume({"name": "t", "shape": "box", "position_mm": [1, 2, 3]}, 0)
        self.assertEqual(vol["position_mm"], [1.0, 2.0, 3.0])

    def test_normalize_volume_defaults_position_to_origin(self):
        vol, w = _normalize_volume({"name": "t", "shape": "box"}, 0)
        self.assertEqual(vol["position_mm"], [0.0, 0.0, 0.0])

from __future__ import annotations

import unittest

from core.contracts.semantic import GeometryFrame, SemanticFrame
from core.contracts.slots import GeometrySlots, SlotFrame
from core.geometry.adapters import (
    diff_geometry_config_fragment,
    geometry_spec_to_config_fragment,
    geometry_spec_to_runtime_geometry,
)
from core.geometry.catalog import get_geometry_catalog_entry, resolve_geometry_structure
from core.geometry.compiler import (
    compile_geometry_spec_from_config,
    compile_geometry_spec_from_semantic_frame,
    compile_geometry_spec_from_slot_frame,
)


class GeometryCompilerTests(unittest.TestCase):
    def test_catalog_resolves_box_alias(self) -> None:
        self.assertEqual(resolve_geometry_structure("box"), "single_box")
        entry = get_geometry_catalog_entry("cube")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.structure, "single_box")
        self.assertIsNone(resolve_geometry_structure("sphere"))

    def test_compile_single_box_from_slot_frame(self) -> None:
        frame = SlotFrame(confidence=0.92, geometry=GeometrySlots(kind="box", size_triplet_mm=[10, 20, 30]))
        result = compile_geometry_spec_from_slot_frame(frame)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.structure, "single_box")
        self.assertEqual(result.spec.params["size_triplet_mm"], [10.0, 20.0, 30.0])
        self.assertIn("geometry.params.module_x", result.spec.required_paths)
        self.assertEqual(result.spec.field_resolutions["size_triplet_mm"].status, "user_explicit")

    def test_compile_single_tubs_from_slot_frame(self) -> None:
        frame = SlotFrame(confidence=0.88, geometry=GeometrySlots(kind="cylinder", radius_mm=15, half_length_mm=40))
        result = compile_geometry_spec_from_slot_frame(frame)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.structure, "single_tubs")
        self.assertEqual(result.spec.params["radius_mm"], 15.0)
        self.assertEqual(result.spec.params["half_length_mm"], 40.0)

    def test_compile_keeps_missing_required_geometry_fields(self) -> None:
        frame = SlotFrame(confidence=0.81, geometry=GeometrySlots(kind="box"))
        result = compile_geometry_spec_from_slot_frame(frame)

        self.assertFalse(result.ok)
        self.assertEqual(result.missing_fields, ("size_triplet_mm",))

    def test_compile_rejects_unknown_geometry_kind(self) -> None:
        frame = SlotFrame(confidence=0.75, geometry=GeometrySlots(kind="ring_lattice"))
        result = compile_geometry_spec_from_slot_frame(frame)

        self.assertFalse(result.ok)
        self.assertEqual(result.errors, ("missing_geometry_structure",))

    def test_compile_single_box_from_semantic_frame(self) -> None:
        frame = SemanticFrame(
            geometry=GeometryFrame(
                structure="single_box",
                params={"module_x": 12.0, "module_y": 14.0, "module_z": 16.0},
            )
        )
        result = compile_geometry_spec_from_semantic_frame(frame)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.params["size_triplet_mm"], [12.0, 14.0, 16.0])

    def test_compile_single_tubs_from_config(self) -> None:
        config = {
            "geometry": {
                "structure": "single_tubs",
                "params": {"child_rmax": 8.0, "child_hz": 25.0},
            }
        }
        result = compile_geometry_spec_from_config(config)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.params["radius_mm"], 8.0)
        self.assertEqual(result.spec.params["half_length_mm"], 25.0)

    def test_runtime_geometry_adapter_for_box(self) -> None:
        frame = SlotFrame(confidence=0.9, geometry=GeometrySlots(kind="box", size_triplet_mm=[5, 6, 7]))
        result = compile_geometry_spec_from_slot_frame(frame)

        assert result.spec is not None
        runtime_geometry = geometry_spec_to_runtime_geometry(result.spec)
        self.assertEqual(runtime_geometry["structure"], "single_box")
        self.assertEqual(runtime_geometry["size_x"], 5.0)
        self.assertEqual(runtime_geometry["size_y"], 6.0)
        self.assertEqual(runtime_geometry["size_z"], 7.0)

    def test_runtime_geometry_adapter_for_tubs(self) -> None:
        config = {
            "geometry": {
                "structure": "single_tubs",
                "params": {"child_rmax": 9.0, "child_hz": 11.0},
            }
        }
        result = compile_geometry_spec_from_config(config)

        assert result.spec is not None
        runtime_geometry = geometry_spec_to_runtime_geometry(result.spec)
        self.assertEqual(runtime_geometry["structure"], "single_tubs")
        self.assertEqual(runtime_geometry["radius"], 9.0)
        self.assertEqual(runtime_geometry["half_length"], 11.0)

    def test_compile_single_orb_from_slot_frame(self) -> None:
        frame = SlotFrame(confidence=0.84, geometry=GeometrySlots(kind="orb", radius_mm=22))
        result = compile_geometry_spec_from_slot_frame(frame)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.structure, "single_orb")
        self.assertEqual(result.spec.params["radius_mm"], 22.0)
        self.assertEqual(result.spec.confidence, 0.84)
        self.assertEqual(result.spec.finalization_status, "ready")

    def test_compile_single_cons_from_config(self) -> None:
        config = {
            "geometry": {
                "structure": "single_cons",
                "params": {"rmax1": 5.0, "rmax2": 9.0, "child_hz": 12.0},
            }
        }
        result = compile_geometry_spec_from_config(config)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.params["radius1_mm"], 5.0)
        self.assertEqual(result.spec.params["radius2_mm"], 9.0)
        self.assertEqual(result.spec.params["half_length_mm"], 12.0)

    def test_compile_single_trd_from_semantic_frame(self) -> None:
        frame = SemanticFrame(
            geometry=GeometryFrame(
                structure="single_trd",
                params={"x1": 1.0, "x2": 2.0, "y1": 3.0, "y2": 4.0, "module_z": 5.0},
            )
        )
        result = compile_geometry_spec_from_semantic_frame(frame)

        self.assertTrue(result.ok)
        assert result.spec is not None
        self.assertEqual(result.spec.params["x1_mm"], 1.0)
        self.assertEqual(result.spec.params["z_mm"], 5.0)

    def test_geometry_spec_to_config_fragment_for_box(self) -> None:
        frame = SlotFrame(confidence=0.9, geometry=GeometrySlots(kind="box", size_triplet_mm=[5, 6, 7]))
        result = compile_geometry_spec_from_slot_frame(frame)

        assert result.spec is not None
        fragment = geometry_spec_to_config_fragment(result.spec)
        self.assertEqual(fragment["geometry"]["structure"], "single_box")
        self.assertEqual(fragment["geometry"]["params"]["module_x"], 5.0)
        self.assertEqual(fragment["geometry"]["params"]["module_y"], 6.0)
        self.assertEqual(fragment["geometry"]["params"]["module_z"], 7.0)
        self.assertEqual(fragment["geometry"]["size_triplet_mm"], [5.0, 6.0, 7.0])

    def test_geometry_spec_to_config_fragment_for_cons(self) -> None:
        config = {
            "geometry": {
                "structure": "single_cons",
                "params": {"rmax1": 5.0, "rmax2": 9.0, "child_hz": 12.0},
            }
        }
        result = compile_geometry_spec_from_config(config)

        assert result.spec is not None
        fragment = geometry_spec_to_config_fragment(result.spec)
        self.assertEqual(fragment["geometry"]["params"]["rmax1"], 5.0)
        self.assertEqual(fragment["geometry"]["params"]["rmax2"], 9.0)
        self.assertEqual(fragment["geometry"]["params"]["child_hz"], 12.0)

    def test_diff_geometry_config_fragment_reports_match(self) -> None:
        frame = SlotFrame(confidence=0.9, geometry=GeometrySlots(kind="box", size_triplet_mm=[5, 6, 7]))
        result = compile_geometry_spec_from_slot_frame(frame)

        assert result.spec is not None
        legacy_geometry = {
            "structure": "single_box",
            "params": {"module_x": 5.0, "module_y": 6.0, "module_z": 7.0},
        }
        diff = diff_geometry_config_fragment(result.spec, legacy_geometry)
        self.assertTrue(diff["matches"])
        self.assertEqual(diff["mismatches"], [])

    def test_diff_geometry_config_fragment_reports_mismatch(self) -> None:
        frame = SlotFrame(confidence=0.9, geometry=GeometrySlots(kind="box", size_triplet_mm=[5, 6, 7]))
        result = compile_geometry_spec_from_slot_frame(frame)

        assert result.spec is not None
        legacy_geometry = {
            "structure": "single_box",
            "params": {"module_x": 5.0, "module_y": 6.0, "module_z": 9.0},
        }
        diff = diff_geometry_config_fragment(result.spec, legacy_geometry)
        self.assertFalse(diff["matches"])
        self.assertEqual(diff["mismatches"][0]["field"], "geometry.params.module_z")


if __name__ == "__main__":
    unittest.main()

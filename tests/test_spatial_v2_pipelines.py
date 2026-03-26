from __future__ import annotations

import unittest

from core.contracts.slots import GeometrySlots, SlotFrame, SourceSlots
from core.pipelines.geometry_v2_pipeline import build_v2_geometry_updates
from core.pipelines.selectors import PIPELINE_LEGACY, PIPELINE_V2, select_pipelines
from core.pipelines.spatial_v2_pipeline import build_v2_spatial_updates
from core.pipelines.source_v2_pipeline import build_v2_source_updates
from core.slots.slot_mapper import slot_frame_to_candidates


class SpatialV2PipelineTests(unittest.TestCase):
    def test_selector_accepts_explicit_v2(self) -> None:
        selection = select_pipelines(geometry="v2", source="v2")
        self.assertEqual(selection.geometry, PIPELINE_V2)
        self.assertEqual(selection.source, PIPELINE_V2)

    def test_geometry_v2_pipeline_builds_box_updates_without_legacy(self) -> None:
        frame = SlotFrame(confidence=0.9, geometry=GeometrySlots(kind="box", size_triplet_mm=[10, 20, 30]))
        updates, targets, meta = build_v2_geometry_updates(frame, turn_id=1)
        mapped = {update.path: update.value for update in updates}
        self.assertTrue(meta["compile_ok"])
        self.assertEqual(mapped["geometry.structure"], "single_box")
        self.assertEqual(mapped["geometry.params.module_x"], 10.0)
        self.assertEqual(mapped["geometry.params.module_y"], 20.0)
        self.assertEqual(mapped["geometry.params.module_z"], 30.0)
        self.assertIn("geometry.root_name", targets)

    def test_geometry_v2_pipeline_builds_trd_updates(self) -> None:
        frame = SlotFrame(
            confidence=0.85,
            geometry=GeometrySlots(kind="trd", x1_mm=1, x2_mm=2, y1_mm=3, y2_mm=4, z_mm=5),
        )
        updates, _, meta = build_v2_geometry_updates(frame, turn_id=2)
        mapped = {update.path: update.value for update in updates}
        self.assertTrue(meta["compile_ok"])
        self.assertEqual(mapped["geometry.structure"], "single_trd")
        self.assertEqual(mapped["geometry.params.x1"], 1.0)
        self.assertEqual(mapped["geometry.params.x2"], 2.0)
        self.assertEqual(mapped["geometry.params.y1"], 3.0)
        self.assertEqual(mapped["geometry.params.y2"], 4.0)
        self.assertEqual(mapped["geometry.params.module_z"], 5.0)

    def test_source_v2_pipeline_builds_beam_updates_without_legacy(self) -> None:
        frame = SlotFrame(
            confidence=0.9,
            source=SourceSlots(
                kind="beam",
                particle="gamma",
                energy_mev=5.0,
                position_mm=[0.0, 0.0, -250.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        updates, _, meta = build_v2_source_updates(frame, turn_id=3)
        mapped = {update.path: update.value for update in updates}
        self.assertTrue(meta["compile_ok"])
        self.assertEqual(mapped["source.type"], "beam")
        self.assertEqual(mapped["source.particle"], "gamma")
        self.assertEqual(mapped["source.energy"], 5.0)
        self.assertEqual(mapped["source.position"]["value"], [0.0, 0.0, -250.0])
        self.assertEqual(mapped["source.direction"]["value"], [0.0, 0.0, 1.0])

    def test_slot_mapper_can_switch_both_pipelines_to_v2(self) -> None:
        frame = SlotFrame(
            confidence=0.9,
            geometry=GeometrySlots(kind="cylinder", radius_mm=15, half_length_mm=40),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -60.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        candidate, _ = slot_frame_to_candidates(frame, turn_id=4, geometry_mode=PIPELINE_V2, source_mode=PIPELINE_V2)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_tubs")
        self.assertEqual(mapped["geometry.params.child_rmax"], 15.0)
        self.assertEqual(mapped["source.type"], "point")
        self.assertEqual(mapped["source.position"]["value"], [0.0, 0.0, -60.0])

    def test_slot_mapper_default_remains_legacy(self) -> None:
        selection = select_pipelines()
        self.assertEqual(selection.geometry, PIPELINE_LEGACY)
        self.assertEqual(selection.source, PIPELINE_LEGACY)

    def test_spatial_v2_pipeline_reports_source_relation_for_box(self) -> None:
        frame = SlotFrame(
            confidence=0.9,
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -20.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        result = build_v2_spatial_updates(frame, turn_id=5)
        self.assertEqual(result.geometry_meta["finalization_status"], "ready")
        self.assertEqual(result.source_meta["finalization_status"], "ready")
        self.assertEqual(result.spatial_meta["source_relation"], "outside_target")
        self.assertGreater(result.spatial_meta["surface_distance_mm"], 0.0)

    def test_spatial_v2_pipeline_warns_when_source_inside_target(self) -> None:
        frame = SlotFrame(
            confidence=0.9,
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, 0.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        result = build_v2_spatial_updates(frame, turn_id=6)
        self.assertIn("source_inside_target", result.warnings)
        self.assertEqual(result.spatial_meta["source_relation"], "inside_target")
        self.assertEqual(result.source_meta["finalization_status"], "review")
        self.assertIn("spatial_source_target_conflict", result.source_meta["errors"])
        self.assertEqual(result.source_updates, ())

    def test_spatial_v2_pipeline_blocks_source_on_target_face(self) -> None:
        frame = SlotFrame(
            confidence=0.9,
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -15.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        result = build_v2_spatial_updates(frame, turn_id=7)
        self.assertIn("source_on_target_face", result.warnings)
        self.assertEqual(result.spatial_meta["source_relation"], "on_target_face")
        self.assertEqual(result.source_meta["finalization_status"], "review")
        self.assertEqual(result.source_updates, ())


if __name__ == "__main__":
    unittest.main()

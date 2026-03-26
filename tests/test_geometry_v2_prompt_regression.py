from __future__ import annotations

import unittest
from unittest.mock import patch

from core.contracts.slots import GeometrySlots, MaterialsSlots, SlotFrame
from core.orchestrator.session_manager import process_turn, reset_session
from core.orchestrator.types import Intent
from nlu.llm.slot_frame import LlmSlotBuildResult


class GeometryV2PromptRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session("geometry-v2-regression")

    def _run_with_frame(self, frame: SlotFrame, text: str) -> dict:
        result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text or text,
            confidence=frame.confidence or 1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result):
            return process_turn(
                {
                    "session_id": "geometry-v2-regression",
                    "text": text,
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "legacy",
                    "enable_compare": False,
                },
                ollama_config_path="",
            )

    def test_v2_box_with_complete_dimensions_stays_ready(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10x20x30 copper box",
            target_slots=["geometry.kind", "geometry.size_triplet_mm", "materials.primary"],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            materials=MaterialsSlots(primary="G4_Cu"),
        )
        out = self._run_with_frame(frame, "10 mm x 20 mm x 30 mm copper box target")
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertNotIn("geometry.params.module_x", out["missing_fields"])
        self.assertTrue(out["slot_debug"]["geometry_v2"]["runtime_ready"])

    def test_v2_box_without_size_keeps_geometry_incomplete(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="copper box target",
            target_slots=["geometry.kind", "geometry.size_triplet_mm", "materials.primary"],
            geometry=GeometrySlots(kind="box"),
            materials=MaterialsSlots(primary="G4_Cu"),
        )
        out = self._run_with_frame(frame, "copper box target")
        self.assertNotEqual(out["config"]["geometry"].get("structure"), "single_box")
        self.assertFalse(out["is_complete"])
        self.assertIn("geometry.params.module_x", out["missing_fields"])
        self.assertIn("geometry.params.module_x", out["asked_fields"])
        self.assertFalse(out["slot_debug"]["geometry_v2"]["runtime_ready"])

    def test_v2_cylinder_without_half_length_asks_for_length(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="copper cylinder radius 15 mm",
            target_slots=["geometry.kind", "geometry.radius_mm", "geometry.half_length_mm", "materials.primary"],
            geometry=GeometrySlots(kind="cylinder", radius_mm=15.0),
            materials=MaterialsSlots(primary="G4_Cu"),
        )
        out = self._run_with_frame(frame, "copper cylinder radius 15 mm")
        self.assertNotEqual(out["config"]["geometry"].get("structure"), "single_tubs")
        self.assertIn("geometry.params.child_hz", out["missing_fields"])
        self.assertIn("geometry.params.child_hz", out["asked_fields"])

    def test_v2_unknown_geometry_kind_stays_unresolved(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="ring lattice target",
            target_slots=["geometry.kind"],
            geometry=GeometrySlots(kind="ring_lattice"),
        )
        out = self._run_with_frame(frame, "ring lattice target")
        self.assertFalse(out["is_complete"])
        self.assertIsNone(out["config"]["geometry"].get("structure"))
        self.assertFalse(out["slot_debug"]["geometry_v2"]["compile_ok"])
        self.assertIn("missing_geometry_structure", out["slot_debug"]["geometry_v2"]["errors"])


if __name__ == "__main__":
    unittest.main()

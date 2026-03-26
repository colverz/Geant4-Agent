from __future__ import annotations

import unittest
from unittest.mock import patch

from core.contracts.slots import GeometrySlots, SlotFrame, SourceSlots
from core.orchestrator.session_manager import process_turn, reset_session
from core.orchestrator.types import Intent
from nlu.llm.slot_frame import LlmSlotBuildResult


class PipelineSelectorIntegrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session("selector-test")

    def test_process_turn_can_run_v2_geometry_and_source_pipelines(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="geometry.kind=box; source.kind=point",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -20.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "box with point source",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["pipelines"]["geometry"], "v2")
        self.assertEqual(out["pipelines"]["source"], "v2")
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["particle"], "gamma")
        self.assertIsNone(out["geometry_compare"])
        self.assertIsNone(out["source_compare"])

    def test_process_turn_keeps_legacy_default_when_not_selected(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="geometry.kind=cylinder; source.kind=beam",
            target_slots=[
                "geometry.kind",
                "geometry.radius_mm",
                "geometry.half_length_mm",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
            ],
            geometry=GeometrySlots(kind="cylinder", radius_mm=15.0, half_length_mm=40.0),
            source=SourceSlots(
                kind="beam",
                particle="gamma",
                energy_mev=5.0,
                position_mm=[0.0, 0.0, -250.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "cylinder with beam source",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["pipelines"]["geometry"], "legacy")
        self.assertEqual(out["pipelines"]["source"], "legacy")
        self.assertEqual(out["config"]["geometry"]["structure"], "single_tubs")
        self.assertEqual(out["config"]["source"]["type"], "beam")

    def test_process_turn_v2_geometry_does_not_accept_incomplete_shape_from_side_candidates(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="geometry.kind=box",
            target_slots=["geometry.kind", "geometry.size_triplet_mm"],
            geometry=GeometrySlots(kind="box"),
        )
        result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "make a box target",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "legacy",
                    "enable_compare": False,
                },
                ollama_config_path="",
            )
        self.assertFalse(out["is_complete"])
        self.assertNotEqual(out["config"]["geometry"]["structure"], "single_box")

    def test_process_turn_v2_source_blocks_runtime_unsupported_isotropic(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="source.kind=isotropic; source.particle=gamma; source.energy_mev=0.8; source.position_mm=[0,0,0]",
            target_slots=["source.kind", "source.particle", "source.energy_mev", "source.position_mm"],
            source=SourceSlots(kind="isotropic", particle="gamma", energy_mev=0.8, position_mm=[0.0, 0.0, 0.0]),
        )
        result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "isotropic gamma source at center",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "legacy",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
            )
        self.assertNotEqual(out["config"]["source"].get("type"), "isotropic")
        self.assertFalse(out["is_complete"])

    def test_process_turn_v2_spatial_blocks_source_inside_target(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10 mm box with source at center",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, 0.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "10 mm copper box target with point source at center",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertNotEqual(out["config"]["source"].get("type"), "point")
        self.assertFalse(out["is_complete"])
        self.assertIn("source.position", out["missing_fields"])
        self.assertIn("source.position", out["asked_fields"])
        self.assertIn("spatial_v2", out["slot_debug"])


if __name__ == "__main__":
    unittest.main()

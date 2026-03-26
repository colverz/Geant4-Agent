from __future__ import annotations

import unittest
from unittest.mock import patch

from core.contracts.slots import GeometrySlots, MaterialsSlots, OutputSlots, PhysicsSlots, SlotFrame, SourceSlots
from core.orchestrator.session_manager import process_turn, reset_session
from core.orchestrator.types import Intent
from nlu.llm.slot_frame import LlmSlotBuildResult


class SourceV2PromptRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session("source-v2-regression")

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
                    "session_id": "source-v2-regression",
                    "text": text,
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
            )

    def test_v2_point_source_stays_runtime_ready_for_clear_prompt(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10x20x30 copper box; gamma point source 1 MeV at (0,0,-20) along +z; FTFP_BERT; output json",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
                "physics.explicit_list",
                "output.format",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            materials=MaterialsSlots(primary="G4_Cu"),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -20.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
            physics=PhysicsSlots(explicit_list="FTFP_BERT"),
            output=OutputSlots(format="json"),
        )
        out = self._run_with_frame(frame, "10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm along +z; physics FTFP_BERT; output json")
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["particle"], "gamma")
        self.assertNotIn("source.position", out["missing_fields"])
        self.assertNotIn("source.direction", out["missing_fields"])

    def test_v2_beam_without_direction_forces_direction_question(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10x20x30 copper box; gamma beam 5 MeV from (0,0,-250)",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            materials=MaterialsSlots(primary="G4_Cu"),
            source=SourceSlots(
                kind="beam",
                particle="gamma",
                energy_mev=5.0,
                position_mm=[0.0, 0.0, -250.0],
            ),
        )
        out = self._run_with_frame(frame, "10 mm x 20 mm x 30 mm copper box target; gamma beam 5 MeV from (0,0,-250) mm toward target")
        self.assertNotEqual(out["config"]["source"].get("type"), "beam")
        self.assertIn("source.direction", out["missing_fields"])
        self.assertIn("source.direction", out["asked_fields"])

    def test_v2_isotropic_stays_review_and_never_reaches_runtime_config(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="isotropic gamma source 0.8 MeV at center",
            target_slots=["source.kind", "source.particle", "source.energy_mev", "source.position_mm"],
            source=SourceSlots(kind="isotropic", particle="gamma", energy_mev=0.8, position_mm=[0.0, 0.0, 0.0]),
        )
        out = self._run_with_frame(frame, "isotropic gamma source 0.8 MeV at center")
        self.assertNotEqual(out["config"]["source"].get("type"), "isotropic")
        self.assertFalse(out["is_complete"])

    def test_v2_plane_without_direction_is_blocked(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10x20x30 copper box; plane gamma source 1 MeV at (0,0,-20)",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
            materials=MaterialsSlots(primary="G4_Cu"),
            source=SourceSlots(kind="plane", particle="gamma", energy_mev=1.0, position_mm=[0.0, 0.0, -20.0]),
        )
        out = self._run_with_frame(frame, "10 mm x 20 mm x 30 mm copper box target; plane gamma source 1 MeV at (0,0,-20)")
        self.assertNotEqual(out["config"]["source"].get("type"), "plane")
        self.assertIn("source.direction", out["missing_fields"])
        self.assertIn("source.direction", out["asked_fields"])

    def test_v2_point_source_inside_target_prioritizes_position_review(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10 mm copper box with source at center",
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
        out = self._run_with_frame(frame, "10 mm copper box target with point source at center")
        self.assertIn("source.position", out["missing_fields"])
        self.assertIn("source.position", out["asked_fields"])
        self.assertNotEqual(out["config"]["source"].get("type"), "point")


if __name__ == "__main__":
    unittest.main()

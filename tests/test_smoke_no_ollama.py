from __future__ import annotations

import unittest
from unittest.mock import patch

from core.orchestrator.constraint_ledger import find_lock
from core.orchestrator.semantic_sync import build_semantic_sync_candidate
from core.orchestrator.session_manager import get_or_create_session, process_turn, reset_session
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.slots.slot_frame import GeometrySlots, SlotFrame, SourceSlots
from nlu.llm.slot_frame import LlmSlotBuildResult, parse_slot_payload
from ui.web.server import solve, step


class SmokeNoOllamaTest(unittest.TestCase):
    def test_llm_slot_graph_request_prefers_runtime_graph_geometry_over_single_solid_slot_hint(self) -> None:
        fake_slot = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="geometry kind = box; geometry size triplet mm = [6,6,2]; materials primary = G4_Si; source kind = point; source particle = gamma; source energy mev = 0.511; output format = json",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "output.format",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[6.0, 6.0, 2.0]),
            source=SourceSlots(kind="point", particle="gamma", energy_mev=0.511),
        )
        fake_result = LlmSlotBuildResult(
            ok=True,
            frame=fake_slot,
            normalized_text=fake_slot.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=fake_result):
            out = step(
                {
                    "text": "Create a complete ring setup: 16 modules, each 6 mm x 6 mm x 2 mm, radius 32 mm, clearance 0.5 mm, material G4_Si; point gamma source 0.511 MeV at (0,0,0) mm; physics QBBC; output json.",
                    "lang": "en",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "autofix": True,
                }
            )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "ring")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "ring_modules")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "ring")

    def test_llm_slot_conflicting_polycone_hint_does_not_break_stack_completion(self) -> None:
        fake_slot = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text=(
                "geometry.kind=polycone; geometry.z_planes_mm=[-10,0,10]; geometry.radii_mm=[15,15,15]; "
                "materials.primary=G4_Al; source.kind=beam; source.particle=gamma; source.energy_mev=1.25; "
                "source.position_mm=[0,0,-120]; source.direction_vec=[0,0,1]; physics.explicit_list=Shielding; output.format=csv"
            ),
            target_slots=[
                "geometry.kind",
                "geometry.z_planes_mm",
                "geometry.radii_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
                "physics.explicit_list",
                "output.format",
            ],
            geometry=GeometrySlots(kind="polycone", z_planes_mm=[-10.0, 0.0, 10.0], radii_mm=[15.0, 15.0, 15.0]),
            source=SourceSlots(kind="beam", particle="gamma", energy_mev=1.25, position_mm=[0.0, 0.0, -120.0], direction_vec=[0.0, 0.0, 1.0]),
        )
        fake_result = LlmSlotBuildResult(
            ok=True,
            frame=fake_slot,
            normalized_text=fake_slot.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=fake_result):
            out = step(
                {
                    "text": (
                        "Use stacked layers with footprint 20 mm x 20 mm, thicknesses 2 mm, 4 mm, 6 mm, "
                        "layer clearance 0.5 mm, outer box 30 mm x 30 mm x 20 mm, container clearance 1 mm, "
                        "material G4_Al; beam gamma 1.25 MeV from (0,0,-120) mm along +z; physics Shielding; output csv."
                    ),
                    "lang": "en",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "autofix": True,
                }
            )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "stack")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "stack_in_box")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "stack")
        self.assertNotIn("geometry.graph_program", out.get("missing_fields", []))

    def test_llm_slot_conflicting_box_hint_does_not_break_nest_completion(self) -> None:
        fake_slot = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="geometry.kind=box; geometry.size_triplet_mm=[80,80,80]; materials.primary=G4_Pb; source.kind=point; source.particle=gamma; source.energy_mev=1; output.format=csv",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "output.format",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[80.0, 80.0, 80.0]),
            source=SourceSlots(kind="point", particle="gamma", energy_mev=1.0),
        )
        fake_result = LlmSlotBuildResult(
            ok=True,
            frame=fake_slot,
            normalized_text=fake_slot.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=fake_result):
            out = step(
                {
                    "text": "Outer box 80 mm x 80 mm x 80 mm, inner lead cylinder radius 15 mm, half length 25 mm, clearance 1 mm; point gamma source 1 MeV at (0,0,-100) mm along +z; physics Shielding; output csv.",
                    "lang": "en",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "autofix": True,
                }
            )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "nest")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "nest_box_tubs")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "nest")
        self.assertNotIn("geometry.graph_program", out.get("missing_fields", []))

    def test_llm_slot_conflicting_box_hint_does_not_break_grid_completion(self) -> None:
        fake_slot = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text=(
                "geometry.kind=box; geometry.size_triplet_mm=[12,12,3]; materials.primary=G4_CsI; "
                "source.kind=point; source.particle=gamma; source.energy_mev=0.662; output.format=root"
            ),
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "output.format",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[12.0, 12.0, 3.0]),
            source=SourceSlots(kind="point", particle="gamma", energy_mev=0.662),
        )
        fake_result = LlmSlotBuildResult(
            ok=True,
            frame=fake_slot,
            normalized_text=fake_slot.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=fake_result):
            out = step(
                {
                    "text": "Create a complete grid setup: 3 x 3 module array, each module 12 mm x 12 mm x 3 mm, pitch_x 15 mm, pitch_y 15 mm, clearance 1 mm, material G4_CsI; point gamma source 0.662 MeV at (0,0,-90) mm along +z; physics QBBC; output root.",
                    "lang": "en",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "autofix": True,
                }
            )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "grid")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "grid_modules")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "grid")

    def test_llm_slot_conflicting_box_hint_does_not_break_shell_completion(self) -> None:
        fake_slot = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text=(
                "geometry.kind=box; geometry.size_triplet_mm=[70,70,70]; materials.primary=G4_Fe; "
                "source.kind=isotropic; source.particle=gamma; source.energy_mev=0.8; output.format=hdf5"
            ),
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "output.format",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[70.0, 70.0, 70.0]),
            source=SourceSlots(kind="isotropic", particle="gamma", energy_mev=0.8),
        )
        fake_result = LlmSlotBuildResult(
            ok=True,
            frame=fake_slot,
            normalized_text=fake_slot.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=fake_result):
            out = step(
                {
                    "text": "Create a complete shell configuration: inner radius 12 mm, thicknesses 4 mm and 6 mm, half length 35 mm, nested core radius 8 mm, nested core half length 25 mm, clearance 1 mm, material G4_Fe; isotropic gamma source 0.8 MeV at (0,0,0) mm; physics QBBC; output hdf5.",
                    "lang": "en",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "autofix": True,
                }
            )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "shell")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "shell_nested")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "shell")

    def test_llm_slot_conflicting_box_hint_does_not_break_boolean_completion(self) -> None:
        fake_slot = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text=(
                "geometry.kind=box; geometry.size_triplet_mm=[30,30,20]; materials.primary=G4_Al; "
                "source.kind=beam; source.particle=gamma; source.energy_mev=0.5; output.format=json"
            ),
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "output.format",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[30.0, 30.0, 20.0]),
            source=SourceSlots(kind="beam", particle="gamma", energy_mev=0.5),
        )
        fake_result = LlmSlotBuildResult(
            ok=True,
            frame=fake_slot,
            normalized_text=fake_slot.normalized_text,
            confidence=1.0,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
            stage_trace={"final_status": "ok"},
        )
        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=fake_result):
            out = step(
                {
                    "text": "Construct a boolean union: A is a 30 mm x 30 mm x 20 mm box, B is a 10 mm x 10 mm x 40 mm box, union B with A; material G4_Al; beam gamma 0.5 MeV from (0,0,-70) mm along +z; physics QBBC; output json.",
                    "lang": "en",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "autofix": True,
                }
            )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "boolean")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "boolean_union_boxes")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "boolean")

    def test_solve_without_ollama(self) -> None:
        out = solve(
            {
                "text": "ring of 12 modules, radius 40 mm, module 8x10x2 mm, clearance 1 mm",
                "min_confidence": 0.6,
                "normalize_input": False,
                "llm_fill_missing": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", out)
        self.assertIn("structure", out)
        self.assertIn("synthesis", out)

    def test_step_without_ollama(self) -> None:
        out = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", out)
        self.assertIn("missing_fields", out)
        self.assertIn("assistant_message", out)
        self.assertIn("phase", out)
        self.assertIn("phase_title", out)
        self.assertIn("dialogue_action", out)
        self.assertIn("dialogue_trace", out)
        self.assertIn("dialogue_summary", out)
        self.assertIn("dialogue_memory", out)
        self.assertIn("raw_dialogue", out)
        self.assertIn("asked_fields", out)
        self.assertIn("asked_fields_friendly", out)
        self.assertIn("answered_this_turn", out)
        self.assertIn("open_questions", out)
        self.assertIn("question_attempts", out)
        self.assertIn("llm_used", out)
        self.assertIn("fallback_reason", out)
        self.assertIn("llm_stage_failures", out)
        self.assertEqual(out.get("inference_backend"), "runtime_semantic")
        self.assertFalse(out.get("is_complete", False))
        self.assertEqual(out.get("dialogue_action"), "summarize_progress")
        self.assertEqual(out.get("dialogue_summary", {}).get("status"), "pending")
        self.assertGreaterEqual(len(out.get("dialogue_memory", [])), 1)
        self.assertGreaterEqual(len(out.get("raw_dialogue", [])), 2)

    def test_schema_closure_progress(self) -> None:
        first = step(
            {
                "text": "ring of 12 modules, radius 40 mm, module 8x10x2 mm, clearance 1 mm, material G4_Cu, gamma source, FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", first)
        sid = first["session_id"]
        m1 = set(first.get("missing_fields", []))
        self.assertIn(first.get("phase"), {"geometry", "materials", "source", "physics", "output", "finalize"})
        self.assertIn("source.energy", m1)
        self.assertIn("source.position", m1)
        self.assertIn("source.direction", m1)

        second = step(
            {
                "session_id": sid,
                "text": "energy 2 MeV, position (0,0,-100), direction (0,0,1)",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", second)
        m2 = set(second.get("missing_fields", []))
        self.assertNotIn("source.energy", m2)
        self.assertNotIn("source.position", m2)
        self.assertNotIn("source.direction", m2)
        self.assertIn("config", second)

    def test_ring_request_preserves_graph_program_in_main_flow(self) -> None:
        out = step(
            {
                "text": "ring of 12 modules, radius 40 mm, module 8x10x2 mm, clearance 1 mm, material G4_Cu, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "ring")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "ring_modules")
        self.assertEqual(
            out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"),
            "ring",
        )

    def test_ring_request_missing_radius_uses_family_aware_prompt_paths(self) -> None:
        out = step(
            {
                "text": "ring of 12 modules, module 8x10x2 mm, material G4_Cu, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertIn("geometry.ask.ring.radius", out.get("missing_fields", []))
        self.assertIn("ring radius", out.get("asked_fields_friendly", []))

    def test_grid_request_preserves_graph_program_in_main_flow(self) -> None:
        out = step(
            {
                "text": "grid of modules, nx 3, ny 4, module 8x10x2 mm, pitch_x 12 mm, pitch_y 14 mm, clearance 1 mm, material G4_Cu, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "grid")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "grid_modules")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "grid")

    def test_ring_request_with_missing_module_size_does_not_commit_sampled_defaults(self) -> None:
        out = step(
            {
                "text": "ring of 12 modules, radius 40 mm, clearance 1 mm, material G4_Cu, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        geometry = out.get("config", {}).get("geometry", {})
        self.assertEqual(geometry.get("structure"), "ring")
        self.assertEqual(geometry.get("chosen_skeleton"), "ring_modules")
        self.assertEqual(geometry.get("graph_program", {}).get("root"), "ring")
        self.assertEqual(geometry.get("params", {}).get("n"), 12)
        self.assertEqual(geometry.get("params", {}).get("radius"), 40.0)
        self.assertEqual(geometry.get("params", {}).get("clearance"), 1.0)
        self.assertNotIn("module_x", geometry.get("params", {}))
        self.assertIn("geometry.ask.ring.module_size", out.get("missing_fields", []))

    def test_chinese_grid_intent_keeps_graph_family_when_params_are_missing(self) -> None:
        out = step(
            {
                "text": "我想做一个二维阵列探测板，但 pitch 和材料还没定。",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": False,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "grid")
        self.assertIn("geometry.ask.grid.pitch_x", out.get("missing_fields", []))

    def test_chinese_nest_intent_keeps_graph_family_when_params_are_missing(self) -> None:
        out = step(
            {
                "text": "我想在一个盒子里嵌一个圆柱靶，先别急着补全。",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": False,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "nest")
        self.assertIn("geometry.ask.nest.parent_size", out.get("missing_fields", []))

    def test_chinese_boolean_intent_keeps_graph_family_when_only_intent_is_given(self) -> None:
        out = step(
            {
                "text": "我想做一个挖空的盒体几何，但先只给你意图。",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": False,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "boolean")
        self.assertIn("geometry.ask.boolean.solid_a_size", out.get("missing_fields", []))

    def test_stack_request_preserves_graph_program_in_main_flow(self) -> None:
        out = step(
            {
                "text": "stack of layers along z, x 20 mm, y 20 mm, thicknesses 2 3 4 mm, clearance 0.5 mm, material G4_Si, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "stack")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "stack_in_box")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "stack")

    def test_chinese_grid_request_preserves_graph_program_in_main_flow(self) -> None:
        out = step(
            {
                "text": "请创建完整的阵列配置：3 x 3 盒体阵列，每个模块 12 mm x 12 mm x 3 mm，pitch_x 15 mm，pitch_y 15 mm，间隙 1 mm，材料 G4_CsI；点源 gamma 0.662 MeV，位置 (0,0,-90) mm，方向 +z；物理列表 QBBC；输出 root。",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": False,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "grid")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "grid_modules")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "grid")

    def test_chinese_nest_request_preserves_graph_program_in_main_flow(self) -> None:
        out = step(
            {
                "text": "请创建完整的嵌套配置：外盒 80 mm x 80 mm x 80 mm，内嵌铅圆柱半径 15 mm、半长 25 mm，嵌套间隙 1 mm；材料 G4_Pb；点源 gamma 1 MeV，位置 (0,0,-100) mm，方向 +z；物理列表 Shielding；输出 csv。",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": False,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "nest")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "nest_box_tubs")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"), "nest")

    def test_boolean_request_preserves_graph_program_in_main_flow(self) -> None:
        out = step(
            {
                "text": "union of two boxes, 10x20x30 mm and 5x6x7 mm, material G4_Al, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "boolean")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "boolean_union_boxes")
        self.assertEqual(
            out.get("config", {}).get("geometry", {}).get("graph_program", {}).get("root"),
            "boolean",
        )

    def test_shell_request_with_missing_child_geometry_keeps_graph_family_without_sampled_defaults(self) -> None:
        out = step(
            {
                "text": "concentric shell with inner radius 15 mm, thicknesses 5 mm and 8 mm, half length 40 mm, material G4_Pb, gamma point source, 1 MeV, direction (0,0,1), physics Shielding, output json",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        geometry = out.get("config", {}).get("geometry", {})
        self.assertEqual(geometry.get("structure"), "shell")
        self.assertEqual(geometry.get("chosen_skeleton"), "shell_nested")
        self.assertEqual(geometry.get("graph_program", {}).get("root"), "shell")
        self.assertEqual(geometry.get("params", {}).get("inner_r"), 15.0)
        self.assertEqual(geometry.get("params", {}).get("th1"), 5.0)
        self.assertEqual(geometry.get("params", {}).get("th2"), 8.0)
        self.assertEqual(geometry.get("params", {}).get("hz"), 40.0)
        self.assertNotIn("th3", geometry.get("params", {}))
        self.assertNotIn("child_rmax", geometry.get("params", {}))
        self.assertNotIn("geometry.ask.shell.child_radius", out.get("missing_fields", []))
        self.assertNotIn("geometry.ask.shell.child_half_length", out.get("missing_fields", []))
        self.assertNotIn("geometry.ask.shell.clearance", out.get("missing_fields", []))
        self.assertIn("source.position", out.get("missing_fields", []))

    def test_boolean_minus_request_prefers_subtraction_skeleton(self) -> None:
        out = step(
            {
                "text": "10 by 20 by 30 mm box minus 5 by 6 by 7 mm box, material G4_Al, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "boolean")
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("chosen_skeleton"), "boolean_subtraction_boxes")

    def test_semantic_sync_can_infer_structure_from_graph_root(self) -> None:
        config = {
            "geometry": {
                "graph_program": {
                    "nodes": [
                        {"id": "module", "type": "Box", "x": 8.0, "y": 10.0, "z": 2.0},
                        {"id": "ring", "type": "Ring", "module": "module", "n": 12, "radius": 40.0, "clearance": 1.0},
                    ],
                    "root": "ring",
                    "constraints": [],
                },
                "chosen_skeleton": "ring_modules",
                "root_name": "",
                "params": {},
            },
            "materials": {"selected_materials": ["G4_Cu"]},
            "source": {},
            "physics": {},
            "output": {"format": "root"},
        }
        sync_candidate = build_semantic_sync_candidate(config, turn_id=1, recent_updates=None)
        self.assertIsNotNone(sync_candidate)
        assert sync_candidate is not None
        mapped = {u.path: u.value for u in sync_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "ring")
        self.assertEqual(mapped["geometry.root_name"], "ring")

    def test_non_english_fallback_observable(self) -> None:
        out = step(
            {
                "text": "\u6211\u60f3\u8981\u4e00\u4e2a1m\u7684\u94dc\u7acb\u65b9\u4f53\u548cgamma\u6e90",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
            }
        )
        self.assertNotIn("error", out)
        self.assertEqual(out.get("llm_used"), False)
        self.assertEqual(out.get("fallback_reason"), "E_LLM_DISABLED")

    def test_chinese_payload_sets_source_type_without_reask(self) -> None:
        out = step(
            {
                "text": "\u94dc\u7acb\u65b9\u4f53 1m x 1m x 1m\uff0cGamma \u70b9\u6e90\uff0c1 MeV\uff0c\u4f4d\u7f6e\u5728\u4e2d\u5fc3\uff0c\u65b9\u5411 +z\uff0c\u7269\u7406 FTFP_BERT\uff0c\u8f93\u51fa json",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotIn("error", out)
        self.assertEqual(out.get("config", {}).get("source", {}).get("type"), "point")
        self.assertNotIn("source.type", out.get("missing_fields", []))

    def test_router_flag_is_observable_in_strict_path(self) -> None:
        out = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": True,
                "autofix": True,
            }
        )
        self.assertNotIn("error", out)
        self.assertEqual(out.get("llm_used"), False)
        self.assertEqual(out.get("fallback_reason"), "E_LLM_ROUTER_DISABLED")

    def test_question_turn_cannot_implicitly_overwrite(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target, gamma point source, energy 1 MeV, position (0,0,0), direction +z, physics FTFP_BERT, output json",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]
        self.assertEqual(first.get("config", {}).get("source", {}).get("particle"), "gamma")

        second = step(
            {
                "session_id": sid,
                "text": "what if electron?",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("config", {}).get("source", {}).get("particle"), "gamma")
        reason_codes = {x.get("reason_code") for x in second.get("rejected_updates", [])}
        self.assertIn("E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT", reason_codes)

    def test_explicit_overwrite_requires_confirmation(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]
        mats1 = first.get("config", {}).get("materials", {}).get("selected_materials", [])
        self.assertIn("G4_Cu", mats1)

        second = step(
            {
                "session_id": sid,
                "text": "change material to G4_Al",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")
        mats2 = second.get("config", {}).get("materials", {}).get("selected_materials", [])
        self.assertEqual(mats2, ["G4_Cu"])
        self.assertEqual(len(second.get("dialogue_trace", {}).get("overwrite_preview", [])), 1)
        state = get_or_create_session(sid)
        material_lock = find_lock(state.constraint_ledger, "materials.selected_materials")
        self.assertIsNotNone(material_lock)
        self.assertTrue(material_lock.locked)
        self.assertEqual(material_lock.value, ["G4_Cu"])

        third = step(
            {
                "session_id": sid,
                "text": "status",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(third.get("dialogue_action"), "confirm_overwrite")
        self.assertEqual(third.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Cu"])
        self.assertEqual(len(third.get("dialogue_trace", {}).get("overwrite_preview", [])), 1)

        fourth = step(
            {
                "session_id": sid,
                "text": "confirm",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        mats4 = fourth.get("config", {}).get("materials", {}).get("selected_materials", [])
        self.assertEqual(mats4, ["G4_Al"])
        self.assertEqual(
            fourth.get("config", {}).get("materials", {}).get("volume_material_map"),
            {"box": "G4_Al"},
        )

    def test_pending_overwrite_marks_session_incomplete_until_confirmed(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target, gamma point source, energy 1 MeV, position (0,0,-100), direction +z, physics FTFP_BERT, output csv",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]
        self.assertTrue(first.get("is_complete"))
        self.assertEqual(first.get("config", {}).get("output", {}).get("format"), "csv")

        second = step(
            {
                "session_id": sid,
                "text": "change output to root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")
        self.assertFalse(second.get("is_complete"))
        self.assertTrue(second.get("pending_overwrite_required"))
        self.assertEqual(second.get("config", {}).get("output", {}).get("format"), "csv")

        third = step(
            {
                "session_id": sid,
                "text": "confirm",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(third.get("config", {}).get("output", {}).get("format"), "root")
        self.assertTrue(third.get("is_complete"))
        self.assertFalse(third.get("pending_overwrite_required"))
        self.assertEqual(third.get("dialogue_action"), "finalize")

    def test_reject_pending_overwrite_keeps_existing_value(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target, gamma point source, energy 1 MeV, position (0,0,-100), direction +z, physics FTFP_BERT, output csv",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]

        second = step(
            {
                "session_id": sid,
                "text": "change output to root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")

        third = step(
            {
                "session_id": sid,
                "text": "keep existing",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(third.get("dialogue_action"), "reject_overwrite")
        self.assertFalse(third.get("pending_overwrite_required"))
        self.assertEqual(third.get("config", {}).get("output", {}).get("format"), "csv")

    def test_chinese_confirmation_turn_is_recognized(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]

        second = step(
            {
                "session_id": sid,
                "text": "\u628a\u6750\u6599\u6539\u6210 G4_Al\u3002",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")
        self.assertEqual(second.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Cu"])

        third = step(
            {
                "session_id": sid,
                "text": "\u786e\u8ba4",
                "lang": "zh",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(third.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Al"])

    def test_narrow_output_turn_does_not_trigger_unrelated_overwrite(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target, gamma point source, energy 1 MeV, position (0,0,-20), direction (0,0,1), physics FTFP_BERT",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]

        second = step(
            {
                "session_id": sid,
                "text": "Output json.",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertNotEqual(second.get("dialogue_action"), "confirm_overwrite")
        self.assertEqual(second.get("dialogue_trace", {}).get("overwrite_preview", []), [])
        self.assertEqual(
            second.get("config", {}).get("source", {}).get("position"),
            {"type": "vector", "value": [0.0, 0.0, -20.0]},
        )
        self.assertEqual(second.get("config", {}).get("output", {}).get("format"), "json")

    def test_pending_overwrite_blocks_only_conflicting_fields(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]

        second = step(
            {
                "session_id": sid,
                "text": "change material to G4_Al",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")
        self.assertEqual(second.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Cu"])

        third = step(
            {
                "session_id": sid,
                "text": "Output json.",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(third.get("dialogue_action"), "confirm_overwrite")
        self.assertEqual(third.get("config", {}).get("output", {}).get("format"), "json")
        self.assertEqual(third.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Cu"])
        self.assertEqual(
            third.get("dialogue_trace", {}).get("overwrite_preview", [])[0].get("path"),
            "materials.selected_materials",
        )

        fourth = step(
            {
                "session_id": sid,
                "text": "confirm",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(fourth.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Al"])
        self.assertEqual(fourth.get("config", {}).get("output", {}).get("format"), "json")

    def test_pending_overwrite_can_refresh_same_field_before_confirmation(self) -> None:
        first = step(
            {
                "text": "1m x 1m x 1m copper box target with gamma source",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]

        second = step(
            {
                "session_id": sid,
                "text": "change material to G4_Al",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")

        third = step(
            {
                "session_id": sid,
                "text": "change material to G4_Pb",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(third.get("dialogue_action"), "confirm_overwrite")
        self.assertEqual(third.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Cu"])
        self.assertEqual(
            third.get("dialogue_trace", {}).get("overwrite_preview", [])[0].get("new"),
            ["G4_Pb"],
        )

        fourth = step(
            {
                "session_id": sid,
                "text": "confirm",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(fourth.get("config", {}).get("materials", {}).get("selected_materials", []), ["G4_Pb"])

    def test_unrelated_turn_does_not_clear_graph_semantic_missing_paths(self) -> None:
        first = step(
            {
                "text": "ring of 12 modules, module 8x10x2 mm, material G4_Cu, gamma point source, 1 MeV, direction (0,0,1), physics FTFP_BERT, output root",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        sid = first["session_id"]
        self.assertIn("geometry.ask.ring.radius", first.get("missing_fields", []))

        second = step(
            {
                "session_id": sid,
                "text": "output json",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertIn("geometry.ask.ring.radius", second.get("missing_fields", []))
        self.assertTrue(
            any("ring " in label for label in second.get("asked_fields_friendly", [])),
        )

    def test_llm_slot_first_bridges_into_runtime_extractor(self) -> None:
        sid = "llm-slot-bridge-test"
        reset_session(sid)
        payload = {
            "intent": "SET",
            "confidence": 0.95,
            "normalized_text": "geometry.size:1m,1m,1m; materials.primary:copper; source.particle:gamma; physics.explicit_list:FTFP_BERT; output.format:root",
            "target_slots": [
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.particle",
                "physics.explicit_list",
                "output.format",
            ],
            "slots": {
                "geometry": {"kind": "box", "size_triplet_mm": "1m x 1m x 1m"},
                "materials": {"primary": "copper"},
                "source": {"particle": "gamma"},
                "physics": {"explicit_list": "FTFP_BERT"},
                "output": {"format": "root"},
            },
        }
        frame, meta = parse_slot_payload(payload)
        assert frame is not None
        slot_result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=str(meta.get("normalized_text", "")),
            confidence=float(meta.get("confidence", 0.95)),
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=list(meta.get("schema_errors", [])),
        )

        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=slot_result), patch(
            "core.orchestrator.session_manager.build_llm_semantic_frame"
        ) as semantic_mock, patch("core.orchestrator.session_manager.recommend_physics_list", return_value=None):
            out = process_turn(
                {
                    "session_id": sid,
                    "text": "1m x 1m x 1m copper box target with gamma source, physics FTFP_BERT, output root",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                },
                ollama_config_path="",
                lang="en",
            )

        semantic_mock.assert_not_called()
        config = out.get("config", {})
        self.assertEqual(config.get("geometry", {}).get("structure"), "single_box")
        self.assertEqual(config.get("geometry", {}).get("params", {}).get("module_x"), 1000.0)
        self.assertEqual(config.get("geometry", {}).get("params", {}).get("module_y"), 1000.0)
        self.assertEqual(config.get("geometry", {}).get("params", {}).get("module_z"), 1000.0)
        self.assertEqual(config.get("materials", {}).get("selected_materials"), ["G4_Cu"])
        self.assertEqual(config.get("source", {}).get("particle"), "gamma")
        self.assertEqual(config.get("physics", {}).get("physics_list"), "FTFP_BERT")
        self.assertEqual(config.get("output", {}).get("format"), "root")
        self.assertEqual(out.get("inference_backend"), "llm_slot_frame+runtime_semantic")
        self.assertEqual(out.get("llm_stage_failures"), [])
        self.assertEqual(out.get("slot_debug", {}).get("final_status"), "ok")

    def test_llm_slot_first_allows_extractor_to_fill_empty_collections(self) -> None:
        sid = "llm-slot-collection-fill-test"
        reset_session(sid)
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=0.9,
            normalized_text="geometry.size:1m,1m,1m; materials.target:copper; source.particle:gamma",
            target_slots=["geometry.kind", "geometry.size_triplet_mm", "materials.primary", "source.particle"],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[1000.0, 1000.0, 1000.0]),
            source=SourceSlots(particle="gamma"),
        )
        slot_result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=frame.confidence,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
        )

        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=slot_result), patch(
            "core.orchestrator.session_manager.build_llm_semantic_frame"
        ) as semantic_mock, patch("core.orchestrator.session_manager.recommend_physics_list", return_value=None):
            out = process_turn(
                {
                    "session_id": sid,
                    "text": "1m x 1m x 1m copper box target with gamma source",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                },
                ollama_config_path="",
                lang="en",
            )

        semantic_mock.assert_not_called()
        self.assertEqual(out.get("config", {}).get("materials", {}).get("selected_materials"), ["G4_Cu"])
        reason_codes = {x.get("reason_code") for x in out.get("rejected_updates", [])}
        self.assertNotIn("E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT", reason_codes)

    def test_slot_anchor_filters_duplicate_extractor_updates(self) -> None:
        sid = "llm-slot-duplicate-filter"
        reset_session(sid)
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=0.9,
            normalized_text="geometry.kind:box; geometry.size:1m,1m,1m; source.particle:gamma",
            target_slots=["geometry.kind", "geometry.size_triplet_mm", "source.particle"],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[1000.0, 1000.0, 1000.0]),
            source=SourceSlots(particle="gamma"),
        )
        slot_result = LlmSlotBuildResult(
            ok=True,
            frame=frame,
            normalized_text=frame.normalized_text,
            confidence=frame.confidence,
            llm_raw="{}",
            fallback_reason=None,
            schema_errors=[],
        )
        duplicate_candidate = CandidateUpdate(
            producer=Producer.BERT_EXTRACTOR,
            intent=Intent.SET,
            target_paths=["geometry.params.module_x", "source.energy"],
            updates=[
                UpdateOp(
                    path="geometry.params.module_x",
                    op="set",
                    value=1000.0,
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.7,
                    turn_id=1,
                ),
                UpdateOp(
                    path="source.energy",
                    op="set",
                    value=1.0,
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.7,
                    turn_id=1,
                ),
            ],
            confidence=0.7,
            rationale="duplicate_test",
        )

        with patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=slot_result), patch(
            "core.orchestrator.session_manager.extract_candidates_from_normalized_text",
            return_value=(duplicate_candidate, {"graph_candidates": [], "inference_backend": "runtime_semantic"}),
        ), patch("core.orchestrator.session_manager.build_llm_semantic_frame") as semantic_mock, patch(
            "core.orchestrator.session_manager.recommend_physics_list", return_value=None
        ):
            out = process_turn(
                {
                    "session_id": sid,
                    "text": "1m x 1m x 1m box target with gamma source",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                },
                ollama_config_path="",
                lang="en",
            )

        semantic_mock.assert_not_called()
        rejected_paths = {(x.get("path"), x.get("producer")) for x in out.get("rejected_updates", [])}
        self.assertNotIn(("geometry.params.module_x", "bert_extractor"), rejected_paths)
        self.assertIsNone(out.get("config", {}).get("source", {}).get("energy"))

    def test_explicit_physics_choice_blocks_recommender_override(self) -> None:
        sid = "explicit-physics-choice"
        reset_session(sid)
        with patch(
            "core.orchestrator.session_manager.recommend_physics_list",
            return_value=CandidateUpdate(
                producer=Producer.LLM_RECOMMENDER,
                intent=Intent.SET,
                target_paths=["physics.physics_list"],
                updates=[
                    UpdateOp(
                        path="physics.physics_list",
                        op="set",
                        value="FTFP_BERT",
                        producer=Producer.LLM_RECOMMENDER,
                        confidence=0.8,
                        turn_id=1,
                    )
                ],
                confidence=0.8,
                rationale="forced_test_recommender",
            ),
        ):
            out = process_turn(
                {
                    "session_id": sid,
                    "text": "Use physics list QBBC and output root.",
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": False,
                },
                ollama_config_path="",
                lang="en",
            )
        self.assertEqual(out.get("config", {}).get("physics", {}).get("physics_list"), "QBBC")
        rejected_producers = {item.get("producer") for item in out.get("rejected_updates", [])}
        self.assertNotIn("llm_recommender", rejected_producers)

    def test_full_stack_prompt_can_close_without_parent_box(self) -> None:
        out = step(
            {
                "text": "Stack three steel slabs along z with x 100 mm and y 100 mm, thicknesses 4 mm, 8 mm, 12 mm, clearance 1 mm; gamma beam 5 MeV from (0,0,-250) mm along +z; physics Shielding; output csv.",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "stack")
        self.assertTrue(out.get("is_complete"))

    def test_full_shell_prompt_can_close_without_nested_child_defaults(self) -> None:
        out = step(
            {
                "text": "Shell geometry with inner radius 60 mm, thicknesses 4 mm, 4 mm, 4 mm, half length 100 mm, material copper; point electron source 2 MeV at center; FTFP_BERT; output json.",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "shell")
        self.assertTrue(out.get("is_complete"))

    def test_boolean_subtraction_prefers_solid_material_over_air_hole(self) -> None:
        out = step(
            {
                "text": "Create a subtraction boolean: subtract a 20 mm x 20 mm x 20 mm air box from a 100 mm x 60 mm x 40 mm iron box; point gamma 1.17 MeV at (0,0,-120) mm toward +z; physics QBBC; output json.",
                "lang": "en",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "autofix": True,
            }
        )
        self.assertEqual(out.get("config", {}).get("geometry", {}).get("structure"), "boolean")
        self.assertEqual(out.get("config", {}).get("materials", {}).get("selected_materials"), ["G4_Fe"])


if __name__ == "__main__":
    unittest.main()

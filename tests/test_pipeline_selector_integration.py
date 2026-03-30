from __future__ import annotations

import unittest
from unittest.mock import patch

from core.contracts.slots import GeometrySlots, SlotFrame, SourceSlots
from core.orchestrator.session_manager import get_or_create_session, process_turn, reset_session
from core.orchestrator.types import Intent
from core.interpreter import (
    EvidenceSpan,
    GeometryCandidate,
    InterpreterParseResult,
    InterpreterRunResult,
    SourceCandidate,
    TurnSummary,
)
from nlu.llm.slot_frame import LlmSlotBuildResult


class PipelineSelectorIntegrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session("selector-test")
        reset_session("selector-memory-test")

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

    def test_process_turn_can_attach_interpreter_sidecar(self) -> None:
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
        interpreter_result = InterpreterRunResult(
            ok=True,
            parsed=InterpreterParseResult(
                ok=True,
                turn_summary=TurnSummary(intent="set", explicit_domains=["geometry", "source"]),
                geometry_candidate=GeometryCandidate(
                    kind_candidate="box",
                    confidence=0.9,
                    evidence_spans=[EvidenceSpan(text="box", role="geometry")],
                ),
                source_candidate=SourceCandidate(
                    source_type_candidate="point",
                    particle_candidate="gamma",
                    confidence=0.9,
                    evidence_spans=[EvidenceSpan(text="point source", role="source_type")],
                ),
                raw_payload={},
                error=None,
            ),
            llm_raw="{}",
            cleaned_text="{}",
            fallback_reason=None,
        )
        with (
            patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result),
            patch("core.orchestrator.session_manager.run_interpreter", return_value=interpreter_result),
        ):
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
                    "enable_interpreter": True,
                },
                ollama_config_path="",
            )
        self.assertIsNotNone(out["interpreter_debug"])
        self.assertTrue(out["interpreter_debug"]["ok"])
        self.assertEqual(out["interpreter_debug"]["geometry_candidate"]["kind_candidate"], "box")
        self.assertEqual(out["interpreter_debug"]["merged"]["merged_source"]["source_type"]["value"], "point")

    def test_process_turn_can_use_interpreter_sidecar_as_source_bridge(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="geometry.kind=box",
            target_slots=["geometry.kind", "geometry.size_triplet_mm"],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
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
        interpreter_debug = {
            "ok": True,
            "turn_summary": {"intent": "set", "explicit_domains": ["geometry", "source"]},
            "geometry_candidate": {"kind_candidate": "box", "confidence": 0.9},
            "source_candidate": {"source_type_candidate": "point", "particle_candidate": "gamma", "confidence": 0.9},
            "merged": {
                "merged_geometry": {
                    "kind": {"value": "box", "chosen_from": "llm", "confidence": 0.9, "conflict": False, "note": ""},
                    "material": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "dimensions": {},
                    "ambiguities": [],
                },
                "merged_source": {
                    "source_type": {"value": "point", "chosen_from": "llm", "confidence": 0.9, "conflict": False, "note": ""},
                    "particle": {"value": "gamma", "chosen_from": "llm", "confidence": 0.9, "conflict": False, "note": ""},
                    "energy_mev": {"value": 1.0, "chosen_from": "llm", "confidence": 0.9, "conflict": False, "note": ""},
                    "position": {"value": {"position_mm": [0.0, 0.0, -20.0]}, "chosen_from": "llm", "confidence": 0.9, "conflict": False, "note": ""},
                    "direction": {
                        "value": {"mode": "explicit_vector", "hint": {"direction_vec": [0.0, 0.0, 1.0]}},
                        "chosen_from": "llm",
                        "confidence": 0.9,
                        "conflict": False,
                        "note": "",
                    },
                    "ambiguities": [],
                },
                "open_questions": [],
                "conflicts": [],
                "trust_report": {},
            },
        }
        with (
            patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result),
            patch("core.orchestrator.session_manager._build_interpreter_sidecar", return_value=interpreter_debug),
        ):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "box target with gamma point source",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": True,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["particle"], "gamma")
        self.assertEqual(out["slot_debug"]["interpreter_source"]["used"], True)

    def test_process_turn_interpreter_can_fill_geometry_material_without_overriding_shape(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10 mm box target",
            target_slots=["geometry.kind", "geometry.size_triplet_mm"],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
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
        interpreter_debug = {
            "ok": True,
            "turn_summary": {"intent": "set", "explicit_domains": ["geometry"]},
            "geometry_candidate": {"kind_candidate": "box", "material_candidate": "G4_Cu", "confidence": 0.9},
            "source_candidate": {},
            "merged": {
                "merged_geometry": {
                    "kind": {"value": "box", "chosen_from": "shared", "confidence": 0.9, "conflict": False, "note": ""},
                    "material": {"value": "G4_Cu", "chosen_from": "llm", "confidence": 0.9, "conflict": False, "note": ""},
                    "dimensions": {
                        "size_triplet_mm": {
                            "value": [10.0, 10.0, 10.0],
                            "chosen_from": "shared",
                            "confidence": 0.9,
                            "conflict": False,
                            "note": "",
                        }
                    },
                    "ambiguities": [],
                },
                "merged_source": {
                    "source_type": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "particle": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "energy_mev": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "position": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "direction": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "ambiguities": [],
                },
                "open_questions": [],
                "conflicts": [],
                "trust_report": {},
            },
        }
        with (
            patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result),
            patch("core.orchestrator.session_manager._build_interpreter_sidecar", return_value=interpreter_debug),
        ):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "10 mm box copper target",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "legacy",
                    "enable_compare": False,
                    "enable_interpreter": True,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["materials"]["selected_materials"], ["G4_Cu"])
        self.assertTrue(out["slot_debug"]["interpreter_geometry"]["used"])

    def test_process_turn_interpreter_geometry_bridge_prefers_resolved_geometry_draft(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10 mm target",
            target_slots=[],
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
        interpreter_debug = {
            "ok": True,
            "turn_summary": {"intent": "set", "explicit_domains": ["geometry"]},
            "geometry_candidate": {"kind_candidate": "slab", "material_candidate": None, "confidence": 0.7},
            "source_candidate": {},
            "merged": {
                "merged_geometry": {
                    "kind": {"value": "slab", "chosen_from": "llm", "confidence": 0.7, "conflict": False, "note": ""},
                    "material": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "dimensions": {
                        "thickness_mm": {
                            "value": 10.0,
                            "chosen_from": "llm",
                            "confidence": 0.7,
                            "conflict": False,
                            "note": "",
                        }
                    },
                    "ambiguities": [],
                },
                "merged_source": {
                    "source_type": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "particle": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "energy_mev": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "position": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "direction": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "ambiguities": [],
                },
                "open_questions": [],
                "conflicts": [],
                "trust_report": {},
            },
            "geometry_resolution": {
                "draft": {
                    "structure": "single_box",
                    "material": "G4_Cu",
                    "params": {"size_triplet_mm": [10.0, 10.0, 10.0]},
                    "conflicts": [],
                    "ambiguities": [],
                    "open_questions": [],
                    "trust_report": {},
                    "bridge_allowed": True,
                },
                "intent": {
                    "structure": "single_box",
                    "kind": "single_box",
                    "params": {"size_triplet_mm": [10.0, 10.0, 10.0]},
                    "missing_fields": [],
                    "ambiguities": [],
                },
            },
        }
        with (
            patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result),
            patch("core.orchestrator.session_manager._build_interpreter_sidecar", return_value=interpreter_debug),
        ):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "10 mm copper target",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "legacy",
                    "enable_compare": False,
                    "enable_interpreter": True,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["materials"]["selected_materials"], ["G4_Cu"])
        self.assertTrue(out["slot_debug"]["interpreter_geometry"]["used_resolution"])

    def test_process_turn_interpreter_can_fill_missing_beam_direction(self) -> None:
        frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="10 mm box target with gamma beam from (0,0,-50) mm",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
            source=SourceSlots(
                kind="beam",
                particle="gamma",
                energy_mev=2.0,
                position_mm=[0.0, 0.0, -50.0],
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
        interpreter_debug = {
            "ok": True,
            "turn_summary": {"intent": "set", "explicit_domains": ["geometry", "source"]},
            "geometry_candidate": {"kind_candidate": "box", "confidence": 0.9},
            "source_candidate": {
                "source_type_candidate": "beam",
                "particle_candidate": "gamma",
                "energy_candidate_mev": 2.0,
                "confidence": 0.9,
            },
            "merged": {
                "merged_geometry": {
                    "kind": {"value": "box", "chosen_from": "shared", "confidence": 0.9, "conflict": False, "note": ""},
                    "material": {"value": None, "chosen_from": None, "confidence": 0.0, "conflict": False, "note": ""},
                    "dimensions": {
                        "size_triplet_mm": {
                            "value": [10.0, 10.0, 10.0],
                            "chosen_from": "shared",
                            "confidence": 0.9,
                            "conflict": False,
                            "note": "",
                        }
                    },
                    "ambiguities": [],
                },
                "merged_source": {
                    "source_type": {"value": "beam", "chosen_from": "shared", "confidence": 0.9, "conflict": False, "note": ""},
                    "particle": {"value": "gamma", "chosen_from": "shared", "confidence": 0.9, "conflict": False, "note": ""},
                    "energy_mev": {"value": 2.0, "chosen_from": "shared", "confidence": 0.9, "conflict": False, "note": ""},
                    "position": {
                        "value": {"position_mm": [0.0, 0.0, -50.0]},
                        "chosen_from": "shared",
                        "confidence": 0.9,
                        "conflict": False,
                        "note": "",
                    },
                    "direction": {
                        "value": {"mode": "explicit_vector", "hint": {"direction_vec": [0.0, 0.0, 1.0]}},
                        "chosen_from": "llm",
                        "confidence": 0.9,
                        "conflict": False,
                        "note": "",
                    },
                    "ambiguities": [],
                },
                "open_questions": [],
                "conflicts": [],
                "trust_report": {},
            },
        }
        with (
            patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=result),
            patch("core.orchestrator.session_manager._build_interpreter_sidecar", return_value=interpreter_debug),
        ):
            out = process_turn(
                {
                    "session_id": "selector-test",
                    "text": "keep the 10 mm box target but use a 2 MeV gamma beam from (0,0,-50) mm",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": True,
                },
                ollama_config_path="",
            )
        self.assertEqual(out["config"]["source"]["type"], "beam")
        self.assertEqual(out["config"]["source"]["particle"], "gamma")
        self.assertEqual(out["config"]["source"]["direction"], {"type": "vector", "value": [0.0, 0.0, 1.0]})
        self.assertTrue(out["slot_debug"]["interpreter_source"]["used"])

    def test_process_turn_preserves_partial_source_slots_across_rounds(self) -> None:
        turn1_frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="partial source turn 1",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.position_mm",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
            source=SourceSlots(kind="point", particle="gamma", position_mm=[0.0, 0.0, -20.0]),
        )
        turn2_frame = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="partial source turn 2",
            target_slots=[
                "source.energy_mev",
                "source.direction_vec",
            ],
            source=SourceSlots(energy_mev=1.0, direction_vec=[0.0, 0.0, 1.0]),
        )
        with patch(
            "core.orchestrator.session_manager.build_llm_slot_frame",
            side_effect=[
                LlmSlotBuildResult(
                    ok=True,
                    frame=turn1_frame,
                    normalized_text=turn1_frame.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
                LlmSlotBuildResult(
                    ok=True,
                    frame=turn2_frame,
                    normalized_text=turn2_frame.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
            ],
        ):
            first = process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "做一个铜靶，10 mm 见方，gamma点源，位于(0,0,-20) mm。",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )
            state = get_or_create_session("selector-memory-test")
            state.open_questions = ["source.direction"]
            state.last_asked_paths = ["source.direction"]
            second = process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "能量 1 MeV，朝 +z。",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )

        self.assertEqual(first["config"]["geometry"]["structure"], "single_box")
        self.assertIsNone(first["config"]["source"]["type"])
        self.assertEqual(second["config"]["source"]["type"], "point")
        self.assertEqual(second["config"]["source"]["particle"], "gamma")
        self.assertEqual(second["config"]["source"]["energy"], 1.0)
        self.assertEqual(second["config"]["source"]["position"]["value"], [0.0, 0.0, -20.0])
        self.assertEqual(second["config"]["source"]["direction"]["value"], [0.0, 0.0, 1.0])

    def test_process_turn_confirmed_source_overwrite_clears_stale_direction(self) -> None:
        initial = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="initial complete source",
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
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -20.0],
                direction_vec=[0.0, 0.0, 1.0],
            ),
        )
        overwrite = SlotFrame(
            intent=Intent.MODIFY,
            confidence=1.0,
            normalized_text="overwrite source without direction",
            target_slots=[
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
            ],
            source=SourceSlots(
                kind="beam",
                particle="gamma",
                energy_mev=2.0,
                position_mm=[0.0, 0.0, -50.0],
            ),
        )
        confirm = SlotFrame(
            intent=Intent.CONFIRM,
            confidence=1.0,
            normalized_text="confirm overwrite",
            target_slots=[],
        )
        with patch(
            "core.orchestrator.session_manager.build_llm_slot_frame",
            side_effect=[
                LlmSlotBuildResult(
                    ok=True,
                    frame=initial,
                    normalized_text=initial.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
                LlmSlotBuildResult(
                    ok=True,
                    frame=overwrite,
                    normalized_text=overwrite.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
                LlmSlotBuildResult(
                    ok=True,
                    frame=confirm,
                    normalized_text=confirm.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
            ],
        ):
            process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "initial full point source",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )
            state = get_or_create_session("selector-memory-test")
            state.open_questions = ["source.direction"]
            state.last_asked_paths = ["source.direction"]
            second = process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "change source to beam from (0,0,-50) mm with 2 MeV",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )
            third = process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "确认",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )

        self.assertTrue(second["pending_overwrite_required"])
        self.assertEqual(third["config"]["source"]["type"], "beam")
        self.assertEqual(third["config"]["source"]["energy"], 2.0)
        self.assertEqual(third["config"]["source"]["position"]["value"], [0.0, 0.0, -50.0])
        self.assertIsNone(third["config"]["source"]["direction"])

    def test_process_turn_answering_direction_question_does_not_rewrite_source(self) -> None:
        initial = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="initial source missing direction",
            target_slots=[
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
            ],
            geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 10.0, 10.0]),
            source=SourceSlots(
                kind="point",
                particle="gamma",
                energy_mev=1.0,
                position_mm=[0.0, 0.0, -20.0],
            ),
        )
        answer = SlotFrame(
            intent=Intent.SET,
            confidence=1.0,
            normalized_text="answer source direction only",
            target_slots=[
                "source.kind",
                "source.particle",
                "source.position_mm",
                "source.direction_vec",
            ],
            source=SourceSlots(
                kind="beam",
                particle="proton",
                position_mm=[0.0, 0.0, -50.0],
                direction_vec=[0.0, 0.0, -1.0],
            ),
        )
        with patch(
            "core.orchestrator.session_manager.build_llm_slot_frame",
            side_effect=[
                LlmSlotBuildResult(
                    ok=True,
                    frame=initial,
                    normalized_text=initial.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
                LlmSlotBuildResult(
                    ok=True,
                    frame=answer,
                    normalized_text=answer.normalized_text,
                    confidence=1.0,
                    llm_raw="{}",
                    fallback_reason=None,
                    schema_errors=[],
                    stage_trace={"final_status": "ok"},
                ),
            ],
        ):
            first = process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "10 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm.",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )
            state = get_or_create_session("selector-memory-test")
            state.open_questions = ["source.direction"]
            state.last_asked_paths = ["source.direction"]
            second = process_turn(
                {
                    "session_id": "selector-memory-test",
                    "text": "沿着z轴负方向",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )

        self.assertFalse(second.get("pending_overwrite_required", False))
        self.assertEqual(second["config"]["source"]["type"], "point")
        self.assertEqual(second["config"]["source"]["particle"], "gamma")
        self.assertEqual(second["config"]["source"]["energy"], 1.0)
        self.assertEqual(second["config"]["source"]["position"]["value"], [0.0, 0.0, -20.0])
        self.assertEqual(second["config"]["source"]["direction"]["value"], [0.0, 0.0, -1.0])

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

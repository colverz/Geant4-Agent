from __future__ import annotations

import unittest
from unittest.mock import patch

from core.orchestrator.session_manager import process_turn, reset_session
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.slots.slot_frame import GeometrySlots, SlotFrame, SourceSlots
from nlu.llm.slot_frame import LlmSlotBuildResult, parse_slot_payload
from ui.web.server import solve, step


class SmokeNoOllamaTest(unittest.TestCase):
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
        self.assertEqual(out.get("dialogue_action"), "ask_clarification")
        self.assertEqual(out.get("dialogue_summary", {}).get("status"), "pending")
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

    def test_explicit_overwrite_is_allowed(self) -> None:
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
        mats2 = second.get("config", {}).get("materials", {}).get("selected_materials", [])
        self.assertEqual(mats2, ["G4_Al"])
        self.assertEqual(
            second.get("config", {}).get("materials", {}).get("volume_material_map"),
            {"box": "G4_Al"},
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
        self.assertEqual(out.get("config", {}).get("source", {}).get("energy"), 1.0)


if __name__ == "__main__":
    unittest.main()

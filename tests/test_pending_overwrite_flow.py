from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.orchestrator.session_manager import (
    _augment_geometry_targets,
    _candidate_from_pending_overwrite,
    _extract_low_confidence_updates,
    _extract_pending_overwrites,
    get_or_create_session,
    process_turn,
    reset_session,
)
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.validation.error_codes import E_CANDIDATE_REJECTED_BY_GATE


def _turn(session_id: str, text: str) -> dict:
    return process_turn(
        {
            "session_id": session_id,
            "text": text,
            "llm_router": False,
            "llm_question": False,
            "normalize_input": False,
            "autofix": True,
        },
        ollama_config_path="",
        lang="en",
    )


class PendingOverwriteFlowTest(unittest.TestCase):
    def test_non_geometry_edit_does_not_get_geometry_targets_augmented(self) -> None:
        user_candidate = CandidateUpdate(
            producer=Producer.USER_EXPLICIT,
            intent=Intent.MODIFY,
            target_paths=["materials.selected_materials", "output.format"],
            updates=[],
            confidence=1.0,
            rationale="explicit_non_geometry_edit",
        )
        extracted_candidate = CandidateUpdate(
            producer=Producer.BERT_EXTRACTOR,
            intent=Intent.MODIFY,
            target_paths=["geometry.structure"],
            updates=[
                UpdateOp(
                    path="geometry.structure",
                    op="set",
                    value="boolean",
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.8,
                    turn_id=1,
                ),
                UpdateOp(
                    path="geometry.graph_program",
                    op="set",
                    value={"root": "boolean"},
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.8,
                    turn_id=1,
                ),
            ],
            confidence=0.8,
            rationale="spurious_geometry_graph",
        )

        augmented = _augment_geometry_targets(user_candidate, extracted_candidate)
        self.assertIsNotNone(augmented)
        self.assertEqual(augmented.target_paths, ["materials.selected_materials", "output.format"])

    def test_low_confidence_update_is_staged_before_write(self) -> None:
        state_like = SimpleNamespace(config={"output": {}})
        candidate = CandidateUpdate(
            producer=Producer.LLM_SEMANTIC_FRAME,
            intent=Intent.SET,
            target_paths=["output.format"],
            updates=[
                UpdateOp(
                    path="output.format",
                    op="set",
                    value="json",
                    producer=Producer.LLM_SEMANTIC_FRAME,
                    confidence=0.42,
                    turn_id=1,
                )
            ],
            confidence=0.42,
            rationale="low_confidence_test",
        )

        filtered, pending = _extract_low_confidence_updates(
            state_like,
            [candidate],
            min_confidence=0.6,
            lang="en",
        )

        self.assertEqual(filtered, [])
        self.assertEqual(pending[0]["path"], "output.format")
        self.assertEqual(pending[0]["reason"], "low_confidence")
        self.assertEqual(pending[0]["confidence"], 0.42)

    def test_remove_update_is_staged_and_confirm_preserves_remove_op(self) -> None:
        state_like = SimpleNamespace(config={"output": {"path": "old.json"}})
        user_candidate = CandidateUpdate(
            producer=Producer.USER_EXPLICIT,
            intent=Intent.REMOVE,
            target_paths=["output.path"],
            updates=[],
            confidence=1.0,
            rationale="user_remove",
        )
        candidate = CandidateUpdate(
            producer=Producer.LLM_SEMANTIC_FRAME,
            intent=Intent.REMOVE,
            target_paths=["output.path"],
            updates=[
                UpdateOp(
                    path="output.path",
                    op="remove",
                    value=None,
                    producer=Producer.LLM_SEMANTIC_FRAME,
                    confidence=0.9,
                    turn_id=2,
                )
            ],
            confidence=0.9,
            rationale="remove_test",
        )

        filtered, pending = _extract_pending_overwrites(state_like, user_candidate, [candidate], lang="en")
        confirmed = _candidate_from_pending_overwrite(pending, turn_id=3)

        self.assertEqual(filtered, [])
        self.assertEqual(pending[0]["reason"], "remove")
        self.assertEqual(pending[0]["op"], "remove")
        self.assertEqual(confirmed.updates[0].op, "remove")

    def test_geometry_overwrite_stages_structure_and_params_atomically(self) -> None:
        sid = "pending-atomic-geometry"
        reset_session(sid)

        first = _turn(
            sid,
            "1m x 1m x 1m copper box target, gamma point source, energy 1 MeV, position (0,0,-100), direction +z, physics FTFP_BERT, output json path output/result.json",
        )
        self.assertTrue(first.get("is_complete"))
        self.assertEqual(first.get("config", {}).get("geometry", {}).get("structure"), "single_box")

        second = _turn(
            sid,
            "change geometry to cylinder with radius 40 mm and half-length 80 mm",
        )
        self.assertEqual(second.get("dialogue_action"), "confirm_overwrite")
        self.assertTrue(second.get("pending_overwrite_required"))
        self.assertEqual(second.get("config", {}).get("geometry", {}).get("structure"), "single_box")

        state = get_or_create_session(sid)
        pending_paths = {str(item.get("path")) for item in state.pending_overwrite}
        self.assertIn("geometry.structure", pending_paths)
        self.assertIn("geometry.params.child_rmax", pending_paths)
        self.assertIn("geometry.params.child_hz", pending_paths)

    def test_confirm_rollback_keeps_pending_and_marks_incomplete(self) -> None:
        sid = "pending-confirm-rollback"
        reset_session(sid)

        first = _turn(
            sid,
            "1m x 1m x 1m copper box target, gamma point source, energy 1 MeV, position (0,0,-100), direction +z, physics FTFP_BERT, output json path output/result.json",
        )
        self.assertTrue(first.get("is_complete"))
        self.assertEqual(first.get("config", {}).get("source", {}).get("energy"), 1.0)

        state = get_or_create_session(sid)
        state.pending_overwrite = [
            {
                "path": "source.energy",
                "field": "source energy",
                "old": 1.0,
                "new": -1.0,
                "producer": "user_explicit",
            }
        ]

        second = _turn(sid, "confirm")
        self.assertFalse(second.get("is_complete"))
        self.assertTrue(second.get("pending_overwrite_required"))
        self.assertEqual(second.get("config", {}).get("source", {}).get("energy"), 1.0)
        self.assertGreaterEqual(len(get_or_create_session(sid).pending_overwrite), 1)
        reason_codes = {item.get("reason_code") for item in second.get("rejected_updates", [])}
        self.assertIn(E_CANDIDATE_REJECTED_BY_GATE, reason_codes)

    def test_reject_clears_stale_semantic_missing_paths(self) -> None:
        sid = "pending-reject-clears-semantic-missing"
        reset_session(sid)

        first = _turn(
            sid,
            "100 mm x 100 mm x 100 mm copper box, gamma point source, energy 1 MeV, position (0,0,-50), direction +z, physics QBBC, output json path output/result.json",
        )
        self.assertTrue(first.get("is_complete"))

        state = get_or_create_session(sid)
        state.pending_overwrite = [
            {
                "path": "materials.selected_materials",
                "field": "materials",
                "old": ["G4_Cu"],
                "new": ["G4_Al"],
                "producer": "user_explicit",
            }
        ]
        state.semantic_missing_paths = [
            "geometry.ask.boolean.solid_a_size",
            "geometry.ask.boolean.solid_b_size",
        ]

        second = _turn(sid, "keep existing")
        self.assertEqual(second.get("dialogue_action"), "reject_overwrite")
        self.assertTrue(second.get("is_complete"))
        self.assertEqual(second.get("missing_fields"), [])
        self.assertEqual(get_or_create_session(sid).semantic_missing_paths, [])


if __name__ == "__main__":
    unittest.main()

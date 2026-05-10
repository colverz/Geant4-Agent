from __future__ import annotations

import unittest

from core.agent.interrupt_resume import InterruptResumeController, InterruptResumeStatus
from core.agent.staged_patch import build_staged_patch
from core.agent.staged_patch_bridge import (
    apply_confirmation_candidate,
    preview_staged_patch_compatibility,
    staged_patch_to_confirmation_candidate,
)
from core.orchestrator.confirmation_policy import ConfirmationReason


def _config() -> dict:
    return {
        "source": {"energy": 1.0, "particle": "gamma"},
        "output": {"path": "old.json"},
    }


class StagedPatchBridgeTest(unittest.TestCase):
    def test_staged_approval_matches_legacy_confirmation_candidate_application(self) -> None:
        config = _config()
        patch = build_staged_patch(
            session_id="s1",
            base_turn_id=2,
            base_config=config,
            pending_items=[
                {
                    "path": "source.energy",
                    "op": "set",
                    "old": 1.0,
                    "new": 10.0,
                    "producer": "llm_semantic_frame",
                    "reason": ConfirmationReason.OVERWRITE,
                }
            ],
        )

        preview = preview_staged_patch_compatibility(config, patch, turn_id=3)

        self.assertTrue(preview.equivalent)
        self.assertEqual(preview.staged_config["source"]["energy"], 10.0)
        self.assertEqual(preview.candidate_config, preview.staged_config)
        self.assertEqual(preview.candidate.updates[0].path, "source.energy")

    def test_remove_approval_matches_legacy_confirmation_candidate_application(self) -> None:
        config = _config()
        patch = build_staged_patch(
            session_id="s1",
            base_turn_id=2,
            base_config=config,
            pending_items=[
                {
                    "path": "output.path",
                    "op": "remove",
                    "old": "old.json",
                    "new": None,
                    "producer": "user_explicit",
                    "reason": ConfirmationReason.REMOVE,
                }
            ],
        )

        preview = preview_staged_patch_compatibility(config, patch, turn_id=3)

        self.assertTrue(preview.equivalent)
        self.assertNotIn("path", preview.staged_config["output"])
        self.assertEqual(preview.candidate.updates[0].op, "remove")

    def test_bridge_candidate_application_does_not_mutate_original_config(self) -> None:
        config = _config()
        patch = build_staged_patch(
            session_id="s1",
            base_turn_id=2,
            base_config=config,
            pending_items=[
                {
                    "path": "source.energy",
                    "op": "set",
                    "old": 1.0,
                    "new": 10.0,
                    "producer": "llm_semantic_frame",
                    "reason": ConfirmationReason.OVERWRITE,
                }
            ],
        )
        candidate = staged_patch_to_confirmation_candidate(patch, turn_id=3)

        updated = apply_confirmation_candidate(config, candidate)

        self.assertEqual(updated["source"]["energy"], 10.0)
        self.assertEqual(config["source"]["energy"], 1.0)

    def test_controller_reject_matches_legacy_keep_original_semantics(self) -> None:
        config = _config()
        controller = InterruptResumeController()
        patch = controller.stage_confirmation(
            session_id="s1",
            base_turn_id=2,
            base_config=config,
            pending_items=[
                {
                    "path": "source.energy",
                    "op": "set",
                    "old": 1.0,
                    "new": 10.0,
                    "producer": "llm_semantic_frame",
                    "reason": ConfirmationReason.OVERWRITE,
                }
            ],
        )

        result = controller.reject(patch.confirmation_id)

        self.assertEqual(result.status, InterruptResumeStatus.REJECTED)
        self.assertIsNone(result.config)
        self.assertEqual(config["source"]["energy"], 1.0)

    def test_controller_stale_guard_is_additive_to_legacy_semantics(self) -> None:
        config = _config()
        controller = InterruptResumeController()
        patch = controller.stage_confirmation(
            session_id="s1",
            base_turn_id=2,
            base_config=config,
            pending_items=[
                {
                    "path": "source.energy",
                    "op": "set",
                    "old": 1.0,
                    "new": 10.0,
                    "producer": "llm_semantic_frame",
                    "reason": ConfirmationReason.OVERWRITE,
                }
            ],
        )

        result = controller.approve(patch.confirmation_id, current_turn_id=3, current_config=config)

        self.assertEqual(result.status, InterruptResumeStatus.STALE)
        self.assertEqual(result.error, "turn_id_changed")


if __name__ == "__main__":
    unittest.main()

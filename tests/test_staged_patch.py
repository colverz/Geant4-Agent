from __future__ import annotations

import unittest

from core.agent.interrupt_resume import InterruptResumeController, InterruptResumeStatus
from core.agent.staged_patch import (
    StagedPatch,
    StagedPatchStatus,
    StagedPatchStore,
    apply_staged_patch,
    build_staged_patch,
    build_staged_patch_reference,
    verify_staged_patch_hash,
)
from core.orchestrator.confirmation_policy import ConfirmationReason


def _config() -> dict:
    return {
        "source": {"energy": 1.0, "particle": "gamma"},
        "output": {"path": "old.json"},
    }


def _pending() -> list[dict]:
    return [
        {
            "path": "source.energy",
            "op": "set",
            "old": 1.0,
            "new": 10.0,
            "producer": "llm_semantic_frame",
            "reason": ConfirmationReason.OVERWRITE,
        }
    ]


class StagedPatchTest(unittest.TestCase):
    def test_build_staged_patch_creates_confirmation_id_and_stable_hash(self) -> None:
        first = build_staged_patch(
            session_id="s1",
            base_turn_id=3,
            base_config=_config(),
            pending_items=_pending(),
            confirmation_id="fixed-confirmation",
        )
        second = build_staged_patch(
            session_id="s1",
            base_turn_id=3,
            base_config=_config(),
            pending_items=_pending(),
            confirmation_id="fixed-confirmation",
        )

        self.assertEqual(first.confirmation_id, "fixed-confirmation")
        self.assertEqual(first.patch_hash, second.patch_hash)
        self.assertTrue(verify_staged_patch_hash(first))
        self.assertEqual(first.status, StagedPatchStatus.PENDING)

    def test_build_staged_patch_reference_uses_stable_confirmation_id(self) -> None:
        first = build_staged_patch_reference(
            session_id="s1",
            base_turn_id=3,
            base_config=_config(),
            pending_items=_pending(),
        )
        second = build_staged_patch_reference(
            session_id="s1",
            base_turn_id=3,
            base_config=_config(),
            pending_items=_pending(),
        )

        self.assertEqual(first.confirmation_id, second.confirmation_id)
        self.assertEqual(first.patch_hash, second.patch_hash)
        self.assertTrue(verify_staged_patch_hash(first))

    def test_apply_staged_patch_applies_exact_values_without_mutating_input(self) -> None:
        config = _config()
        patch = build_staged_patch(session_id="s1", base_turn_id=3, base_config=config, pending_items=_pending())

        updated = apply_staged_patch(config, patch)

        self.assertEqual(updated["source"]["energy"], 10.0)
        self.assertEqual(config["source"]["energy"], 1.0)

    def test_apply_staged_patch_preserves_remove_op(self) -> None:
        config = _config()
        patch = build_staged_patch(
            session_id="s1",
            base_turn_id=3,
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

        updated = apply_staged_patch(config, patch)

        self.assertNotIn("path", updated["output"])

    def test_store_reject_marks_patch_without_changing_config(self) -> None:
        controller = InterruptResumeController()
        patch = controller.stage_confirmation(
            session_id="s1",
            base_turn_id=3,
            base_config=_config(),
            pending_items=_pending(),
        )

        result = controller.reject(patch.confirmation_id)

        self.assertTrue(result.ok)
        self.assertEqual(result.status, InterruptResumeStatus.REJECTED)
        self.assertEqual(result.patch.status, StagedPatchStatus.REJECTED)
        self.assertIsNone(result.config)

    def test_approve_applies_patch_and_marks_approved(self) -> None:
        controller = InterruptResumeController()
        config = _config()
        patch = controller.stage_confirmation(
            session_id="s1",
            base_turn_id=3,
            base_config=config,
            pending_items=_pending(),
        )

        result = controller.approve(patch.confirmation_id, current_turn_id=3, current_config=config)

        self.assertTrue(result.ok)
        self.assertEqual(result.status, InterruptResumeStatus.APPROVED)
        self.assertEqual(result.patch.status, StagedPatchStatus.APPROVED)
        self.assertEqual(result.config["source"]["energy"], 10.0)

    def test_stale_turn_id_is_rejected(self) -> None:
        controller = InterruptResumeController()
        patch = controller.stage_confirmation(
            session_id="s1",
            base_turn_id=3,
            base_config=_config(),
            pending_items=_pending(),
        )

        result = controller.approve(patch.confirmation_id, current_turn_id=4, current_config=_config())

        self.assertFalse(result.ok)
        self.assertEqual(result.status, InterruptResumeStatus.STALE)
        self.assertEqual(result.error, "turn_id_changed")

    def test_mutated_config_is_rejected_as_stale(self) -> None:
        controller = InterruptResumeController()
        config = _config()
        patch = controller.stage_confirmation(
            session_id="s1",
            base_turn_id=3,
            base_config=config,
            pending_items=_pending(),
        )
        mutated = _config()
        mutated["source"]["particle"] = "proton"

        result = controller.approve(patch.confirmation_id, current_turn_id=3, current_config=mutated)

        self.assertFalse(result.ok)
        self.assertEqual(result.status, InterruptResumeStatus.STALE)
        self.assertEqual(result.error, "config_changed")

    def test_patch_hash_mismatch_cannot_apply(self) -> None:
        store = StagedPatchStore()
        controller = InterruptResumeController(store)
        config = _config()
        patch = store.stage(session_id="s1", base_turn_id=3, base_config=config, pending_items=_pending())
        store._patches[patch.confirmation_id] = StagedPatch(
            confirmation_id=patch.confirmation_id,
            patch_hash="bad-hash",
            session_id=patch.session_id,
            base_turn_id=patch.base_turn_id,
            base_config_hash=patch.base_config_hash,
            pending_items=patch.pending_items,
            status=patch.status,
            schema_version=patch.schema_version,
        )

        result = controller.approve(patch.confirmation_id, current_turn_id=3, current_config=config)

        self.assertFalse(result.ok)
        self.assertEqual(result.status, InterruptResumeStatus.HASH_MISMATCH)


if __name__ == "__main__":
    unittest.main()

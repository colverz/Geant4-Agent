from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.agent.candidate_patch import (
    CONFIRM_DELETE,
    CONFIRM_LOW_CONFIDENCE,
    CONFIRM_OVERWRITE,
    envelope_to_candidate_update,
    normalize_interpreter_v2_payload,
    preview_candidate_patch_confirmation,
)
from core.orchestrator.confirmation_policy import ConfirmationReason
from core.orchestrator.types import Intent, Producer


def _payload() -> dict:
    return {
        "turn_summary": {
            "intent": "modify",
            "focus": "mixed",
            "user_goal": "change source energy and run",
            "requires_confirmation": False,
        },
        "candidate_updates": [
            {
                "path": "source.energy_mev",
                "op": "set",
                "value": 10.0,
                "confidence": 0.92,
                "evidence": [{"text": "10 MeV", "source": "user", "role": "energy"}],
                "requires_confirmation": False,
            }
        ],
        "ambiguities": [],
        "unsupported_requests": [],
        "guarded_actions": [
            {"action": "run_beam", "safety_class": "expensive_runtime", "requested": True, "reason": "user asked to run"}
        ],
    }


class CandidatePatchNormalizerTest(unittest.TestCase):
    def test_normalizes_interpreter_v2_payload_into_patch_envelope(self) -> None:
        envelope = normalize_interpreter_v2_payload(_payload())

        self.assertEqual(envelope.intent, Intent.MODIFY)
        self.assertEqual(envelope.operations[0].path, "source.energy_mev")
        self.assertEqual(envelope.operations[0].value, 10.0)
        self.assertFalse(envelope.operations[0].requires_confirmation)
        self.assertEqual(envelope.operations[0].evidence[0].text, "10 MeV")
        self.assertEqual(envelope.guarded_actions[0].action, "run_beam")
        self.assertEqual(envelope.to_dict()["guarded_actions"][0]["safety_class"], "expensive_runtime")
        self.assertTrue(envelope.patch_hash)

    def test_converts_patch_envelope_to_existing_candidate_update_without_applying_session(self) -> None:
        envelope = normalize_interpreter_v2_payload(_payload())

        candidate = envelope_to_candidate_update(envelope, turn_id=7)

        self.assertEqual(candidate.producer, Producer.LLM_SEMANTIC_FRAME)
        self.assertEqual(candidate.intent, Intent.MODIFY)
        self.assertEqual(candidate.target_paths, ["source.energy_mev"])
        self.assertEqual(candidate.updates[0].path, "source.energy_mev")
        self.assertEqual(candidate.updates[0].value, 10.0)
        self.assertEqual(candidate.updates[0].turn_id, 7)
        self.assertIn(envelope.patch_hash, candidate.rationale)

    def test_low_confidence_update_requires_confirmation(self) -> None:
        payload = _payload()
        payload["candidate_updates"][0]["confidence"] = 0.42

        envelope = normalize_interpreter_v2_payload(payload, min_confidence=0.6)

        self.assertTrue(envelope.operations[0].requires_confirmation)
        self.assertIn(CONFIRM_LOW_CONFIDENCE, envelope.operations[0].confirmation_reasons)

    def test_explicit_requires_confirmation_marks_overwrite_reason(self) -> None:
        payload = _payload()
        payload["candidate_updates"][0]["requires_confirmation"] = True

        envelope = normalize_interpreter_v2_payload(payload)

        self.assertTrue(envelope.operations[0].requires_confirmation)
        self.assertIn(CONFIRM_OVERWRITE, envelope.operations[0].confirmation_reasons)

    def test_remove_update_requires_delete_confirmation_and_preserves_remove_op(self) -> None:
        payload = _payload()
        payload["turn_summary"]["intent"] = "remove"
        payload["candidate_updates"][0] = {
            "path": "output.path",
            "op": "remove",
            "value": None,
            "confidence": 0.9,
            "evidence": [{"text": "delete output path", "source": "user", "role": "delete"}],
            "requires_confirmation": False,
        }

        envelope = normalize_interpreter_v2_payload(payload)
        candidate = envelope_to_candidate_update(envelope, turn_id=3)

        self.assertEqual(envelope.intent, Intent.REMOVE)
        self.assertIn(CONFIRM_DELETE, envelope.operations[0].confirmation_reasons)
        self.assertEqual(candidate.updates[0].op, "remove")

    def test_keep_update_is_retained_in_envelope_but_not_written_as_update_op(self) -> None:
        payload = _payload()
        payload["candidate_updates"][0]["op"] = "keep"

        envelope = normalize_interpreter_v2_payload(payload)
        candidate = envelope_to_candidate_update(envelope, turn_id=3)

        self.assertEqual(envelope.operations[0].op, "keep")
        self.assertEqual(candidate.updates, [])
        self.assertEqual(candidate.target_paths, ["source.energy_mev"])

    def test_patch_hash_is_stable_for_same_payload(self) -> None:
        first = normalize_interpreter_v2_payload(_payload())
        second = normalize_interpreter_v2_payload(_payload())

        self.assertEqual(first.patch_hash, second.patch_hash)

    def test_confirmation_preview_stages_low_confidence_without_session_apply(self) -> None:
        payload = _payload()
        payload["candidate_updates"][0]["confidence"] = 0.42
        envelope = normalize_interpreter_v2_payload(payload)
        state_like = SimpleNamespace(config={"source": {}})

        preview = preview_candidate_patch_confirmation(envelope, state_like=state_like, turn_id=1, min_confidence=0.6)

        self.assertTrue(preview.requires_confirmation)
        self.assertIsNone(preview.kept_candidate)
        self.assertEqual(preview.pending[0]["path"], "source.energy_mev")
        self.assertEqual(preview.pending[0]["reason"], ConfirmationReason.LOW_CONFIDENCE)
        self.assertEqual(state_like.config, {"source": {}})

    def test_confirmation_preview_stages_explicit_overwrite(self) -> None:
        payload = _payload()
        payload["candidate_updates"][0]["requires_confirmation"] = True
        envelope = normalize_interpreter_v2_payload(payload)
        state_like = SimpleNamespace(config={"source": {"energy_mev": 1.0}})

        preview = preview_candidate_patch_confirmation(envelope, state_like=state_like, turn_id=1)

        self.assertTrue(preview.requires_confirmation)
        self.assertIsNone(preview.kept_candidate)
        self.assertEqual(preview.pending[0]["path"], "source.energy_mev")
        self.assertEqual(preview.pending[0]["old"], 1.0)
        self.assertEqual(preview.pending[0]["new"], 10.0)
        self.assertEqual(preview.pending[0]["reason"], ConfirmationReason.OVERWRITE)
        self.assertEqual(state_like.config["source"]["energy_mev"], 1.0)

    def test_confirmation_preview_stages_delete(self) -> None:
        payload = _payload()
        payload["turn_summary"]["intent"] = "remove"
        payload["candidate_updates"][0] = {
            "path": "output.path",
            "op": "remove",
            "value": None,
            "confidence": 0.9,
            "evidence": [{"text": "delete output path", "source": "user", "role": "delete"}],
            "requires_confirmation": False,
        }
        envelope = normalize_interpreter_v2_payload(payload)
        state_like = SimpleNamespace(config={"output": {"path": "old.json"}})

        preview = preview_candidate_patch_confirmation(envelope, state_like=state_like, turn_id=1)

        self.assertTrue(preview.requires_confirmation)
        self.assertIsNone(preview.kept_candidate)
        self.assertEqual(preview.pending[0]["path"], "output.path")
        self.assertEqual(preview.pending[0]["op"], "remove")
        self.assertEqual(preview.pending[0]["reason"], ConfirmationReason.REMOVE)
        self.assertEqual(state_like.config["output"]["path"], "old.json")

    def test_confirmation_preview_keeps_non_conflicting_candidate(self) -> None:
        envelope = normalize_interpreter_v2_payload(_payload())
        state_like = SimpleNamespace(config={"source": {}})

        preview = preview_candidate_patch_confirmation(envelope, state_like=state_like, turn_id=1)

        self.assertFalse(preview.requires_confirmation)
        self.assertEqual(preview.pending, [])
        self.assertIsNotNone(preview.kept_candidate)
        self.assertEqual(preview.kept_candidate.updates[0].path, "source.energy_mev")


if __name__ == "__main__":
    unittest.main()

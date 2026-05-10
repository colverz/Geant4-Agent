from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.orchestrator.confirmation_policy import (
    ConfirmationReason,
    ConfirmationResponse,
    ConfirmationPolicyResult,
    build_candidate_from_pending_confirmation,
    build_confirmation_payload,
    has_pending_confirmation_path,
    is_unset_for_confirmation,
    merge_pending_confirmations,
    evaluate_confirmation_requirements,
)
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.validation.error_codes import E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT


def _candidate(
    *,
    intent: Intent = Intent.MODIFY,
    path: str = "source.energy",
    value: object = 10.0,
    confidence: float = 0.9,
    op: str = "set",
    producer: Producer = Producer.LLM_SEMANTIC_FRAME,
) -> CandidateUpdate:
    return CandidateUpdate(
        producer=producer,
        intent=intent,
        target_paths=[path],
        updates=[
            UpdateOp(
                path=path,
                op="remove" if op == "remove" else "set",
                value=value,
                producer=producer,
                confidence=confidence,
                turn_id=1,
            )
        ],
        confidence=confidence,
        rationale="test",
    )


class ConfirmationPolicyPublicApiTest(unittest.TestCase):
    def test_public_api_returns_result_object_for_non_conflicting_update(self) -> None:
        state_like = SimpleNamespace(config={"source": {}})
        candidate = _candidate()

        result = evaluate_confirmation_requirements(state_like, candidate, [candidate], lang="en", min_confidence=0.6)

        self.assertIsInstance(result, ConfirmationPolicyResult)
        self.assertFalse(result.requires_confirmation)
        self.assertEqual(result.pending, [])
        self.assertEqual(result.filtered_candidates[0].updates[0].path, "source.energy")

    def test_public_api_stages_low_confidence_llm_update(self) -> None:
        state_like = SimpleNamespace(config={"source": {}})
        candidate = _candidate(confidence=0.42)

        result = evaluate_confirmation_requirements(state_like, candidate, [candidate], lang="en", min_confidence=0.6)

        self.assertTrue(result.requires_confirmation)
        self.assertEqual(result.filtered_candidates, [])
        self.assertEqual(result.pending[0]["reason"], ConfirmationReason.LOW_CONFIDENCE)
        self.assertEqual(result.pending[0]["confidence"], 0.42)

    def test_public_api_stages_remove_update(self) -> None:
        state_like = SimpleNamespace(config={"output": {"path": "old.json"}})
        candidate = _candidate(intent=Intent.REMOVE, path="output.path", value=None, op="remove")

        result = evaluate_confirmation_requirements(state_like, candidate, [candidate], lang="en", min_confidence=0.6)

        self.assertTrue(result.requires_confirmation)
        self.assertEqual(result.pending[0]["reason"], ConfirmationReason.REMOVE)
        self.assertEqual(result.pending[0]["op"], "remove")

    def test_public_api_can_enforce_no_implicit_overwrite(self) -> None:
        state_like = SimpleNamespace(config={"source": {"particle": "gamma"}})
        user_candidate = _candidate(path="source.energy", value=10.0)
        implicit_candidate = _candidate(path="source.particle", value="proton", producer=Producer.BERT_EXTRACTOR)

        result = evaluate_confirmation_requirements(
            state_like,
            user_candidate,
            [implicit_candidate],
            lang="en",
            min_confidence=0.0,
            enforce_no_implicit_overwrite=True,
        )

        self.assertEqual(result.filtered_candidates, [])
        self.assertEqual(result.pending, [])
        self.assertEqual(result.rejected[0]["reason_code"], E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT)

    def test_public_api_can_preserve_session_manager_confirmation_order(self) -> None:
        state_like = SimpleNamespace(config={"source": {"energy": 1.0}})
        candidate = _candidate(path="source.energy", value=10.0, confidence=0.42)

        result = evaluate_confirmation_requirements(
            state_like,
            candidate,
            [candidate],
            lang="en",
            min_confidence=0.6,
            low_confidence_first=False,
        )

        self.assertTrue(result.requires_confirmation)
        self.assertEqual(result.pending[0]["reason"], ConfirmationReason.OVERWRITE)
        self.assertEqual(result.pending[0]["old"], 1.0)
        self.assertEqual(result.pending[0]["new"], 10.0)

    def test_public_api_can_skip_pending_evaluation_for_confirm_apply_path(self) -> None:
        state_like = SimpleNamespace(config={"source": {"energy": 1.0}})
        candidate = _candidate(path="source.energy", value=10.0)

        result = evaluate_confirmation_requirements(
            state_like,
            candidate,
            [candidate],
            lang="en",
            min_confidence=0.6,
            low_confidence_first=False,
            evaluate_pending=False,
        )

        self.assertFalse(result.requires_confirmation)
        self.assertEqual(result.pending, [])
        self.assertEqual(result.filtered_candidates[0].updates[0].value, 10.0)

    def test_public_api_builds_confirm_candidate_from_pending_items(self) -> None:
        pending = [
            {"path": "source.energy", "op": "set", "new": 10.0},
            {"path": "output.path", "op": "remove", "new": None},
        ]

        candidate = build_candidate_from_pending_confirmation(pending, turn_id=7)

        self.assertEqual(candidate.producer, Producer.USER_EXPLICIT)
        self.assertEqual(candidate.intent, Intent.MODIFY)
        self.assertEqual(candidate.target_paths, ["output.path", "source.energy"])
        self.assertEqual(candidate.updates[0].path, "source.energy")
        self.assertEqual(candidate.updates[0].value, 10.0)
        self.assertEqual(candidate.updates[1].op, "remove")
        self.assertEqual(candidate.updates[1].turn_id, 7)

    def test_public_api_merges_pending_confirmations_by_path(self) -> None:
        existing = [
            {"path": "source.energy", "new": 1.0},
            {"path": "source.particle", "new": "gamma"},
        ]
        additions = [
            {"path": "source.energy", "new": 5.0},
            {"path": "source.direction", "new": [0, 0, 1]},
        ]

        merged = merge_pending_confirmations(existing, additions)

        by_path = {item["path"]: item for item in merged}
        self.assertEqual(by_path["source.energy"]["new"], 5.0)
        self.assertEqual(by_path["source.particle"]["new"], "gamma")
        self.assertEqual(by_path["source.direction"]["new"], [0, 0, 1])

    def test_public_api_checks_pending_path_and_unset_values(self) -> None:
        pending = [{"path": "geometry.structure", "new": "box"}]

        self.assertTrue(has_pending_confirmation_path(pending, "geometry.structure"))
        self.assertFalse(has_pending_confirmation_path(pending, "geometry.params.x"))
        self.assertTrue(is_unset_for_confirmation(None))
        self.assertTrue(is_unset_for_confirmation({}))
        self.assertFalse(is_unset_for_confirmation("gamma"))

    def test_public_reason_constants_are_stable_wire_values(self) -> None:
        self.assertEqual(ConfirmationReason.OVERWRITE, "overwrite")
        self.assertEqual(ConfirmationReason.REMOVE, "remove")
        self.assertEqual(ConfirmationReason.LOW_CONFIDENCE, "low_confidence")

    def test_public_api_builds_user_visible_confirmation_payload(self) -> None:
        payload = build_confirmation_payload(
            [
                {
                    "path": "source.energy",
                    "field": "source energy",
                    "old": 1.0,
                    "new": 10.0,
                    "producer": "llm_semantic_frame",
                    "reason": ConfirmationReason.OVERWRITE,
                    "confidence": 0.72,
                }
            ],
            lang="en",
        )

        self.assertTrue(payload["required"])
        self.assertEqual(payload["status"], "waiting_confirmation")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["path"], "source.energy")
        self.assertEqual(payload["items"][0]["reason"], ConfirmationReason.OVERWRITE)
        self.assertEqual(payload["items"][0]["confidence"], 0.72)
        self.assertEqual(
            payload["available_responses"],
            [
                ConfirmationResponse.CONFIRM,
                ConfirmationResponse.REJECT,
                ConfirmationResponse.KEEP_ORIGINAL,
            ],
        )

    def test_public_api_builds_empty_confirmation_payload(self) -> None:
        payload = build_confirmation_payload([], lang="en")

        self.assertFalse(payload["required"])
        self.assertEqual(payload["status"], "none")
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["items"], [])


if __name__ == "__main__":
    unittest.main()

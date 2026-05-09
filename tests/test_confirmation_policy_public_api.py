from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.orchestrator.confirmation_policy import ConfirmationPolicyResult, evaluate_confirmation_requirements
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
        self.assertEqual(result.pending[0]["reason"], "low_confidence")
        self.assertEqual(result.pending[0]["confidence"], 0.42)

    def test_public_api_stages_remove_update(self) -> None:
        state_like = SimpleNamespace(config={"output": {"path": "old.json"}})
        candidate = _candidate(intent=Intent.REMOVE, path="output.path", value=None, op="remove")

        result = evaluate_confirmation_requirements(state_like, candidate, [candidate], lang="en", min_confidence=0.6)

        self.assertTrue(result.requires_confirmation)
        self.assertEqual(result.pending[0]["reason"], "remove")
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
        self.assertEqual(result.pending[0]["reason"], "overwrite")
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


if __name__ == "__main__":
    unittest.main()

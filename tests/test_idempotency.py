from __future__ import annotations

import unittest

from core.agent.idempotency import (
    IdempotencyActionClass,
    IdempotencyDecision,
    IdempotencyReplayPolicy,
    build_action_id,
    classify_idempotent_action,
)


class IdempotencyReplayPolicyTest(unittest.TestCase):
    def test_read_only_summary_can_be_repeated_safely(self) -> None:
        policy = IdempotencyReplayPolicy()
        payload = {"session_id": "s1"}
        first = policy.check_before_execute("read_summary", payload)
        self.assertEqual(first.decision, IdempotencyDecision.EXECUTE)

        policy.record_result("read_summary", payload, {"events_completed": 10}, action_id=first.action_id)
        second = policy.check_before_execute("read_summary", payload, action_id=first.action_id)

        self.assertEqual(second.decision, IdempotencyDecision.REPLAY_RESULT)
        self.assertFalse(second.should_execute)
        self.assertEqual(second.record.result["events_completed"], 10)

    def test_validation_retry_replays_result_and_does_not_request_runtime_execution(self) -> None:
        policy = IdempotencyReplayPolicy()
        payload = {"config_hash": "abc"}
        action_id = build_action_id("validate_config", payload)

        first = policy.check_before_execute("validate_config", payload, action_id=action_id)
        self.assertTrue(first.should_execute)
        policy.record_result("validate_config", payload, {"ok": True}, action_id=action_id)
        retry = policy.check_before_execute("validate_config", payload, action_id=action_id)

        self.assertEqual(retry.decision, IdempotencyDecision.REPLAY_RESULT)
        self.assertEqual(retry.record.result, {"ok": True})

    def test_runtime_run_requires_explicit_action_id(self) -> None:
        policy = IdempotencyReplayPolicy()

        decision = policy.check_before_execute("run_beam", {"events": 10})

        self.assertEqual(decision.decision, IdempotencyDecision.REQUIRE_ACTION_ID)
        self.assertEqual(decision.reason, "explicit_action_id_required")

    def test_duplicate_runtime_run_returns_existing_result_without_rerun(self) -> None:
        policy = IdempotencyReplayPolicy()
        payload = {"events": 10}
        action_id = "run-001"

        first = policy.check_before_execute("run_beam", payload, action_id=action_id)
        self.assertEqual(first.decision, IdempotencyDecision.EXECUTE)
        policy.record_result("run_beam", payload, {"result_summary": {"events_completed": 10}}, action_id=action_id)
        duplicate = policy.check_before_execute("run_beam", payload, action_id=action_id)

        self.assertEqual(duplicate.decision, IdempotencyDecision.REPLAY_RESULT)
        self.assertFalse(duplicate.should_execute)
        self.assertEqual(duplicate.record.result["result_summary"]["events_completed"], 10)

    def test_viewer_launch_is_never_replayed_automatically(self) -> None:
        policy = IdempotencyReplayPolicy()
        payload = {"events": 12}
        action_id = "viewer-001"

        first = policy.check_before_execute("viewer_open", payload, action_id=action_id)
        self.assertEqual(first.decision, IdempotencyDecision.EXECUTE)
        policy.record_result("viewer_open", payload, {"viewer_pid": 1234}, action_id=action_id)
        duplicate = policy.check_before_execute("viewer_open", payload, action_id=action_id)

        self.assertEqual(duplicate.decision, IdempotencyDecision.REJECT_DUPLICATE)
        self.assertEqual(duplicate.reason, "runtime_action_must_not_be_replayed")
        self.assertEqual(duplicate.record.result["viewer_pid"], 1234)

    def test_file_write_and_batch_run_duplicates_are_blocked(self) -> None:
        for action_name in ("file_write", "batch_run"):
            with self.subTest(action_name=action_name):
                policy = IdempotencyReplayPolicy()
                payload = {"path": "out.json"}
                action_id = f"{action_name}-001"

                self.assertEqual(policy.check_before_execute(action_name, payload, action_id=action_id).decision, IdempotencyDecision.EXECUTE)
                policy.record_result(action_name, payload, {"ok": True}, action_id=action_id)
                duplicate = policy.check_before_execute(action_name, payload, action_id=action_id)

                self.assertEqual(duplicate.decision, IdempotencyDecision.REJECT_DUPLICATE)
                self.assertEqual(duplicate.reason, "external_side_effect_duplicate_blocked")

    def test_reusing_action_id_for_different_request_is_conflict(self) -> None:
        policy = IdempotencyReplayPolicy()
        policy.record_result("run_beam", {"events": 10}, {"ok": True}, action_id="runtime-001")

        decision = policy.check_before_execute("run_beam", {"events": 20}, action_id="runtime-001")

        self.assertEqual(decision.decision, IdempotencyDecision.CONFLICT)
        self.assertEqual(decision.reason, "action_id_reused_for_different_request")

    def test_action_classification_is_explicit_for_runtime_and_read_paths(self) -> None:
        self.assertEqual(classify_idempotent_action("read_summary"), IdempotencyActionClass.READ_ONLY)
        self.assertEqual(classify_idempotent_action("validate_config"), IdempotencyActionClass.DETERMINISTIC_VALIDATION)
        self.assertEqual(classify_idempotent_action("run_beam"), IdempotencyActionClass.RESULT_REUSABLE_RUNTIME)
        self.assertEqual(classify_idempotent_action("viewer_open"), IdempotencyActionClass.NEVER_REPLAY_RUNTIME)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import unittest

from core.orchestrator.path_ops import set_path
from core.orchestrator.session_manager import get_or_create_session, reset_session
from planner.runtime_intent import action_for_runtime_intent, classify_user_runtime_intent
from ui.web.request_router import handle_post_request


CASEBANK_PATH = Path("docs/eval/session_behavior_casebank.json")


def _seed_session(session_id: str) -> int:
    reset_session(session_id)
    state = get_or_create_session(session_id)
    set_path(state.config, "geometry.structure", "single_box")
    set_path(state.config, "source.type", "point")
    set_path(state.config, "source.particle", "gamma")
    return int(state.turn_id)


class SessionBehaviorGuardCasebankTest(unittest.TestCase):
    def test_session_behavior_casebank_matches_api_boundaries(self) -> None:
        cases = json.loads(CASEBANK_PATH.read_text(encoding="utf-8"))

        for index, case in enumerate(cases):
            with self.subTest(case=case["id"]):
                session_id = f"session-behavior-test-{index}"
                turn_before = _seed_session(session_id)
                step_calls: list[dict] = []

                def fail_step(_payload: dict, **_kwargs) -> dict:
                    raise AssertionError("read-only guard path must not call step_fn")

                def ok_step(payload: dict, **_kwargs) -> dict:
                    step_calls.append(dict(payload))
                    return {"ok": True, "session_id": payload.get("session_id")}

                result = classify_user_runtime_intent(case["text"], case["lang"])
                action = action_for_runtime_intent(result.intent).value

                self.assertEqual(result.intent.value, case["expected_intent"])
                self.assertEqual(action, case["expected_action"])

                body: dict = {}
                status = 200
                runtime_run_called = False
                if action == "config_summary":
                    status, body = handle_post_request(
                        "/api/config/summary",
                        {"session_id": session_id, "lang": case["lang"]},
                        legacy_sessions={},
                        solve_fn=lambda _payload: {"unexpected": "solve"},
                        step_fn=fail_step,
                    )
                elif action == "runtime_summary":
                    status, body = handle_post_request(
                        "/api/geant4/summary",
                        {"lang": case["lang"], "question": case["text"]},
                        legacy_sessions={},
                        solve_fn=lambda _payload: {"unexpected": "solve"},
                        step_fn=fail_step,
                    )
                elif action == "step_async":
                    status, body = handle_post_request(
                        "/api/step_async",
                        {"session_id": session_id, "text": case["text"], "lang": case["lang"]},
                        legacy_sessions={},
                        solve_fn=lambda _payload: {"unexpected": "solve"},
                        step_fn=ok_step,
                    )
                elif action == "guarded_runtime_ui":
                    runtime_run_called = False
                elif action == "read_only_chat":
                    body = {"action_safety_class": "read_only"}

                if action == "runtime_summary":
                    self.assertIn(status, {200, 400})
                else:
                    self.assertEqual(status, 200)
                self.assertEqual(get_or_create_session(session_id).turn_id - turn_before, case["expect_session_turn_delta"])
                self.assertEqual(bool(body.get("job_id")), case["expect_step_job"])
                self.assertEqual(runtime_run_called, case["expect_runtime_run"])
                self.assertEqual(action == "step_async", case["expect_config_mutation_allowed"])
                if action in {"config_summary", "runtime_summary"}:
                    self.assertEqual(body.get("action_safety_class"), "read_only")


if __name__ == "__main__":
    unittest.main()

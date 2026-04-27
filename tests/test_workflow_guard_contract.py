from __future__ import annotations

import json
from pathlib import Path
import unittest

from core.orchestrator.path_ops import set_path
from core.orchestrator.session_manager import get_or_create_session, reset_session
from core.runtime.types import ActionSafetyClass
from planner.runtime_intent import RuntimeIntent, classify_user_runtime_intent
from ui.web.request_router import handle_post_request


CASEBANK_PATH = Path("docs/eval/workflow_guard_casebank.json")


class WorkflowGuardContractTest(unittest.TestCase):
    def test_natural_language_intent_casebank_safety_classes(self) -> None:
        cases = json.loads(CASEBANK_PATH.read_text(encoding="utf-8"))

        for case in cases:
            with self.subTest(case=case["id"]):
                result = classify_user_runtime_intent(case["text"], case["lang"])

                self.assertEqual(result.intent, RuntimeIntent(case["expected_intent"]))
                self.assertEqual(result.action_safety_class, ActionSafetyClass(case["expected_safety"]))
                self.assertTrue(result.prompt_validation["ok"])

    def test_config_summary_is_read_only_and_never_enters_step_workflow(self) -> None:
        sid = "workflow-guard-config-summary"
        reset_session(sid)
        state = get_or_create_session(sid)
        set_path(state.config, "geometry.structure", "single_box")
        set_path(state.config, "source.type", "point")
        set_path(state.config, "source.particle", "gamma")
        turn_id_before = state.turn_id

        def fail_step(_payload: dict) -> dict:
            raise AssertionError("/api/config/summary must not call step_fn")

        status, body = handle_post_request(
            "/api/config/summary",
            {"session_id": sid, "lang": "en"},
            legacy_sessions={},
            solve_fn=lambda _payload: {"unexpected": "solve"},
            step_fn=fail_step,
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["action_safety_class"], "read_only")
        self.assertEqual(get_or_create_session(sid).turn_id, turn_id_before)

    def test_missing_config_summary_does_not_create_session(self) -> None:
        sid = "workflow-guard-missing-config"
        reset_session(sid)

        status, body = handle_post_request(
            "/api/config/summary",
            {"session_id": sid, "lang": "en"},
            legacy_sessions={},
            solve_fn=lambda _payload: {"unexpected": "solve"},
            step_fn=lambda _payload: {"unexpected": "step"},
        )

        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "no_session_available")

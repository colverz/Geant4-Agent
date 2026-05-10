from __future__ import annotations

import unittest

from core.orchestrator.path_ops import set_path
from core.orchestrator.confirmation_policy import ConfirmationReason, ConfirmationResponse
from core.orchestrator.session_manager import get_or_create_session, reset_session
from ui.web.request_router import handle_post_request


class ConfigSummaryApiTest(unittest.TestCase):
    def test_config_summary_is_read_only_and_does_not_create_step_job(self) -> None:
        sid = "config-summary-test"
        reset_session(sid)
        state = get_or_create_session(sid)
        set_path(state.config, "geometry.structure", "single_box")
        set_path(state.config, "materials.selected_materials", ["G4_Cu"])
        set_path(state.config, "source.type", "point")
        set_path(state.config, "source.particle", "gamma")
        turn_id_before = state.turn_id

        status, body = handle_post_request(
            "/api/config/summary",
            {"session_id": sid, "lang": "en"},
            legacy_sessions={},
            solve_fn=lambda payload: {"unexpected": "solve"},
            step_fn=lambda payload: {"unexpected": "step"},
        )

        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["action_safety_class"], "read_only")
        self.assertEqual(body["config_identity"]["geometry_structure"], "single_box")
        self.assertEqual(body["config_identity"]["particle"], "gamma")
        self.assertIn("Current phase", body["message"])
        self.assertEqual(get_or_create_session(sid).turn_id, turn_id_before)

    def test_config_summary_reports_missing_session(self) -> None:
        status, body = handle_post_request(
            "/api/config/summary",
            {"session_id": "missing-config-summary", "lang": "en"},
            legacy_sessions={},
            solve_fn=lambda payload: {},
            step_fn=lambda payload: {},
        )

        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "no_session_available")

    def test_config_summary_exposes_read_only_confirmation_payload(self) -> None:
        sid = "config-summary-confirmation-test"
        reset_session(sid)
        state = get_or_create_session(sid)
        state.pending_overwrite = [
            {
                "path": "source.energy",
                "field": "source energy",
                "old": 1.0,
                "new": 10.0,
                "producer": "llm_semantic_frame",
                "reason": ConfirmationReason.OVERWRITE,
            }
        ]
        turn_id_before = state.turn_id

        status, body = handle_post_request(
            "/api/config/summary",
            {"session_id": sid, "lang": "en"},
            legacy_sessions={},
            solve_fn=lambda payload: {"unexpected": "solve"},
            step_fn=lambda payload: {"unexpected": "step"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(get_or_create_session(sid).turn_id, turn_id_before)
        self.assertTrue(body["confirmation"]["required"])
        self.assertEqual(body["confirmation"]["status"], "waiting_confirmation")
        self.assertEqual(body["confirmation"]["items"][0]["path"], "source.energy")
        self.assertIn(ConfirmationResponse.KEEP_ORIGINAL, body["confirmation"]["available_responses"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from core.orchestrator.path_ops import set_path
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


if __name__ == "__main__":
    unittest.main()

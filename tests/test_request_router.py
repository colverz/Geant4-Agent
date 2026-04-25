from __future__ import annotations

import unittest

from ui.web.request_router import handle_post_request, is_supported_post_path


class RequestRouterTest(unittest.TestCase):
    def test_supported_paths_registry(self) -> None:
        self.assertTrue(is_supported_post_path("/api/solve"))
        self.assertTrue(is_supported_post_path("/api/step"))
        self.assertTrue(is_supported_post_path("/api/config/summary"))
        self.assertTrue(is_supported_post_path("/api/geant4/intent"))
        self.assertTrue(is_supported_post_path("/api/geant4/summary"))
        self.assertFalse(is_supported_post_path("/api/unknown"))

    def test_reset_clears_legacy_session(self) -> None:
        legacy_sessions = {"s1": object()}

        status, body = handle_post_request(
            "/api/reset",
            {"session_id": "s1"},
            legacy_sessions=legacy_sessions,
            solve_fn=lambda payload: {"route": "solve"},
            step_fn=lambda payload: {"route": "step"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True})
        self.assertNotIn("s1", legacy_sessions)


if __name__ == "__main__":
    unittest.main()

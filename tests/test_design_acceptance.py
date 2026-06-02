from __future__ import annotations

import unittest

from core.agent.design_acceptance import DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION, build_design_acceptance_patch
from core.orchestrator.session_manager import reset_session
from ui.web.request_router import handle_post_request


class DesignAcceptancePatchTest(unittest.TestCase):
    def test_build_design_acceptance_patch_from_recommended_config(self) -> None:
        patch = build_design_acceptance_patch(
            {
                "geometry": {"structure": "single_box"},
                "materials": {"selected_materials": ["G4_Pb"]},
                "source": {"type": "beam", "particle": "gamma"},
                "physics": {"physics_list": "FTFP_BERT"},
                "output": {"format": "json"},
            },
            base_config={},
            design_advice={
                "goal": "gamma shielding",
                "user_visible_summary": "A runnable shielding design is ready.",
                "primary_option": {"title": "lead shielding benchmark"},
            },
        )

        self.assertEqual(patch["schema_version"], DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION)
        self.assertTrue(patch["patch_hash"])
        self.assertEqual(patch["candidate_patch"]["role"] if "role" in patch["candidate_patch"] else "candidate_patch", "candidate_patch")
        self.assertEqual(patch["candidate_patch"]["source"], "accepted_simulation_design")
        self.assertIn("geometry", [item["path"] for item in patch["operations"]])
        self.assertIn("design_advice", {item["source"] for item in patch["evidence"]})

    def test_accept_simulation_design_returns_acceptance_patch(self) -> None:
        session_id = "design-acceptance-web-api"
        reset_session(session_id)
        common = {"legacy_sessions": {}, "solve_fn": lambda payload: {}, "step_fn": lambda payload: {}}
        try:
            design_status, design_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "Design a lead shielding gamma transmission benchmark with a silicon detector.",
                    "lang": "en",
                    "llm_router": False,
                },
                **common,
            )
            accept_status, accept_body = handle_post_request(
                "/api/simulation/accept",
                {"session_id": session_id},
                **common,
            )

            self.assertEqual(design_status, 200)
            self.assertIn("design_advice", design_body)
            self.assertEqual(accept_status, 200)
            self.assertTrue(accept_body["ok"])
            acceptance_patch = accept_body["design_acceptance_patch"]
            self.assertEqual(acceptance_patch["schema_version"], DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION)
            self.assertEqual(acceptance_patch["source"], "accepted_simulation_design")
            self.assertTrue(acceptance_patch["patch_hash"])
            self.assertIn("geometry", [item["path"] for item in acceptance_patch["operations"]])
            self.assertIn("candidate_patch", acceptance_patch)
            self.assertIn("design_advice", {item["source"] for item in acceptance_patch["evidence"]})
        finally:
            reset_session(session_id)


if __name__ == "__main__":
    unittest.main()

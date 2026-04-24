from __future__ import annotations

from pathlib import Path
import unittest

from mcp.geant4.adapter import InMemoryGeant4Adapter
from mcp.geant4.server import Geant4McpServer
import ui.web.geant4_api as geant4_api


def _runtime_patch() -> dict:
    return {
        "geometry": {"structure": "single_box"},
        "source": {"type": "point", "particle": "gamma"},
        "physics_list": {"name": "FTFP_BERT"},
    }


class Geant4WebApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_server = geant4_api._GEANT4_SERVER
        geant4_api._GEANT4_SERVER = Geant4McpServer(adapter=InMemoryGeant4Adapter())

    def tearDown(self) -> None:
        geant4_api._GEANT4_SERVER = self._previous_server

    def test_summary_requires_completed_run(self) -> None:
        status, body = geant4_api.handle_geant4_post("/api/geant4/summary", {})

        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "rejected")
        self.assertEqual(body["action_safety_class"], "read_only")
        self.assertIn("no_result_summary_available", body["errors"])
        self.assertNotIn("runtime_smoke_report", body)

    def test_runtime_intent_endpoint_classifies_without_running(self) -> None:
        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/intent",
            {"text": "What was the latest simulation result?", "lang": "en"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "read_summary")
        self.assertEqual(body["action_safety_class"], "read_only")
        self.assertEqual(body["prompt_profile_id"], "result_question_route_en_v1")
        self.assertTrue(body["prompt_validation"]["ok"])

        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/intent",
            {"text": "run 10 events now", "lang": "en"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "run_requested")
        self.assertEqual(body["action_safety_class"], "expensive_runtime")

    def test_run_returns_runtime_smoke_report_and_summary_reuses_it(self) -> None:
        apply_status, _ = geant4_api.handle_geant4_post("/api/geant4/apply", {"patch": _runtime_patch()})
        init_status, _ = geant4_api.handle_geant4_post("/api/geant4/initialize", {})
        run_status, run_body = geant4_api.handle_geant4_post("/api/geant4/run", {"events": 4})

        self.assertEqual(apply_status, 200)
        self.assertEqual(init_status, 200)
        self.assertEqual(run_status, 200)
        self.assertEqual(run_body["action_safety_class"], "expensive_runtime")
        report = run_body["runtime_smoke_report"]
        explanation = run_body["runtime_result_explanation"]
        self.assertEqual(report["events_requested"], 4)
        self.assertEqual(report["events_completed"], 4)
        self.assertEqual(report["configuration"]["geometry_structure"], "single_box")
        self.assertEqual(report["configuration"]["particle"], "gamma")
        self.assertEqual(report["configuration"]["physics_list"], "FTFP_BERT")
        self.assertIn("result_summary", report)
        self.assertEqual(explanation["source"], "deterministic")
        self.assertIn("4 / 4", explanation["message"])
        self.assertEqual(explanation["prompt_profile_id"], "runtime_result_explain_zh_v1")
        self.assertTrue(explanation["prompt_validation"]["ok"])

        summary_status, summary_body = geant4_api.handle_geant4_post("/api/geant4/summary", {})
        self.assertEqual(summary_status, 200)
        self.assertEqual(summary_body["action_safety_class"], "read_only")
        self.assertEqual(summary_body["runtime_smoke_report"]["events_completed"], 4)
        self.assertEqual(summary_body["runtime_smoke_report"]["configuration"]["particle"], "gamma")
        self.assertIn("runtime_result_explanation", summary_body)


class RuntimeResultFrontendStaticTest(unittest.TestCase):
    def test_frontend_exposes_runtime_result_card_and_formatter(self) -> None:
        index_html = Path("ui/web/index.html").read_text(encoding="utf-8")
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")

        self.assertIn('id="runtime-result-summary"', index_html)
        self.assertIn("renderRuntimeResultSummary", app_js)
        self.assertIn("runtime_smoke_report", app_js)
        self.assertIn("runtime_result_explanation", app_js)
        self.assertIn("isRuntimeResultQuestion", app_js)
        self.assertIn("classifyRuntimeIntent", app_js)
        self.assertIn("answerRuntimeResultQuestion", app_js)
        self.assertIn("/api/geant4/summary", app_js)

    def test_frontend_runtime_result_question_uses_summary_not_run(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        question_branch = app_js[
            app_js.index('runtimeIntent.intent === "read_summary"') : app_js.index("const payload = {", app_js.index('runtimeIntent.intent === "read_summary"'))
        ]

        self.assertIn("answerRuntimeResultQuestion", question_branch)
        self.assertIn("return;", question_branch)
        self.assertNotIn("/api/geant4/run", question_branch)
        self.assertNotIn("/api/step_async", question_branch)

    def test_frontend_explicit_runtime_requests_do_not_auto_run_from_chat(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        action_branch = app_js[
            app_js.index('runtimeIntent.intent === "run_requested"') : app_js.index("const payload = {", app_js.index('runtimeIntent.intent === "run_requested"'))
        ]

        self.assertIn("explicitRuntimeActionMessage", action_branch)
        self.assertIn("return;", action_branch)
        self.assertNotIn("/api/geant4/run", action_branch)
        self.assertNotIn("/api/geant4/viewer/open", action_branch)


if __name__ == "__main__":
    unittest.main()

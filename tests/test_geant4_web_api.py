from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

from mcp.geant4.adapter import InMemoryGeant4Adapter, LocalProcessGeant4Adapter
from mcp.geant4.server import Geant4McpServer
import ui.web.geant4_api as geant4_api


def _runtime_patch() -> dict:
    return {
        "geometry": {"structure": "single_box"},
        "source": {"type": "point", "particle": "gamma"},
        "physics_list": {"name": "FTFP_BERT"},
    }


def _complete_runtime_patch() -> dict:
    return {
        "geometry": {
            "structure": "single_box",
            "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
        },
        "source": {
            "type": "point",
            "particle": "gamma",
            "energy": 1.0,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "physics_list": {"name": "FTFP_BERT"},
    }


class Geant4WebApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_server = geant4_api._GEANT4_SERVER
        self._previous_idempotency_policy = geant4_api._IDEMPOTENCY_POLICY
        geant4_api._GEANT4_SERVER = Geant4McpServer(adapter=InMemoryGeant4Adapter())
        geant4_api._IDEMPOTENCY_POLICY = geant4_api.IdempotencyReplayPolicy()

    def tearDown(self) -> None:
        geant4_api._GEANT4_SERVER = self._previous_server
        geant4_api._IDEMPOTENCY_POLICY = self._previous_idempotency_policy

    def test_default_web_server_uses_in_memory_without_runtime_env(self) -> None:
        geant4_api._GEANT4_SERVER = None
        with mock.patch.dict(
            os.environ,
            {"GEANT4_RUNTIME_COMMAND_JSON": "", "GEANT4_RUNTIME_COMMAND": ""},
        ):
            server = geant4_api._build_server()
            geant4_api._GEANT4_SERVER = server
            state = geant4_api.geant4_state_payload()

        adapter = server._adapter  # type: ignore[attr-defined]
        self.assertIsInstance(adapter, InMemoryGeant4Adapter)
        self.assertIn("metadata", state)
        self.assertEqual(state["metadata"]["adapter"], "in_memory")

    def test_web_server_uses_local_process_when_runtime_env_is_configured(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GEANT4_RUNTIME_COMMAND_JSON": json.dumps([sys.executable, "-c", "print('ok')"]),
                "GEANT4_ROOT": "F:\\Geant4Test",
                "GEANT4_WORKING_DIR": "F:\\geant4agent",
            },
        ):
            server = geant4_api._build_server()

        adapter = server._adapter  # type: ignore[attr-defined]
        self.assertIsInstance(adapter, LocalProcessGeant4Adapter)
        snapshot = adapter.snapshot()
        self.assertEqual(snapshot.metadata["adapter"], "local_process")
        self.assertEqual(snapshot.metadata["geant4_root"], "F:\\Geant4Test")

    def test_viewer_open_without_runtime_env_returns_guarded_failure(self) -> None:
        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/viewer/open",
            {"patch": _runtime_patch(), "events": 2},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["action_safety_class"], "expensive_runtime")
        self.assertIn("local_process_required", body["errors"])
        self.assertIn("missing_runtime_command", body["errors"])

    def test_validate_config_is_read_only_and_reports_missing_fields(self) -> None:
        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/validate",
            {
                "events": 5,
                "config": {"geometry": {"structure": "single_box", "params": {"module_x": 10.0}}},
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["action_safety_class"], "read_only")
        self.assertFalse(body["payload"]["ok"])
        self.assertIn("source.type", body["payload"]["missing_paths"])
        self.assertIn("physics.physics_list", body["payload"]["missing_paths"])
        state = geant4_api.geant4_state_payload()
        self.assertEqual(state["runtime_phase"], "idle")
        self.assertFalse(state["geometry_ready"])

    def test_validate_config_returns_runtime_payload_preview_for_complete_config(self) -> None:
        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/validate",
            {
                "events": 7,
                "config": {
                    "geometry": {
                        "structure": "single_box",
                        "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
                    },
                    "source": {
                        "type": "point",
                        "particle": "gamma",
                        "energy": 1.0,
                        "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
                        "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
                    },
                    "physics_list": {"name": "FTFP_BERT"},
                },
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["action_safety_class"], "read_only")
        self.assertTrue(body["payload"]["ok"])
        preview = body["payload"]["runtime_payload_preview"]
        self.assertEqual(preview["structure"], "single_box")
        self.assertEqual(preview["source_type"], "point")
        self.assertEqual(preview["run"]["events"], 7)

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

        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/intent",
            {"text": "What is the current configured source?", "lang": "en"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "read_config")
        self.assertEqual(body["action_safety_class"], "read_only")

        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/intent",
            {"text": "Change source energy to 1 MeV", "lang": "en"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "config_mutation")
        self.assertEqual(body["action_safety_class"], "config_mutation")

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

        qa_status, qa_body = geant4_api.handle_geant4_post(
            "/api/geant4/summary",
            {"lang": "en", "question": "What was the dose?"},
        )
        self.assertEqual(qa_status, 200)
        self.assertEqual(qa_body["runtime_result_explanation"]["prompt_profile_id"], "runtime_result_qa_en_v1")
        self.assertIn("does not report dose", qa_body["runtime_result_explanation"]["message"])

    def test_run_without_action_id_keeps_compatibility_and_returns_suggested_action_id(self) -> None:
        geant4_api.handle_geant4_post("/api/geant4/apply", {"patch": _runtime_patch()})
        geant4_api.handle_geant4_post("/api/geant4/initialize", {})

        status, body = geant4_api.handle_geant4_post("/api/geant4/run", {"events": 2})

        self.assertEqual(status, 200)
        self.assertFalse(body["idempotency"]["enabled"])
        self.assertEqual(body["idempotency"]["action_name"], "run_beam")
        self.assertTrue(body["idempotency"]["suggested_action_id"])

    def test_duplicate_run_with_action_id_replays_result_without_rerun(self) -> None:
        geant4_api.handle_geant4_post("/api/geant4/apply", {"patch": _runtime_patch()})
        geant4_api.handle_geant4_post("/api/geant4/initialize", {})
        server = geant4_api.get_geant4_server()

        with mock.patch.object(server, "call_tool", wraps=server.call_tool) as call_tool:
            first_status, first_body = geant4_api.handle_geant4_post(
                "/api/geant4/run",
                {"events": 3, "action_id": "run-action-001"},
            )
            second_status, second_body = geant4_api.handle_geant4_post(
                "/api/geant4/run",
                {"events": 3, "action_id": "run-action-001"},
            )

        self.assertEqual(first_status, 200)
        self.assertEqual(second_status, 200)
        self.assertEqual(call_tool.call_count, 1)
        self.assertEqual(second_body["idempotency"]["decision"], "replay_result")
        self.assertEqual(second_body["runtime_smoke_report"]["events_completed"], 3)
        self.assertEqual(second_body["runtime_smoke_report"], first_body["runtime_smoke_report"])

    def test_reusing_run_action_id_for_different_events_is_rejected(self) -> None:
        geant4_api.handle_geant4_post("/api/geant4/apply", {"patch": _runtime_patch()})
        geant4_api.handle_geant4_post("/api/geant4/initialize", {})
        geant4_api.handle_geant4_post("/api/geant4/run", {"events": 3, "action_id": "run-action-002"})

        status, body = geant4_api.handle_geant4_post(
            "/api/geant4/run",
            {"events": 4, "action_id": "run-action-002"},
        )

        self.assertEqual(status, 409)
        self.assertEqual(body["status"], "rejected")
        self.assertEqual(body["idempotency"]["decision"], "conflict")

    def test_duplicate_viewer_action_id_is_rejected_without_relaunch(self) -> None:
        adapter = LocalProcessGeant4Adapter(
            [sys.executable, "-c", "print('viewer_pid=1234')"],
            geant4_root="F:\\Geant4Test",
            working_dir="F:\\geant4agent",
        )
        geant4_api._GEANT4_SERVER = Geant4McpServer(adapter=adapter)

        first_status, first_body = geant4_api.handle_geant4_post(
            "/api/geant4/viewer/open",
            {"patch": _complete_runtime_patch(), "events": 2, "action_id": "viewer-action-001"},
        )
        second_status, second_body = geant4_api.handle_geant4_post(
            "/api/geant4/viewer/open",
            {"patch": _complete_runtime_patch(), "events": 2, "action_id": "viewer-action-001"},
        )

        self.assertEqual(first_status, 200)
        self.assertEqual(first_body["payload"]["viewer_pid"], 1234)
        self.assertEqual(second_status, 409)
        self.assertEqual(second_body["status"], "rejected")
        self.assertEqual(second_body["idempotency"]["decision"], "reject_duplicate")


class RuntimeResultFrontendStaticTest(unittest.TestCase):
    def test_frontend_exposes_runtime_result_card_and_formatter(self) -> None:
        index_html = Path("ui/web/index.html").read_text(encoding="utf-8")
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")

        self.assertIn('id="runtime-result-summary"', index_html)
        self.assertIn("renderRuntimeResultSummary", app_js)
        self.assertIn("runtime_smoke_report", app_js)
        self.assertIn("runtime_result_explanation", app_js)
        self.assertIn("isRuntimeResultQuestion", app_js)
        self.assertIn("isConfigQuestion", app_js)
        self.assertIn("classifyRuntimeIntent", app_js)
        self.assertIn("answerRuntimeResultQuestion", app_js)
        self.assertIn("answerConfigQuestion", app_js)
        self.assertIn("normalChatReadOnlyMessage", app_js)
        self.assertIn("question: questionText", app_js)
        self.assertIn("/api/geant4/summary", app_js)
        self.assertIn("/api/config/summary", app_js)
        self.assertIn("validateGeant4Config", app_js)
        self.assertIn("runtimePreflightMessage", app_js)
        self.assertIn("/api/geant4/validate", app_js)
        self.assertIn("metadata.adapter", app_js)

    def test_frontend_runtime_result_question_uses_summary_not_run(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        question_branch = app_js[
            app_js.index('runtimeIntent.intent === "read_summary"') : app_js.index("const payload = {", app_js.index('runtimeIntent.intent === "read_summary"'))
        ]

        self.assertIn("answerRuntimeResultQuestion", question_branch)
        self.assertIn("return;", question_branch)
        self.assertNotIn("/api/geant4/run", question_branch)
        self.assertNotIn("/api/step_async", question_branch)

    def test_frontend_config_question_uses_config_summary_not_step(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        question_branch = app_js[
            app_js.index('runtimeIntent.intent === "read_config"') : app_js.index("const payload = {", app_js.index('runtimeIntent.intent === "read_config"'))
        ]

        self.assertIn("answerConfigQuestion", question_branch)
        self.assertIn("return;", question_branch)
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

    def test_frontend_only_config_mutation_reaches_step_async(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        guard_branch = app_js[
            app_js.index('runtimeIntent.intent !== "config_mutation"') : app_js.index("const payload = {", app_js.index('runtimeIntent.intent !== "config_mutation"'))
        ]

        self.assertIn("normalChatReadOnlyMessage", guard_branch)
        self.assertIn("return;", guard_branch)
        self.assertNotIn("/api/step_async", guard_branch)

    def test_frontend_sync_config_preflights_before_apply(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        branch = app_js[
            app_js.index("async function syncGeant4Config") : app_js.index("async function initializeGeant4")
        ]

        self.assertIn("validateGeant4Config", branch)
        self.assertIn("return;", branch)
        self.assertLess(branch.index("validateGeant4Config"), branch.index('"/api/geant4/apply"'))

    def test_frontend_initialize_preflights_before_initialize(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        branch = app_js[
            app_js.index("async function initializeGeant4") : app_js.index("async function openGeant4Viewer")
        ]

        self.assertIn("validateGeant4Config", branch)
        self.assertIn("return;", branch)
        self.assertLess(branch.index("validateGeant4Config"), branch.index('"/api/geant4/initialize"'))

    def test_frontend_viewer_preflights_before_viewer(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        branch = app_js[
            app_js.index("async function openGeant4Viewer") : app_js.index("async function runGeant4")
        ]

        self.assertIn("validateGeant4Config", branch)
        self.assertIn("return;", branch)
        self.assertLess(branch.index("validateGeant4Config"), branch.index('"/api/geant4/viewer/open"'))

    def test_frontend_run_preflights_before_run(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        branch = app_js[
            app_js.index("async function runGeant4") : app_js.index("async function loadRuntimeConfigs")
        ]

        self.assertIn("validateGeant4Config", branch)
        self.assertIn("return;", branch)
        self.assertLess(branch.index("validateGeant4Config"), branch.index('"/api/geant4/run"'))


if __name__ == "__main__":
    unittest.main()

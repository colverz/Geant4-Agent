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
from core.orchestrator.session_manager import reset_session
from ui.web.request_router import handle_post_request


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
        self.assertIn("runtime_capabilities", state)
        self.assertIn("step_wedge", state["runtime_capabilities"]["geometry_primitives"])
        self.assertTrue(state["runtime_capabilities"]["region_scoring_support"])

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

    def test_simulation_design_recommended_config_can_run_full_runtime_chain(self) -> None:
        session_id = "design-runtime-chain"
        reset_session(session_id)
        common = {"legacy_sessions": {}, "solve_fn": lambda payload: {}, "step_fn": lambda payload: {}}
        try:
            design_status, design_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm along +z; physics FTFP_BERT; output json",
                    "lang": "en",
                    "llm_router": False,
                },
                **common,
            )
            config = design_body.get("recommended_config") or {}
            validate_status, validate_body = handle_post_request(
                "/api/geant4/validate",
                {"patch": config, "events": 1},
                **common,
            )
            apply_status, _ = handle_post_request("/api/geant4/apply", {"patch": config}, **common)
            init_status, _ = handle_post_request("/api/geant4/initialize", {}, **common)
            run_status, run_body = handle_post_request(
                "/api/geant4/run",
                {"session_id": session_id, "events": 1, "action_id": "design-runtime-chain-run"},
                **common,
            )
            summary_status, summary_body = handle_post_request("/api/geant4/summary", {"lang": "en"}, **common)

            self.assertEqual(design_status, 200)
            self.assertTrue(config)
            self.assertIn("agent_plan", design_body)
            self.assertEqual(design_body["agent_state"]["plan_proposed"], True)
            self.assertEqual(validate_status, 200)
            self.assertTrue(validate_body["payload"]["ok"])
            self.assertEqual(validate_body["payload"]["missing_paths"], [])
            self.assertEqual(apply_status, 200)
            self.assertEqual(init_status, 200)
            self.assertEqual(run_status, 200)
            self.assertEqual(run_body["runtime_smoke_report"]["events_completed"], 1)
            self.assertIn("critic_report", run_body)
            self.assertEqual(summary_status, 200)
            self.assertEqual(summary_body["runtime_smoke_report"]["events_completed"], 1)
            state_status, state_body = handle_post_request("/api/agent/state", {"session_id": session_id}, **common)
            self.assertEqual(state_status, 200)
            self.assertTrue(state_body["agent_state"]["result_available"])
            self.assertEqual(state_body["agent_plan"]["selected_candidate_id"], "candidate_1")
        finally:
            reset_session(session_id)

    def test_run_can_use_session_recommended_config_without_frontend_patch(self) -> None:
        session_id = "design-runtime-session-fallback"
        reset_session(session_id)
        common = {"legacy_sessions": {}, "solve_fn": lambda payload: {}, "step_fn": lambda payload: {}}
        try:
            design_status, design_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "10 mm copper box target; gamma point source 1 MeV along +z; physics FTFP_BERT; output json",
                    "lang": "en",
                    "llm_router": False,
                },
                **common,
            )
            validate_status, validate_body = handle_post_request(
                "/api/geant4/validate",
                {"session_id": session_id, "events": 1},
                **common,
            )
            run_status, run_body = handle_post_request(
                "/api/geant4/run",
                {"session_id": session_id, "events": 1, "action_id": "design-runtime-session-fallback-run"},
                **common,
            )

            self.assertEqual(design_status, 200)
            self.assertTrue(design_body.get("recommended_config"))
            self.assertEqual(validate_status, 200)
            self.assertTrue(validate_body["payload"]["ok"])
            self.assertEqual(validate_body["payload"]["missing_paths"], [])
            self.assertEqual(run_status, 200)
            self.assertEqual(run_body["runtime_smoke_report"]["events_completed"], 1)
        finally:
            reset_session(session_id)

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
    def test_frontend_is_rebuilt_without_legacy_ui_shell(self) -> None:
        index_html = Path("ui/web/index.html").read_text(encoding="utf-8")
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        css = Path("ui/web/style.css").read_text(encoding="utf-8")

        self.assertIn('data-ui-version="rebuild-v1"', index_html)
        self.assertIn("conversation-stage", index_html)
        self.assertIn("activity-strip", index_html)
        self.assertIn("context-rail", index_html)
        self.assertIn("evidence-drawer", index_html)
        self.assertNotIn("sidebar", index_html)
        self.assertNotIn("inspector-card", index_html)
        self.assertNotIn("debug-panel", index_html)
        self.assertIn("Quiet research console", css)
        self.assertIn("Anthropic/Claude", css)
        self.assertNotIn("v3 visual system", css)
        self.assertNotIn("v4 conversation-first interface", css)
        self.assertIn("activity-trace", css)
        self.assertIn("context-rail", css)
        self.assertIn("100dvh", css)
        self.assertIn("@media (max-width: 1120px)", css)
        self.assertIn("@media (max-width: 760px)", css)
        self.assertIn("designMessage", app_js)
        self.assertIn("activityHtml", app_js)
        self.assertIn("data-agent-activity", app_js)
        self.assertIn("acceptCandidate", app_js)
        self.assertIn('"/api/simulation/accept"', app_js)
        self.assertIn("window-close-btn", index_html)
        self.assertIn("window-controls", index_html)
        self.assertIn("window.geant4Desktop?.close", app_js)
        self.assertIn("#f4f0e8", css)
        self.assertIn("#8a3f2d", css)

    def test_frontend_uses_simulation_design_as_default_conversation_path(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        index_html = Path("ui/web/index.html").read_text(encoding="utf-8")
        self.assertIn('"/api/simulation/design"', app_js)
        self.assertIn('"/api/simulation/accept"', app_js)
        self.assertIn("requestDesign(input)", app_js)
        self.assertIn("applyDesignResponse(data)", app_js)
        self.assertIn("appendAgent(designMessage(data)", app_js)
        self.assertIn("No runnable candidate existed, so the agent designed one first.", app_js)
        self.assertNotIn("run1-btn", index_html)
        self.assertNotIn("run10-btn", index_html)
        self.assertNotIn("validate-btn", index_html)
        self.assertNotIn('"/api/step_async"', app_js)

    def test_frontend_read_only_questions_do_not_run_or_write_config(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        summary_branch = app_js[
            app_js.index('intent.intent === "read_summary"') : app_js.index('intent.intent === "read_config"')
        ]
        config_branch = app_js[
            app_js.index('intent.intent === "read_config"') : app_js.index('intent.intent === "run_requested"')
        ]

        self.assertIn("answerRuntimeQuestion(input)", summary_branch)
        self.assertIn('"/api/geant4/summary"', app_js)
        self.assertNotIn("/api/geant4/run", summary_branch)
        self.assertNotIn("/api/simulation/design", summary_branch)
        self.assertIn("answerConfigQuestion()", config_branch)
        self.assertIn('"/api/config/summary"', app_js)
        self.assertNotIn("/api/geant4/run", config_branch)
        self.assertNotIn("/api/simulation/design", config_branch)

    def test_frontend_runtime_actions_validate_before_run_and_viewer(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        run_branch = app_js[app_js.index("async function runGeant4") : app_js.index("async function openViewer")]
        viewer_branch = app_js[app_js.index("async function openViewer") : app_js.index("async function refreshGeant4State")]

        self.assertIn("validateGeant4Config(events, true)", run_branch)
        self.assertIn("ensureCandidateCommitted()", run_branch)
        self.assertLess(run_branch.index("ensureCandidateCommitted()"), run_branch.index('"/api/geant4/run"'))
        self.assertLess(run_branch.index("validateGeant4Config(events, true)"), run_branch.index('"/api/geant4/run"'))
        self.assertIn("actionToken(\"run_beam\"", run_branch)
        self.assertIn("action_id", run_branch)
        self.assertIn('"/api/geant4/viewer/open"', viewer_branch)
        self.assertIn("actionToken(\"viewer_open\"", viewer_branch)

    def test_frontend_keeps_latest_recommended_config_as_runtime_patch(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        self.assertIn("lastRecommendedConfig", app_js)
        self.assertIn("state.lastRecommendedConfig = data.recommended_config || data.config || state.lastRecommendedConfig", app_js)
        self.assertIn("function runtimePatch()", app_js)
        self.assertIn("return state.lastRecommendedConfig || {}", app_js)

    def test_frontend_model_switch_uses_runtime_available_configs(self) -> None:
        app_js = Path("ui/web/app.js").read_text(encoding="utf-8")
        runtime_branch = app_js[app_js.index("async function loadRuntimeConfig") : app_js.index("async function setRuntimeConfig")]
        switch_branch = app_js[app_js.index("async function setRuntimeConfig") : app_js.index("async function sendPrompt")]

        self.assertIn("data.available", runtime_branch)
        self.assertNotIn("data.config_paths", runtime_branch)
        self.assertIn("item.provider", runtime_branch)
        self.assertIn("item.model", runtime_branch)
        self.assertIn('"/api/runtime"', switch_branch)
        self.assertIn("config_path: path", switch_branch)
        self.assertIn("await loadRuntimeConfig()", switch_branch)

    def test_ui_rebuild_plan_document_exists(self) -> None:
        doc = Path("docs/ui/GEANT4_AGENT_UI_REBUILD_PLAN_CN.md").read_text(encoding="utf-8")
        self.assertIn("当前 UI 不再继续修补", doc)
        self.assertIn("Conversation Stage", doc)
        self.assertIn("Runtime Command Bar", doc)
        self.assertIn("Evidence Drawer", doc)

    def test_desktop_chromium_shell_launch_contract(self) -> None:
        main_js = Path("ui/desktop/main.js").read_text(encoding="utf-8")
        preload_js = Path("ui/desktop/preload.js").read_text(encoding="utf-8")
        package_json = Path("ui/desktop/package.json").read_text(encoding="utf-8")
        start_ps1 = Path("ui/desktop/start_desktop.ps1").read_text(encoding="utf-8")

        self.assertIn("BrowserWindow", main_js)
        self.assertIn("?desktop=1", main_js)
        self.assertIn("ui.desktop.runtime_bridge", main_js)
        self.assertIn("setWindowOpenHandler", main_js)
        self.assertIn("ipcMain.handle(\"window:close\"", main_js)
        self.assertIn("ipcMain.handle(\"window:minimize\"", main_js)
        self.assertIn("ipcMain.handle(\"window:toggle-maximize\"", main_js)
        self.assertIn("contextBridge", preload_js)
        self.assertIn("geant4Desktop", preload_js)
        self.assertIn("close: () => ipcRenderer.invoke(\"window:close\")", preload_js)
        self.assertIn('"electron"', package_json)
        self.assertIn("npm start", start_ps1)


if __name__ == "__main__":
    unittest.main()

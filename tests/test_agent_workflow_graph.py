from __future__ import annotations

import unittest

from core.agent.intent_router import route_user_turn
from core.agent.composite_intent import detect_composite_intent
from core.agent.workflow_graph import WorkflowNode, WorkflowTerminalState, assert_path_invariants, graph_path_for_intent, terminal_state_for_intent
from core.orchestrator.session_manager import process_turn, reset_session
from core.runtime.types import ActionSafetyClass


class AgentWorkflowGraphTest(unittest.TestCase):
    def test_read_config_path_never_reaches_apply_session(self) -> None:
        path = graph_path_for_intent("read_config")

        self.assertNotIn(WorkflowNode.APPLY_SESSION, path)
        assert_path_invariants("read_config", path)
        self.assertEqual(terminal_state_for_intent("read_config"), WorkflowTerminalState.READ_ONLY_ANSWER)

    def test_normal_chat_path_never_reaches_validate(self) -> None:
        path = graph_path_for_intent("normal_chat")

        self.assertNotIn(WorkflowNode.VALIDATE, path)
        assert_path_invariants("normal_chat", path)

    def test_config_mutation_path_validates_before_apply(self) -> None:
        path = graph_path_for_intent("config_mutation", mutation_applied=True)

        self.assertIn(WorkflowNode.VALIDATE, path)
        self.assertIn(WorkflowNode.APPLY_SESSION, path)
        self.assertLess(path.index(WorkflowNode.VALIDATE), path.index(WorkflowNode.APPLY_SESSION))
        assert_path_invariants("config_mutation", path)

    def test_runtime_requests_end_as_guarded_runtime_action(self) -> None:
        for intent in ("run_requested", "viewer_requested"):
            with self.subTest(intent=intent):
                path = graph_path_for_intent(intent)
                self.assertIn(WorkflowNode.RUNTIME_GUARD, path)
                self.assertNotIn(WorkflowNode.APPLY_SESSION, path)
                self.assertEqual(terminal_state_for_intent(intent), WorkflowTerminalState.RUNTIME_ACTION_GUARDED)
                assert_path_invariants(intent, path)

    def test_intent_router_returns_typed_decision(self) -> None:
        decision = route_user_turn("run 10 events now", "en")

        self.assertEqual(decision.intent, "run_requested")
        self.assertEqual(decision.safety_class, ActionSafetyClass.EXPENSIVE_RUNTIME)
        self.assertFalse(decision.requires_kb)
        self.assertIn(WorkflowNode.RUNTIME_GUARD, decision.allowed_next_nodes)
        self.assertTrue(decision.prompt_validation["ok"])

    def test_composite_intent_detects_mutation_plus_runtime_request(self) -> None:
        composite = detect_composite_intent("Change source energy to 10 MeV and run 10 events now.")

        self.assertTrue(composite.has_config_mutation)
        self.assertTrue(composite.has_runtime_request)
        self.assertTrue(composite.requires_staged_runtime_guard)

    def test_composite_intent_detects_run_it_after_mutation(self) -> None:
        composite = detect_composite_intent("Build a full CT scanner gantry with rotating source and run it.")

        self.assertTrue(composite.has_config_mutation)
        self.assertTrue(composite.has_runtime_request)
        self.assertTrue(composite.requires_staged_runtime_guard)

    def test_process_turn_exposes_nlu_turn_trace_without_changing_behavior(self) -> None:
        session_id = "agent-workflow-trace"
        reset_session(session_id)
        out = process_turn(
            {
                "session_id": session_id,
                "text": (
                    "10 mm x 20 mm x 30 mm copper box target; "
                    "gamma point source 1 MeV at (0,0,-20) mm along +z; "
                    "physics FTFP_BERT; output json."
                ),
                "llm_router": False,
                "llm_question": False,
                "normalize_input": True,
                "geometry_pipeline": "v2",
                "source_pipeline": "v2",
                "enable_compare": False,
            },
            ollama_config_path="",
            lang="en",
        )
        try:
            self.assertTrue(out["is_complete"])
            trace = out["nlu_turn_trace"]
            self.assertEqual(trace["schema_version"], "nlu_turn_trace.v1")
            self.assertEqual(trace["intent"], "config_mutation")
            self.assertEqual(trace["action_safety_class"], "config_mutation")
            self.assertIn("validate", trace["node_sequence"])
            self.assertIn("apply_session", trace["node_sequence"])
            self.assertTrue(trace["runtime_payload_ready"])
            self.assertEqual(trace["confirmation_id"], "")
            self.assertEqual(trace["confirmation_patch_hash"], "")
            self.assertIn("geometry.structure", trace["applied_paths"])
            self.assertIn("source.particle", trace["applied_paths"])
            self.assertEqual(out["internal_trace"]["agent"]["nlu_turn_trace"], trace)
        finally:
            reset_session(session_id)

    def test_process_turn_trace_marks_composite_runtime_intent_pending(self) -> None:
        session_id = "agent-workflow-composite-trace"
        reset_session(session_id)
        out = process_turn(
            {
                "session_id": session_id,
                "text": "Change source energy to 10 MeV and run 10 events now.",
                "llm_router": False,
                "llm_question": False,
                "normalize_input": True,
                "geometry_pipeline": "v2",
                "source_pipeline": "v2",
                "enable_compare": False,
            },
            ollama_config_path="",
            lang="en",
        )
        try:
            trace = out["nlu_turn_trace"]
            self.assertTrue(trace["composite_intent"]["has_config_mutation"])
            self.assertTrue(trace["composite_intent"]["has_runtime_request"])
            self.assertTrue(trace["guarded_runtime_intent_pending"])
            self.assertIn("run_beam", trace["tool_calls_blocked"])
        finally:
            reset_session(session_id)


if __name__ == "__main__":
    unittest.main()

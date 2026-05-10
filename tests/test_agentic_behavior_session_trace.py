from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.orchestrator.session_manager import process_turn, reset_session


CASEBANK_PATH = Path("docs/eval/agentic_behavior_casebank.json")


class AgenticBehaviorSessionTraceTest(unittest.TestCase):
    def test_agentic_casebank_trace_matches_process_turn(self) -> None:
        cases = json.loads(CASEBANK_PATH.read_text(encoding="utf-8"))
        checked = 0

        for index, case in enumerate(cases):
            expected = dict(case.get("expected_session_trace") or {})
            if not expected:
                continue
            checked += 1
            session_id = f"agentic-session-trace-{index}"
            reset_session(session_id)
            try:
                out = process_turn(
                    {
                        "session_id": session_id,
                        "text": case["text"],
                        "llm_router": False,
                        "llm_question": False,
                        "normalize_input": True,
                        "geometry_pipeline": "v2",
                        "source_pipeline": "v2",
                        "enable_compare": False,
                    },
                    ollama_config_path="",
                    lang=case["lang"],
                )
                trace = dict(out.get("nlu_turn_trace") or {})

                with self.subTest(case_id=case["id"]):
                    self.assertEqual(trace.get("intent"), expected["intent"])
                    self.assertEqual(trace.get("action_safety_class"), expected["safety"])
                    for node in expected.get("must_include_nodes", []):
                        self.assertIn(node, trace.get("node_sequence", []))
                    if expected.get("must_not_apply_session"):
                        self.assertNotIn("apply_session", trace.get("node_sequence", []))
                        self.assertEqual(trace.get("applied_paths"), [])
                    for tool_name in expected.get("must_block_tools", []):
                        self.assertIn(tool_name, trace.get("tool_calls_blocked", []))
                    self.assertEqual(
                        bool(trace.get("guarded_runtime_intent_pending")),
                        bool(expected.get("guarded_runtime_intent_pending")),
                    )
            finally:
                reset_session(session_id)

        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

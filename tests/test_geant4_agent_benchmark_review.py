from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from tools.review_geant4_agent_benchmark import (
    REVIEW_SCHEMA_VERSION,
    build_review_prompt,
    review_benchmark_with_llm,
)


class Geant4AgentBenchmarkReviewTest(unittest.TestCase):
    def test_review_prompt_contains_benchmark_but_not_local_config(self) -> None:
        prompt = build_review_prompt(Path("docs/eval/agentic_benchmark_v1.json"))

        self.assertIn("Geant4 simulation agent", prompt)
        self.assertIn("smoke-read-config-readonly", prompt)
        self.assertNotIn("api_key", prompt.lower())
        self.assertNotIn("authorization", prompt.lower())

    def test_dry_run_does_not_call_llm(self) -> None:
        with mock.patch("tools.review_geant4_agent_benchmark.chat") as fake_chat:
            report = review_benchmark_with_llm(
                benchmark_path=Path("docs/eval/agentic_benchmark_v1.json"),
                llm_config_path="dummy.json",
                models=["deepseek-v4-flash"],
                live_llm=False,
            )

        fake_chat.assert_not_called()
        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "dry_run")

    def test_live_review_parses_json_response_and_restores_model_override(self) -> None:
        response = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "overall_score": 0.82,
            "passes_review": True,
            "dimension_scores": {
                "necessary": 0.9,
                "comprehensive": 0.75,
                "non_dictionary": 0.85,
                "measurable": 0.9,
                "p7_readiness": 0.8,
            },
            "major_issues": [],
            "missing_capabilities": ["llm_reliability"],
            "recommended_changes": [{"priority": "p1", "change": "add live_llm smoke", "reason": "P7 coverage"}],
            "case_feedback": [{"id": "smoke-read-config-readonly", "verdict": "keep", "reason": "clear read-only guard"}],
        }
        previous = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
        previous_timeout = os.environ.get("GEANT4_LLM_TIMEOUT_S")
        seen_timeouts: list[str | None] = []

        def fake_live_chat(*args, **kwargs):
            seen_timeouts.append(os.environ.get("GEANT4_LLM_TIMEOUT_S"))
            return {"response": json.dumps(response)}

        with mock.patch(
            "tools.review_geant4_agent_benchmark.chat",
            side_effect=fake_live_chat,
        ) as fake_chat:
            report = review_benchmark_with_llm(
                benchmark_path=Path("docs/eval/agentic_benchmark_v1.json"),
                llm_config_path="dummy.json",
                models=["deepseek-v4-flash", "deepseek-v4-pro"],
                live_llm=True,
                timeout_s=123,
            )

        self.assertEqual(os.environ.get("GEANT4_LLM_MODEL_OVERRIDE"), previous)
        self.assertEqual(os.environ.get("GEANT4_LLM_TIMEOUT_S"), previous_timeout)
        self.assertEqual(seen_timeouts, ["123", "123"])
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["reviews"]), 2)
        self.assertEqual(fake_chat.call_count, 2)
        self.assertEqual(report["reviews"][0]["review"]["overall_score"], 0.82)


if __name__ == "__main__":
    unittest.main()

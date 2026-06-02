from __future__ import annotations

import unittest

from core.agent_v3.intent_classifier import V3Intent, IntentResult, classify_intent_llm


class IntentClassifierTest(unittest.TestCase):
    def test_classify_intent_llm_returns_unknown_on_invalid_config(self) -> None:
        result = classify_intent_llm("设计一个铅屏蔽方案", config_path="nonexistent.json")
        self.assertEqual(result.intent, V3Intent.UNKNOWN)
        self.assertEqual(result.source, "llm")

    def test_result_is_serializable(self) -> None:
        result = IntentResult(intent=V3Intent.CONFIRM, confidence=0.9, source="llm")
        d = result.to_dict()
        self.assertEqual(d["intent"], "confirm")
        self.assertEqual(d["source"], "llm")
        self.assertTrue(isinstance(d["confidence"], float))

    def test_v3_intent_values(self) -> None:
        self.assertEqual(V3Intent.DESIGN_NEW.value, "design_new")
        self.assertEqual(V3Intent.CONFIRM.value, "confirm")
        self.assertEqual(V3Intent.UNKNOWN.value, "unknown")

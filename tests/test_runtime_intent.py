from __future__ import annotations

import unittest

from core.runtime.types import ActionSafetyClass
from planner.runtime_intent import RuntimeIntent, classify_user_runtime_intent


class RuntimeIntentTest(unittest.TestCase):
    def test_result_question_is_read_only(self) -> None:
        result = classify_user_runtime_intent("刚才模拟结果怎么样？", "zh")

        self.assertEqual(result.intent, RuntimeIntent.READ_SUMMARY)
        self.assertEqual(result.action_safety_class, ActionSafetyClass.READ_ONLY)
        self.assertTrue(result.prompt_validation["ok"])

    def test_english_result_question_is_read_only(self) -> None:
        result = classify_user_runtime_intent("What was the latest simulation result?", "en")

        self.assertEqual(result.intent, RuntimeIntent.READ_SUMMARY)
        self.assertEqual(result.action_safety_class, ActionSafetyClass.READ_ONLY)

    def test_run_request_is_expensive_runtime(self) -> None:
        result = classify_user_runtime_intent("run 10 events now", "en")

        self.assertEqual(result.intent, RuntimeIntent.RUN_REQUESTED)
        self.assertEqual(result.action_safety_class, ActionSafetyClass.EXPENSIVE_RUNTIME)

    def test_viewer_request_is_expensive_runtime(self) -> None:
        result = classify_user_runtime_intent("打开 Geant4 viewer", "zh")

        self.assertEqual(result.intent, RuntimeIntent.VIEWER_REQUESTED)
        self.assertEqual(result.action_safety_class, ActionSafetyClass.EXPENSIVE_RUNTIME)

    def test_normal_chat_is_read_only(self) -> None:
        result = classify_user_runtime_intent("把源能量改成 1 MeV", "zh")

        self.assertEqual(result.intent, RuntimeIntent.NORMAL_CHAT)
        self.assertEqual(result.action_safety_class, ActionSafetyClass.READ_ONLY)


if __name__ == "__main__":
    unittest.main()

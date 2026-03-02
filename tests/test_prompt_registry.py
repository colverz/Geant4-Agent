from __future__ import annotations

import unittest

from core.config.prompt_registry import (
    clarification_fallback,
    clarification_prompt,
    completion_message,
    single_field_request,
)


class PromptRegistryTest(unittest.TestCase):
    def test_completion_and_single_field_requests_are_shared(self) -> None:
        self.assertEqual(completion_message("en"), "Configuration complete.")
        self.assertEqual(completion_message("zh"), "\u914d\u7f6e\u5df2\u5b8c\u6210\u3002")
        self.assertEqual(single_field_request("source.energy", "en"), "Please provide: source.energy")

    def test_clarification_helpers_cover_both_languages(self) -> None:
        items = ["source energy", "source direction"]
        self.assertEqual(
            clarification_fallback(items, "en"),
            "Please provide: source energy, source direction",
        )
        self.assertIn("Question:", clarification_prompt(items, "en"))
        self.assertIn("\u95ee\u9898\uff1a", clarification_prompt(items, "zh"))


if __name__ == "__main__":
    unittest.main()

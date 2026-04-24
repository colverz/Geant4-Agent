from __future__ import annotations

import unittest

from core.config.prompt_profiles import (
    PromptTask,
    build_prompt,
    get_prompt_profile,
    list_prompt_profiles,
    validate_prompt_output,
)


class PromptProfilesTest(unittest.TestCase):
    def test_each_task_and_language_builds_prompt_with_contract_metadata(self) -> None:
        for task in PromptTask:
            for lang in ("zh", "en"):
                built = build_prompt(
                    task,
                    lang,
                    {
                        "user_text": "latest result?",
                        "context_summary": "none",
                        "recent_user_text": "latest result?",
                        "confirmed_items": "geometry",
                        "missing_items": "source energy",
                        "payload": {"base_message": "Events: 4 / 4"},
                    },
                )
                profile = get_prompt_profile(task, lang)

                self.assertTrue(built.prompt.strip())
                self.assertEqual(built.profile_id, profile.id)
                self.assertEqual(built.validator_name, profile.validator_name)
                self.assertEqual(built.output_contract, profile.output_contract.value)

    def test_profile_ids_are_unique(self) -> None:
        ids = [profile.id for profile in list_prompt_profiles()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_json_validator_rejects_non_json_for_extraction_tasks(self) -> None:
        result = validate_prompt_output(PromptTask.SLOT_EXTRACT, "en", "not json")

        self.assertFalse(result.ok)
        self.assertIn("not_json", result.errors)

    def test_route_validator_rejects_unknown_labels(self) -> None:
        result = validate_prompt_output(PromptTask.RESULT_QUESTION_ROUTE, "en", "please_run_shell")

        self.assertFalse(result.ok)
        self.assertIn("unknown_route_label", result.errors)

    def test_grounded_runtime_validator_rejects_new_numbers(self) -> None:
        result = validate_prompt_output(
            PromptTask.RUNTIME_RESULT_EXPLAIN,
            "en",
            "The result is 99 Gy.",
            {"base_message": "Events: 4 / 4."},
        )

        self.assertFalse(result.ok)
        self.assertIn("new_numeric_value", result.errors)

    def test_question_validator_rejects_internal_field_leak(self) -> None:
        result = validate_prompt_output(PromptTask.CLARIFICATION, "en", "Please provide source.energy.")

        self.assertFalse(result.ok)
        self.assertIn("internal_field_leak", result.errors)


if __name__ == "__main__":
    unittest.main()

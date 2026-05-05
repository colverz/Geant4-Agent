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

    def test_slot_profile_uses_strict_prompt_builder(self) -> None:
        built = build_prompt(
            PromptTask.SLOT_EXTRACT,
            "en",
            {"user_text": "Output json.", "context_summary": "phase=output"},
        )

        self.assertEqual(built.profile_id, "slot_extract_en_strict_slot_v2")
        self.assertIn("Use slots, not config paths.", built.prompt)
        self.assertIn("Do not restate unrelated slots from previous turns", built.prompt)
        self.assertIn('User: "Output json."', built.prompt)

    def test_semantic_profile_uses_strict_prompt_builder(self) -> None:
        built = build_prompt(
            PromptTask.SEMANTIC_EXTRACT,
            "en",
            {"user_text": "set source", "context_summary": "phase=source"},
        )

        self.assertEqual(built.profile_id, "semantic_extract_en_strict_semantic_v1")
        self.assertIn('"target_paths"', built.prompt)
        self.assertIn("Context: phase=source", built.prompt)

    def test_json_validator_rejects_non_json_for_extraction_tasks(self) -> None:
        result = validate_prompt_output(PromptTask.SLOT_EXTRACT, "en", "not json")

        self.assertFalse(result.ok)
        self.assertIn("not_json", result.errors)

    def test_slot_validator_rejects_fields_outside_prompt_contract(self) -> None:
        result = validate_prompt_output(
            PromptTask.SLOT_EXTRACT,
            "en",
            {
                "intent": "SET",
                "confidence": 0.9,
                "normalized_text": "run geant4",
                "target_slots": [],
                "slots": {"source": {"particle": "gamma", "api_key": "secret"}},
                "candidates": {},
                "run_now": True,
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown_json_key:run_now", result.errors)
        self.assertIn("unknown_json_key:slots.source.api_key", result.errors)

    def test_semantic_validator_rejects_fields_outside_prompt_contract(self) -> None:
        result = validate_prompt_output(
            PromptTask.SEMANTIC_EXTRACT,
            "en",
            {
                "intent": "SET",
                "target_paths": ["source.type"],
                "normalized_text": "source type: point",
                "structure_hint": "unknown",
                "confidence": 0.8,
                "updates": [{"path": "source.type", "op": "set", "value": "point", "tool": "run_beam"}],
                "shell_command": "geant4",
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown_json_key:shell_command", result.errors)
        self.assertIn("unknown_json_key:updates[0].tool", result.errors)

    def test_route_validator_rejects_unknown_labels(self) -> None:
        result = validate_prompt_output(PromptTask.RESULT_QUESTION_ROUTE, "en", "please_run_shell")

        self.assertFalse(result.ok)
        self.assertIn("unknown_route_label", result.errors)

    def test_route_validator_accepts_config_mutation(self) -> None:
        result = validate_prompt_output(PromptTask.RESULT_QUESTION_ROUTE, "en", "config_mutation")

        self.assertTrue(result.ok)

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

    def test_physics_recommend_profile_uses_allowed_list_context(self) -> None:
        built = build_prompt(
            PromptTask.PHYSICS_RECOMMEND,
            "en",
            {
                "allowed_lists_csv": "FTFP_BERT, QBBC",
                "context_summary": "particle=gamma",
                "request_text": "recommend a physics list",
            },
        )

        self.assertEqual(built.profile_id, "physics_recommend_en_v1")
        self.assertIn("FTFP_BERT, QBBC", built.prompt)
        self.assertIn("recommend a physics list", built.prompt)

    def test_physics_recommend_validator_rejects_extra_fields_and_unknown_values(self) -> None:
        result = validate_prompt_output(
            PromptTask.PHYSICS_RECOMMEND,
            "en",
            {
                "physics_list": "UNKNOWN_LIST",
                "backup_physics_list": "QBBC",
                "reasons": [],
                "covered_processes": [],
                "confidence": 0.8,
                "tool": "run_beam",
            },
            {"allowed_lists": ["FTFP_BERT", "QBBC"]},
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown_json_key:tool", result.errors)
        self.assertIn("value_not_allowed:physics_list", result.errors)

    def test_normalize_user_turn_profile_uses_controlled_prompt(self) -> None:
        built = build_prompt(
            PromptTask.NORMALIZE_USER_TURN,
            "en",
            {
                "user_text": "Set up a copper target box that is 10 by 20 by 30 millimeters.",
                "context_summary": "",
            },
        )

        self.assertEqual(built.profile_id, "normalize_user_turn_en_v1")
        self.assertIn("module_x:10 mm; module_y:20 mm; module_z:30 mm", built.prompt)
        self.assertIn("MUST NOT contain phrases like 'set ... to ...'", built.prompt)

    def test_normalize_user_turn_validator_rejects_alias_and_unknown_keys(self) -> None:
        result = validate_prompt_output(
            PromptTask.NORMALIZE_USER_TURN,
            "en",
            {
                "normalized_text": "module_size:10 mm; set geometry to copper box",
                "language_detected": "en",
                "structure_hint": "unknown_shape",
                "tool": "run_beam",
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown_json_key:tool", result.errors)
        self.assertIn("value_not_allowed:structure_hint", result.errors)
        self.assertIn("banned_normalized_key:module_size", result.errors)
        self.assertIn("narrative_set_to_phrase", result.errors)

    def test_interpret_user_turn_profile_uses_candidate_boundary_prompt(self) -> None:
        built = build_prompt(
            PromptTask.INTERPRET_USER_TURN,
            "en",
            {
                "user_text": "10 mm copper box target",
                "context_summary": "phase=geometry",
            },
        )

        self.assertEqual(built.profile_id, "interpret_user_turn_en_v1")
        self.assertIn("Do not output final config paths.", built.prompt)
        self.assertIn('"geometry_candidate"', built.prompt)
        self.assertIn("Bind geometry_candidate to the target/object being built.", built.prompt)

    def test_interpret_user_turn_validator_rejects_unknown_and_missing_keys(self) -> None:
        result = validate_prompt_output(
            PromptTask.INTERPRET_USER_TURN,
            "en",
            {
                "turn_summary": {},
                "geometry_candidate": {},
                "tool": "run_beam",
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown_json_key:tool", result.errors)
        self.assertIn("missing_json_key:source_candidate", result.errors)


if __name__ == "__main__":
    unittest.main()

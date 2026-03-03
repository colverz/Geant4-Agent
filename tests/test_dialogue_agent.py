from __future__ import annotations

import unittest

from core.dialogue.policy import decide_dialogue_action
from core.dialogue.renderer import render_dialogue_message
from core.dialogue.types import DialogueAction


class DialogueAgentTest(unittest.TestCase):
    def test_policy_prefers_finalize_when_complete(self) -> None:
        decision = decide_dialogue_action(
            user_intent="SET",
            is_complete=True,
            asked_fields=[],
            missing_fields=[],
            updated_paths=["geometry.structure"],
            answered_this_turn=[],
        )
        self.assertEqual(decision.action, DialogueAction.FINALIZE)

    def test_policy_asks_clarification_when_fields_missing(self) -> None:
        decision = decide_dialogue_action(
            user_intent="SET",
            is_complete=False,
            asked_fields=["source.energy"],
            missing_fields=["source.energy"],
            updated_paths=["geometry.structure"],
            answered_this_turn=[],
        )
        self.assertEqual(decision.action, DialogueAction.ASK_CLARIFICATION)

    def test_policy_requests_overwrite_confirmation_when_preview_exists(self) -> None:
        decision = decide_dialogue_action(
            user_intent="MODIFY",
            is_complete=False,
            asked_fields=[],
            missing_fields=["output.path"],
            updated_paths=[],
            answered_this_turn=[],
            pending_overwrite_preview=[{"path": "materials.selected_materials", "old": "G4_Cu", "new": "G4_Al"}],
        )
        self.assertEqual(decision.action, DialogueAction.CONFIRM_OVERWRITE)

    def test_policy_can_explain_choice_for_question_intent(self) -> None:
        decision = decide_dialogue_action(
            user_intent="QUESTION",
            is_complete=False,
            asked_fields=[],
            missing_fields=[],
            updated_paths=["physics.physics_list"],
            answered_this_turn=[],
            available_explanations={
                "physics": {
                    "label": "Physics",
                    "field": "physics list",
                    "source": "llm_recommender",
                    "reasons": ["Selected for gamma attenuation coverage."],
                }
            },
        )
        self.assertEqual(decision.action, DialogueAction.EXPLAIN_CHOICE)

    def test_policy_summarizes_progress_after_answering_clarification(self) -> None:
        decision = decide_dialogue_action(
            user_intent="SET",
            is_complete=False,
            asked_fields=[],
            missing_fields=["output.path"],
            updated_paths=["source.energy"],
            answered_this_turn=["source.energy"],
            last_dialogue_action="ask_clarification",
        )
        self.assertEqual(decision.action, DialogueAction.SUMMARIZE_PROGRESS)

    def test_renderer_can_emit_overwrite_confirmation(self) -> None:
        decision = decide_dialogue_action(
            user_intent="MODIFY",
            is_complete=False,
            asked_fields=[],
            missing_fields=[],
            updated_paths=[],
            answered_this_turn=[],
            pending_overwrite_preview=[
                {"field": "materials", "path": "materials.selected_materials", "old": "G4_Cu", "new": "G4_Al"}
            ],
        )
        msg = render_dialogue_message(
            decision,
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
        )
        self.assertIn("Please confirm", msg)
        self.assertIn("G4_Cu -> G4_Al", msg)

    def test_renderer_can_explain_choice(self) -> None:
        decision = decide_dialogue_action(
            user_intent="QUESTION",
            is_complete=False,
            asked_fields=[],
            missing_fields=[],
            updated_paths=["physics.physics_list"],
            answered_this_turn=[],
            available_explanations={
                "physics": {
                    "label": "Physics",
                    "field": "physics list",
                    "source": "llm_recommender",
                    "reasons": ["Selected for gamma attenuation coverage."],
                }
            },
        )
        msg = render_dialogue_message(
            decision,
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
        )
        self.assertIn("Physics:", msg)
        self.assertIn("Source: llm_recommender.", msg)
        self.assertIn("Reason:", msg)

    def test_renderer_can_emit_non_llm_status_messages(self) -> None:
        decision = decide_dialogue_action(
            user_intent="MODIFY",
            is_complete=False,
            asked_fields=[],
            missing_fields=["output.path"],
            updated_paths=["materials.selected_materials"],
            answered_this_turn=["materials.selected_materials"],
        )
        msg = render_dialogue_message(
            decision,
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
        )
        self.assertIn("Updated", msg)
        self.assertIn("Still needed:", msg)

    def test_renderer_can_emit_progress_summary(self) -> None:
        decision = decide_dialogue_action(
            user_intent="SET",
            is_complete=False,
            asked_fields=[],
            missing_fields=["output.path"],
            updated_paths=["source.energy"],
            answered_this_turn=["source.energy"],
            last_dialogue_action="ask_clarification",
        )
        msg = render_dialogue_message(
            decision,
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={
                "updated_fields": ["source energy"],
                "pending_fields": ["output path"],
                "recent_confirmed": ["source energy", "source direction"],
            },
        )
        self.assertIn("Updated this turn:", msg)
        self.assertIn("Confirmed so far:", msg)

    def test_renderer_prefers_grouped_progress_summary(self) -> None:
        decision = decide_dialogue_action(
            user_intent="SET",
            is_complete=False,
            asked_fields=[],
            missing_fields=["source.energy", "output.path"],
            updated_paths=["geometry.structure", "materials.selected_materials"],
            answered_this_turn=["geometry.structure"],
            last_dialogue_action="ask_clarification",
        )
        msg = render_dialogue_message(
            decision,
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={
                "grouped_status": {
                    "geometry": {
                        "label": "Geometry",
                        "updated_fields": ["geometry structure"],
                        "pending_fields": [],
                        "confirmed_fields": ["geometry structure"],
                    },
                    "source": {
                        "label": "Source",
                        "updated_fields": [],
                        "pending_fields": ["source energy"],
                        "confirmed_fields": [],
                    },
                }
            },
        )
        self.assertIn("Geometry: updated geometry structure.", msg)
        self.assertIn("Source: still needs source energy.", msg)


if __name__ == "__main__":
    unittest.main()

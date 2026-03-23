from __future__ import annotations

import unittest

from core.dialogue.renderer import render_dialogue_message
from core.dialogue.types import DialogueAction, DialogueDecision


class DialogueRendererTest(unittest.TestCase):
    def test_non_llm_clarification_uses_structured_question_for_graph_path(self) -> None:
        message = render_dialogue_message(
            DialogueDecision(
                action=DialogueAction.ASK_CLARIFICATION,
                asked_fields=["geometry.ask.ring.radius"],
                missing_fields=["geometry.ask.ring.radius"],
                user_intent="SET",
            ),
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={},
            raw_dialogue=[],
        )
        self.assertEqual(message, "Please provide the ring radius.")

    def test_non_llm_clarification_uses_structured_question_for_source_paths(self) -> None:
        message = render_dialogue_message(
            DialogueDecision(
                action=DialogueAction.ASK_CLARIFICATION,
                asked_fields=["source.position", "source.direction"],
                missing_fields=["source.position", "source.direction"],
                user_intent="SET",
            ),
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={},
            raw_dialogue=[],
        )
        self.assertIn("I still need two details.", message)
        self.assertIn("Please provide the source position vector as (x, y, z).", message)
        self.assertIn("Please provide the source direction vector as (dx, dy, dz).", message)

    def test_non_llm_overwrite_confirmation_uses_friendly_field_names(self) -> None:
        message = render_dialogue_message(
            DialogueDecision(
                action=DialogueAction.CONFIRM_OVERWRITE,
                overwrite_preview=[{"path": "materials.selected_materials", "old": "G4_Cu", "new": "G4_Al"}],
                user_intent="MODIFY",
            ),
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={},
            raw_dialogue=[],
        )
        self.assertIn("material", message.lower())
        self.assertIn("Reply 'confirm' to apply it", message)

    def test_overwrite_confirmation_humanizes_structure_value(self) -> None:
        message = render_dialogue_message(
            DialogueDecision(
                action=DialogueAction.CONFIRM_OVERWRITE,
                overwrite_preview=[{"path": "geometry.structure", "old": "single_box", "new": "single_tubs"}],
                user_intent="MODIFY",
            ),
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={},
            raw_dialogue=[],
        )
        self.assertIn("box", message.lower())
        self.assertIn("cylinder", message.lower())
        self.assertNotIn("single_box", message)
        self.assertNotIn("single_tubs", message)

    def test_overwrite_confirmation_hides_graph_program_payload(self) -> None:
        message = render_dialogue_message(
            DialogueDecision(
                action=DialogueAction.CONFIRM_OVERWRITE,
                overwrite_preview=[{"path": "geometry.graph_program", "old": {"root": "box"}, "new": {"root": "ring"}}],
                user_intent="MODIFY",
            ),
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={},
            raw_dialogue=[],
        )
        self.assertIn("updated geometry program", message.lower())
        self.assertNotIn("geometry.graph_program", message)
        self.assertNotIn("{'root': 'box'}", message)



    def test_finalize_prefers_current_turn_confirmed_fields_over_stale_history(self) -> None:
        message = render_dialogue_message(
            DialogueDecision(
                action=DialogueAction.FINALIZE,
                updated_paths=["geometry.structure", "materials.selected_materials"],
                answered_this_turn=["geometry.structure", "materials.selected_materials"],
                user_intent="CONFIRM",
            ),
            lang="en",
            use_llm_question=False,
            ollama_config="",
            user_temperature=1.0,
            dialogue_summary={
                "updated_fields": ["geometry type", "material"],
                "answered_fields": ["geometry type", "material"],
                "recent_confirmed": ["boolean solid A x", "boolean solid B y"],
            },
            raw_dialogue=[],
        )
        self.assertIn("Configuration complete", message)
        self.assertNotIn("boolean solid", message.lower())

if __name__ == "__main__":
    unittest.main()

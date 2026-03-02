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
        self.assertIn("Updated:", msg)
        self.assertIn("Still needed:", msg)


if __name__ == "__main__":
    unittest.main()

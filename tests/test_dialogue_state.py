from __future__ import annotations

import unittest

from core.dialogue.state import build_dialogue_summary, build_raw_dialogue
from core.dialogue.types import DialogueAction, DialogueDecision


class DialogueStateTest(unittest.TestCase):
    def test_build_raw_dialogue_trims_and_normalizes(self) -> None:
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        raw = build_raw_dialogue(history, limit=2)
        self.assertEqual(raw, [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}])

    def test_build_dialogue_summary_uses_friendly_fields(self) -> None:
        decision = DialogueDecision(
            action=DialogueAction.ASK_CLARIFICATION,
            asked_fields=["source.energy"],
            updated_paths=["materials.selected_materials"],
            missing_fields=["source.energy"],
            answered_this_turn=["materials.selected_materials"],
            user_intent="SET",
        )
        summary = build_dialogue_summary(decision, lang="en", is_complete=False)
        self.assertEqual(summary["status"], "pending")
        self.assertEqual(summary["last_action"], "ask_clarification")
        self.assertIn("source energy", summary["pending_fields"])
        self.assertIn("source energy", summary["next_questions"])


if __name__ == "__main__":
    unittest.main()

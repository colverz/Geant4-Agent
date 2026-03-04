from __future__ import annotations

import unittest

from core.config.llm_prompt_registry import (
    STRICT_SEMANTIC_PROMPT_PROFILE,
    STRICT_SLOT_PROMPT_PROFILE,
    build_strict_semantic_prompt,
    build_strict_slot_prompt,
)


class LlmPromptRegistryTest(unittest.TestCase):
    def test_slot_prompt_profile_is_versioned(self) -> None:
        self.assertEqual(STRICT_SLOT_PROMPT_PROFILE, "strict_slot_v2")
        prompt = build_strict_slot_prompt("copper box", "phase=geometry")
        self.assertIn("Use slots, not config paths.", prompt)
        self.assertIn("Context: phase=geometry", prompt)
        self.assertIn("Do not restate unrelated slots from previous turns", prompt)
        self.assertIn("csv|hdf5|root|xml|json|null", prompt)
        self.assertIn("Prefer official Geant4 analysis file types", prompt)
        self.assertIn('User: "Output json."', prompt)

    def test_semantic_prompt_profile_is_versioned(self) -> None:
        self.assertTrue(STRICT_SEMANTIC_PROMPT_PROFILE.startswith("strict_semantic_"))
        prompt = build_strict_semantic_prompt("set source", "phase=source")
        self.assertIn('"target_paths"', prompt)
        self.assertIn("Context: phase=source", prompt)


if __name__ == "__main__":
    unittest.main()

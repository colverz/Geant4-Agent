from __future__ import annotations

import unittest

from core.orchestrator.session_manager import _build_forced_explicit_candidate
from core.orchestrator.types import CandidateUpdate, Intent, Producer


class ForcedExplicitChoicesTest(unittest.TestCase):
    def test_builds_physics_and_output_updates(self) -> None:
        user_candidate = CandidateUpdate(
            producer=Producer.USER_EXPLICIT,
            intent=Intent.SET,
            target_paths=["physics.physics_list", "output.format"],
            updates=[],
            confidence=1.0,
            rationale="test",
        )
        forced = _build_forced_explicit_candidate(
            text="Use QBBC physics list and output root.",
            normalized_text="",
            user_candidate=user_candidate,
            turn_id=1,
        )
        self.assertIsNotNone(forced)
        assert forced is not None
        values = {u.path: u.value for u in forced.updates}
        self.assertEqual(values.get("physics.physics_list"), "QBBC")
        self.assertEqual(values.get("output.format"), "root")

    def test_does_not_force_material_or_source_type_updates(self) -> None:
        user_candidate = CandidateUpdate(
            producer=Producer.USER_EXPLICIT,
            intent=Intent.SET,
            target_paths=["materials.selected_materials", "source.type"],
            updates=[],
            confidence=1.0,
            rationale="test",
        )
        forced = _build_forced_explicit_candidate(
            text="Use tungsten shielding with a pencil beam and output json.",
            normalized_text="",
            user_candidate=user_candidate,
            turn_id=1,
        )
        self.assertIsNotNone(forced)
        assert forced is not None
        values = {u.path: u.value for u in forced.updates}
        self.assertEqual(values.get("output.format"), "json")
        self.assertNotIn("materials.selected_materials", values)
        self.assertNotIn("source.type", values)

    def test_does_not_force_short_material_alias_without_material_context(self) -> None:
        user_candidate = CandidateUpdate(
            producer=Producer.USER_EXPLICIT,
            intent=Intent.SET,
            target_paths=["output.format"],
            updates=[],
            confidence=1.0,
            rationale="test",
        )
        forced = _build_forced_explicit_candidate(
            text="Set output json and keep the ring radius at 120 mm.",
            normalized_text="",
            user_candidate=user_candidate,
            turn_id=1,
        )
        self.assertIsNotNone(forced)
        assert forced is not None
        values = {u.path: u.value for u in forced.updates}
        self.assertNotIn("materials.selected_materials", values)


if __name__ == "__main__":
    unittest.main()

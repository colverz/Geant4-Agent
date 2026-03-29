from __future__ import annotations

import unittest

from core.geometry.resolver import (
    build_geometry_intent_from_resolved_draft,
    resolve_geometry_from_merged,
)
from core.interpreter.merged import MergedField, MergedGeometry


class GeometryResolverTests(unittest.TestCase):
    def test_resolver_builds_box_from_shared_side_length(self) -> None:
        merged = MergedGeometry(
            kind=MergedField(value="box", chosen_from="shared", confidence=1.0),
            material=MergedField(value="G4_Cu", chosen_from="shared", confidence=1.0),
            dimensions={
                "side_length_mm": MergedField(value=10.0, chosen_from="llm", confidence=0.9),
            },
        )
        draft = resolve_geometry_from_merged(merged)
        self.assertEqual(draft.structure, "single_box")
        self.assertEqual(draft.material, "G4_Cu")
        self.assertEqual(draft.params["size_triplet_mm"], [10.0, 10.0, 10.0])
        self.assertEqual(draft.open_questions, ())
        self.assertTrue(draft.bridge_allowed)

    def test_resolver_prefers_cylinder_when_radius_and_half_length_exist(self) -> None:
        merged = MergedGeometry(
            kind=MergedField(value="box", chosen_from="llm", confidence=0.7, conflict=True),
            dimensions={
                "radius_mm": MergedField(value=5.0, chosen_from="evidence", confidence=1.0),
                "half_length_mm": MergedField(value=20.0, chosen_from="evidence", confidence=1.0),
            },
            ambiguities=["shape wording is unclear"],
        )
        draft = resolve_geometry_from_merged(merged)
        self.assertEqual(draft.structure, "single_tubs")
        self.assertEqual(draft.params["radius_mm"], 5.0)
        self.assertEqual(draft.params["half_length_mm"], 20.0)
        self.assertIn("geometry.kind", draft.conflicts)
        self.assertFalse(draft.bridge_allowed)

    def test_resolver_keeps_box_open_when_only_thickness_exists(self) -> None:
        merged = MergedGeometry(
            kind=MergedField(value="box", chosen_from="llm", confidence=0.8),
            dimensions={
                "thickness_mm": MergedField(value=2.0, chosen_from="llm", confidence=0.8),
            },
        )
        draft = resolve_geometry_from_merged(merged)
        self.assertEqual(draft.structure, "single_box")
        self.assertIn("geometry.size_triplet_mm", draft.open_questions)
        self.assertIn("box_thickness_without_face_dimensions", draft.ambiguities)
        self.assertTrue(draft.bridge_allowed)

    def test_build_geometry_intent_from_draft_keeps_resolved_params(self) -> None:
        merged = MergedGeometry(
            kind=MergedField(value="box", chosen_from="shared", confidence=1.0),
            material=MergedField(value="G4_Cu", chosen_from="shared", confidence=1.0),
            dimensions={
                "side_length_mm": MergedField(value=10.0, chosen_from="llm", confidence=0.9),
            },
        )
        draft = resolve_geometry_from_merged(merged)
        intent = build_geometry_intent_from_resolved_draft(draft)
        self.assertEqual(intent.structure, "single_box")
        self.assertEqual(intent.params["size_triplet_mm"], [10.0, 10.0, 10.0])
        self.assertEqual(intent.missing_fields, [])


if __name__ == "__main__":
    unittest.main()

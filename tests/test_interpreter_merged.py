from __future__ import annotations

import unittest

from core.interpreter import (
    GeometryCandidate,
    SourceCandidate,
    TurnSummary,
    merge_candidates,
)


class InterpreterMergedTests(unittest.TestCase):
    def test_merge_prefers_shared_or_evidence_values(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["geometry", "source"]),
            GeometryCandidate(
                kind_candidate="box",
                material_candidate="G4_Cu",
                dimension_hints={"side_length_mm": 10.0},
                confidence=0.8,
            ),
            SourceCandidate(
                source_type_candidate="point",
                particle_candidate="gamma",
                energy_candidate_mev=1.0,
                confidence=0.9,
            ),
            geometry_evidence={"kind": "box", "material": "G4_Cu", "dimensions": {"side_length_mm": 10.0}},
            source_evidence={"source_type": "point", "particle": "gamma", "energy_mev": 1.0},
        )
        self.assertEqual(merged.merged_geometry.kind.value, "box")
        self.assertEqual(merged.merged_geometry.kind.chosen_from, "shared")
        self.assertEqual(merged.merged_source.source_type.value, "point")
        self.assertEqual(merged.conflicts, [])

    def test_merge_records_conflict_when_llm_and_evidence_disagree(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["geometry"]),
            GeometryCandidate(kind_candidate="slab", confidence=0.7),
            SourceCandidate(),
            geometry_evidence={"kind": "box"},
        )
        self.assertIn("geometry.kind", merged.conflicts)
        self.assertEqual(merged.merged_geometry.kind.value, "box")
        self.assertTrue(merged.merged_geometry.kind.conflict)

    def test_merge_opens_question_when_explicit_domain_has_no_value(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["source"]),
            GeometryCandidate(),
            SourceCandidate(),
        )
        self.assertIn("source.type", merged.open_questions)

    def test_merge_prefers_richer_llm_geometry_when_evidence_kind_is_weak(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["geometry"]),
            GeometryCandidate(
                kind_candidate="box",
                dimension_hints={"size_triplet_mm": [10.0, 10.0, 10.0]},
                confidence=0.85,
            ),
            SourceCandidate(),
            geometry_evidence={"kind": "sphere", "dimensions": {}},
        )
        self.assertIn("geometry.kind", merged.conflicts)
        self.assertEqual(merged.merged_geometry.kind.value, "box")
        self.assertEqual(merged.merged_geometry.kind.chosen_from, "llm")
        self.assertTrue(merged.merged_geometry.kind.conflict)

    def test_merge_opens_direction_question_for_beam_without_direction(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["source"]),
            GeometryCandidate(),
            SourceCandidate(
                source_type_candidate="beam",
                particle_candidate="gamma",
                energy_candidate_mev=5.0,
                position_mode="absolute",
                position_hint={"position_mm": [0.0, 0.0, -250.0]},
                confidence=0.8,
            ),
            source_evidence={
                "source_type": "beam",
                "particle": "gamma",
                "energy_mev": 5.0,
                "position": {"position_mm": [0.0, 0.0, -250.0]},
            },
        )
        self.assertIn("source.direction", merged.open_questions)

    def test_merge_treats_equivalent_source_position_as_shared(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["source"]),
            GeometryCandidate(),
            SourceCandidate(
                source_type_candidate="point",
                position_mode="absolute",
                position_hint={"position_mm": [0.0, 0.0, -20.0], "offset_mm": None, "axis": None},
                confidence=0.9,
            ),
            source_evidence={"position": {"position_mm": [0.0, 0.0, -20.0]}},
        )
        self.assertEqual(merged.merged_source.position.chosen_from, "shared")
        self.assertFalse(merged.merged_source.position.conflict)
        self.assertNotIn("source.position", merged.conflicts)

    def test_merge_treats_equivalent_source_direction_as_shared(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["source"]),
            GeometryCandidate(),
            SourceCandidate(
                source_type_candidate="point",
                direction_mode="explicit_vector",
                direction_hint={"direction_vec": [0.0, 0.0, 1.0], "axis": "+z"},
                confidence=0.9,
            ),
            source_evidence={"direction": {"mode": "explicit_vector", "hint": {"direction_vec": [0.0, 0.0, 1.0]}}},
        )
        self.assertEqual(merged.merged_source.direction.chosen_from, "shared")
        self.assertFalse(merged.merged_source.direction.conflict)
        self.assertNotIn("source.direction", merged.conflicts)

    def test_merge_treats_side_length_and_uniform_triplet_as_equivalent(self) -> None:
        merged = merge_candidates(
            TurnSummary(intent="set", explicit_domains=["geometry"]),
            GeometryCandidate(
                kind_candidate="box",
                dimension_hints={"side_length_mm": 10.0},
                confidence=0.9,
            ),
            SourceCandidate(),
            geometry_evidence={"kind": "box", "dimensions": {"side_length_mm": 10.0}},
        )
        self.assertEqual(merged.merged_geometry.dimensions["side_length_mm"].chosen_from, "shared")
        self.assertNotIn("geometry.side_length_mm", merged.conflicts)


if __name__ == "__main__":
    unittest.main()

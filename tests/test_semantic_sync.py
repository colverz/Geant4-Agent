from __future__ import annotations

import unittest

from core.orchestrator.semantic_sync import build_semantic_sync_candidate
from core.orchestrator.types import Producer, UpdateOp


class SemanticSyncTest(unittest.TestCase):
    def test_builds_semantic_sync_candidate(self) -> None:
        config = {
            "geometry": {"structure": "single_box", "root_name": None},
            "materials": {
                "selected_materials": ["G4_Cu"],
                "volume_material_map": {},
                "selection_source": None,
                "selection_reasons": [],
            },
            "source": {
                "type": None,
                "particle": "gamma",
                "energy": 1.0,
                "position": [0.0, 0.0, -100.0],
                "direction": [0.0, 0.0, 1.0],
                "selection_source": None,
                "selection_reasons": [],
            },
            "physics": {},
            "output": {"format": "root", "path": None},
        }
        candidate = build_semantic_sync_candidate(config, turn_id=9)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.rationale, "semantic_sync")
        mapped = {u.path: u.value for u in candidate.updates}
        self.assertEqual(mapped["geometry.root_name"], "box")
        self.assertEqual(mapped["materials.volume_material_map"], {"box": "G4_Cu"})
        self.assertEqual(mapped["output.path"], "output/result.root")
        self.assertEqual(mapped["source.type"], "point")

    def test_sync_geometry_overrides_single_solid_when_graph_root_is_boolean(self) -> None:
        config = {
            "geometry": {
                "structure": "single_box",
                "graph_program": {"root": "boolean", "nodes": [], "constraints": []},
                "chosen_skeleton": "boolean_union_boxes",
                "root_name": "box",
            },
            "materials": {},
            "source": {},
            "physics": {},
            "output": {},
        }
        candidate = build_semantic_sync_candidate(config, turn_id=10)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        mapped = {u.path: u.value for u in candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "boolean")
        self.assertEqual(mapped["geometry.root_name"], "boolean")

    def test_sync_preserves_explicit_root_name_and_material_binding(self) -> None:
        config = {
            "geometry": {"structure": "single_box", "root_name": "LeadShield"},
            "materials": {
                "selected_materials": ["G4_Pb"],
                "volume_material_map": {},
                "selection_source": None,
                "selection_reasons": [],
            },
            "source": {},
            "physics": {},
            "output": {},
        }
        candidate = build_semantic_sync_candidate(
            config,
            turn_id=11,
            recent_updates=[
                UpdateOp(
                    path="geometry.root_name",
                    op="set",
                    value="LeadShield",
                    producer=Producer.LLM_SEMANTIC_FRAME,
                    confidence=0.9,
                    turn_id=11,
                )
            ],
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        mapped = {u.path: u.value for u in candidate.updates}
        self.assertNotIn("geometry.root_name", mapped)
        self.assertEqual(mapped["materials.volume_material_map"], {"LeadShield": "G4_Pb"})


if __name__ == "__main__":
    unittest.main()

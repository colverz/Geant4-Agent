from __future__ import annotations

import unittest
from unittest.mock import patch

from core.orchestrator.semantic_sync import build_semantic_sync_candidate
from core.semantic_frame import SemanticFrame
from builder.geometry.synthesize import synthesize_from_params
from nlu.bert.extractor import extract_candidates_from_normalized_text
from nlu.runtime_components.graph_search import search_candidate_graphs


class GeometryGraphFlowTest(unittest.TestCase):
    def test_extractor_uses_raw_text_for_graph_geometry_when_normalized_text_is_under_specified(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            "geometry kind = box; geometry size triplet mm = [6,6,2]; materials primary = G4_Si;",
            raw_text="Create a complete ring setup: 16 modules, each 6 mm x 6 mm x 2 mm, radius 32 mm, clearance 0.5 mm, material G4_Si.",
            turn_id=1,
            min_confidence=0.6,
            context_summary="",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped.get("geometry.structure"), "ring")
        self.assertEqual(mapped.get("geometry.chosen_skeleton"), "ring_modules")
        self.assertEqual(mapped.get("geometry.graph_program", {}).get("root"), "ring")
        self.assertEqual(debug.get("graph_choice", {}).get("structure"), "ring")

    def test_extractor_uses_normalized_stack_params_without_losing_raw_graph_intent(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            (
                "geometry_intent: z_layer_sequence; structure: stack; "
                "stack_x: 20 mm; stack_y: 20 mm; t1: 2 mm; t2: 4 mm; t3: 6 mm; "
                "stack_clearance: 0.5 mm; parent_x: 30 mm; parent_y: 30 mm; parent_z: 20 mm; nest_clearance: 1 mm"
            ),
            raw_text=(
                "Use stacked layers with footprint 20 mm x 20 mm, thicknesses 2 mm, 4 mm, 6 mm, "
                "layer clearance 0.5 mm, outer box 30 mm x 30 mm x 20 mm, container clearance 1 mm."
            ),
            turn_id=2,
            min_confidence=0.6,
            context_summary="phase=geometry; structure=stack",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped.get("geometry.structure"), "stack")
        self.assertEqual(mapped.get("geometry.chosen_skeleton"), "stack_in_box")
        self.assertEqual(mapped.get("geometry.graph_program", {}).get("root"), "stack")
        self.assertEqual(mapped.get("geometry.params.stack_x"), 20.0)
        self.assertEqual(mapped.get("geometry.params.parent_x"), 30.0)
        self.assertEqual(debug.get("graph_choice", {}).get("structure"), "stack")

    def test_extractor_recovers_graph_program_when_normalized_hint_conflicts_with_stack(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            (
                "geometry.kind=polycone; geometry.z_planes_mm=[-10,0,10]; geometry.radii_mm=[15,15,15]; "
                "materials.primary=G4_Al; source.kind=beam; source.particle=gamma; source.energy_mev=1.25; "
                "source.position_mm=[0,0,-120]; source.direction_vec=[0,0,1]; physics.explicit_list=Shielding; output.format=csv"
            ),
            raw_text=(
                "Use stacked layers with footprint 20 mm x 20 mm, thicknesses 2 mm, 4 mm, 6 mm, "
                "layer clearance 0.5 mm, outer box 30 mm x 30 mm x 20 mm, container clearance 1 mm."
            ),
            turn_id=2,
            min_confidence=0.6,
            context_summary="phase=geometry; structure=stack",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped.get("geometry.structure"), "stack")
        self.assertEqual(mapped.get("geometry.chosen_skeleton"), "stack_in_box")
        self.assertEqual(mapped.get("geometry.graph_program", {}).get("root"), "stack")
        self.assertEqual(mapped.get("geometry.params.t1"), 2.0)
        self.assertEqual(mapped.get("geometry.params.t2"), 4.0)
        self.assertEqual(mapped.get("geometry.params.t3"), 6.0)
        self.assertEqual(debug.get("graph_choice", {}).get("missing_params"), [])

    def test_extractor_recovers_graph_program_for_nest_when_normalized_hint_conflicts(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            "geometry.kind=box; geometry.size_triplet_mm=[80,80,80]; materials.primary=G4_Pb; output.format=csv",
            raw_text="Outer box 80 mm x 80 mm x 80 mm, inner lead cylinder radius 15 mm, half length 25 mm, clearance 1 mm.",
            turn_id=1,
            min_confidence=0.6,
            context_summary="phase=geometry; structure=nest",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped.get("geometry.structure"), "nest")
        self.assertEqual(mapped.get("geometry.chosen_skeleton"), "nest_box_tubs")
        self.assertEqual(mapped.get("geometry.graph_program", {}).get("root"), "nest")
        self.assertNotIn("geometry.graph_program", debug.get("graph_choice", {}).get("dialogue_missing_paths", []))
        self.assertEqual(mapped.get("geometry.params.parent_x"), 80.0)
        self.assertEqual(mapped.get("geometry.params.parent_y"), 80.0)
        self.assertEqual(mapped.get("geometry.params.parent_z"), 80.0)

    def test_extractor_does_not_commit_sampled_defaults_for_incomplete_ring(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            "geometry.kind=sphere; geometry.radius_mm=30; output.format=json",
            raw_text="Ring of 12 modules, radius 40 mm, clearance 1 mm.",
            turn_id=1,
            min_confidence=0.6,
            context_summary="phase=geometry; structure=ring",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped.get("geometry.structure"), "ring")
        self.assertEqual(mapped.get("geometry.chosen_skeleton"), "ring_modules")
        self.assertEqual(mapped.get("geometry.graph_program", {}).get("root"), "ring")
        self.assertEqual(mapped.get("geometry.params.n"), 12)
        self.assertEqual(mapped.get("geometry.params.radius"), 40.0)
        self.assertEqual(mapped.get("geometry.params.clearance"), 1.0)
        self.assertNotIn("geometry.params.module_x", mapped)
        self.assertNotIn("geometry.params.module_y", mapped)
        self.assertNotIn("geometry.params.module_z", mapped)
        self.assertEqual(debug.get("graph_choice", {}).get("missing_params"), ["module_x", "module_y", "module_z"])

    def test_extractor_keeps_shell_graph_without_sampled_nested_defaults(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            "geometry.kind=box; geometry.size_triplet_mm=[100,100,100]; materials.primary=G4_Pb; output.format=json",
            raw_text="Create a concentric shell with inner radius 15 mm, thicknesses 5 mm, 8 mm, half length 40 mm.",
            turn_id=1,
            min_confidence=0.6,
            context_summary="phase=geometry; structure=shell",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped.get("geometry.structure"), "shell")
        self.assertEqual(mapped.get("geometry.chosen_skeleton"), "shell_nested")
        self.assertEqual(mapped.get("geometry.graph_program", {}).get("root"), "shell")
        self.assertEqual(mapped.get("geometry.params.inner_r"), 15.0)
        self.assertEqual(mapped.get("geometry.params.th1"), 5.0)
        self.assertEqual(mapped.get("geometry.params.th2"), 8.0)
        self.assertEqual(mapped.get("geometry.params.hz"), 40.0)
        self.assertNotIn("geometry.params.th3", mapped)
        self.assertNotIn("geometry.params.child_rmax", mapped)
        self.assertNotIn("geometry.params.child_hz", mapped)
        self.assertNotIn("geometry.params.clearance", mapped)
        self.assertEqual(debug.get("graph_choice", {}).get("dialogue_missing_paths"), [])

    def test_extractor_preserves_graph_metadata(self) -> None:
        frame = SemanticFrame()
        frame.geometry.structure = "ring"
        frame.geometry.chosen_skeleton = "ring_modules"
        frame.geometry.graph_program = {
            "nodes": [
                {"id": "module", "type": "Box", "x": 8.0, "y": 10.0, "z": 2.0},
                {
                    "id": "ring",
                    "type": "Ring",
                    "module": "module",
                    "n": 12,
                    "radius": 40.0,
                    "clearance": 1.0,
                },
            ],
            "root": "ring",
            "constraints": [],
        }
        frame.geometry.params = {"module_x": 8.0, "module_y": 10.0, "module_z": 2.0, "n": 12.0, "radius": 40.0}

        with patch("nlu.bert.extractor.extract_runtime_semantic_frame", return_value=(frame, {"scores": {"best_prob": 0.91}})):
            candidate, _ = extract_candidates_from_normalized_text(
                "intent=SET; structure=ring; n=12; radius=40 mm;",
                raw_text="ring of 12 modules",
                turn_id=3,
                min_confidence=0.6,
                context_summary="",
                config_path="dummy.json",
            )

        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "ring")
        self.assertEqual(mapped["geometry.chosen_skeleton"], "ring_modules")
        self.assertEqual(mapped["geometry.graph_program"]["root"], "ring")

    def test_shell_synthesis_accepts_two_explicit_thicknesses(self) -> None:
        result = synthesize_from_params(
            "shell",
            {
                "inner_r": 15.0,
                "th1": 5.0,
                "th2": 8.0,
                "hz": 40.0,
                "child_rmax": 10.0,
                "child_hz": 30.0,
                "clearance": 1.0,
            },
            seed=7,
            apply_autofix=True,
        )
        self.assertEqual(result["skeleton"], "shell_nested")
        self.assertEqual(result["missing_params"], [])
        self.assertTrue(result["feasible"])

    def test_semantic_sync_anchors_root_name_to_graph_root(self) -> None:
        config = {
            "geometry": {
                "structure": "ring",
                "graph_program": {
                    "nodes": [
                        {"id": "module", "type": "Box", "x": 8.0, "y": 10.0, "z": 2.0},
                        {
                            "id": "ring",
                            "type": "Ring",
                            "module": "module",
                            "n": 12,
                            "radius": 40.0,
                            "clearance": 1.0,
                        },
                    ],
                    "root": "ring",
                    "constraints": [],
                },
                "params": {},
                "root_name": None,
                "feasible": None,
            },
            "materials": {
                "selected_materials": ["G4_Cu"],
                "volume_material_map": {"target": "G4_Cu"},
                "selection_source": None,
                "selection_reasons": [],
            },
            "source": {
                "type": None,
                "particle": None,
                "energy": None,
                "position": None,
                "direction": None,
                "selection_source": None,
                "selection_reasons": [],
            },
            "physics": {
                "physics_list": None,
                "backup_physics_list": None,
                "selection_source": None,
                "selection_reasons": [],
                "covered_processes": [],
            },
            "output": {"format": None, "path": None},
        }
        candidate = build_semantic_sync_candidate(config, turn_id=7, recent_updates=[])
        self.assertIsNotNone(candidate)
        mapped = {update.path: update.value for update in candidate.updates}
        self.assertEqual(mapped["geometry.root_name"], "ring")
        self.assertEqual(mapped["materials.volume_material_map"], {"ring": "G4_Cu"})

    def test_graph_search_prefers_ring_for_module_triplet_plus_radius_signature(self) -> None:
        result = search_candidate_graphs(
            "PET detector ring with module 6 mm x 6 mm x 15 mm and radius 90 mm",
            {"module_x": 6.0, "module_y": 6.0, "module_z": 15.0, "radius": 90.0},
            min_confidence=0.6,
            seed=7,
            top_k=3,
            apply_autofix=True,
        )
        self.assertEqual(result.structure, "ring")

    def test_graph_search_prefers_boolean_for_dual_box_signature(self) -> None:
        result = search_candidate_graphs(
            "set geometry to box with size 60 mm x 60 mm x 30 mm; set geometry to box with size 40 mm x 80 mm x 30 mm",
            {
                "bool_a_x": 60.0,
                "bool_a_y": 60.0,
                "bool_a_z": 30.0,
                "bool_b_x": 40.0,
                "bool_b_y": 80.0,
                "bool_b_z": 30.0,
            },
            min_confidence=0.6,
            seed=7,
            top_k=3,
            apply_autofix=True,
        )
        self.assertEqual(result.structure, "boolean")

    def test_graph_search_does_not_treat_negative_coordinate_minus_as_boolean(self) -> None:
        result = search_candidate_graphs(
            "geometry.kind:box; geometry.size_triplet_mm:[10,20,30]; source.position_mm:[0,0,-20]; source.direction_vec:[0,0,1]",
            {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
            min_confidence=0.6,
            seed=7,
            top_k=3,
            apply_autofix=True,
        )
        self.assertNotEqual(result.structure, "boolean")

    def test_runtime_semantic_does_not_fallback_to_boolean_for_minus_coordinate(self) -> None:
        candidate, debug = extract_candidates_from_normalized_text(
            "geometry.kind:box; geometry.size_triplet_mm:[10,20,30]; materials.primary:G4_Cu; source.position_mm:[0,0,-20]; source.direction_vec:[0,0,1]",
            raw_text="Put a 1 MeV gamma point source at z equals minus 20 mm near a 10 by 20 by 30 millimeters copper target box.",
            turn_id=1,
            min_confidence=0.6,
            context_summary="",
            config_path="dummy.json",
            apply_autofix=True,
        )
        mapped = {update.path: update.value for update in candidate.updates}

        self.assertNotEqual(mapped.get("geometry.structure"), "boolean")
        self.assertNotEqual(debug.get("graph_choice", {}).get("structure"), "boolean")

    def test_graph_search_still_treats_box_minus_box_as_boolean(self) -> None:
        result = search_candidate_graphs(
            "10 by 20 by 30 mm box minus 5 by 6 by 7 mm box",
            {
                "bool_a_x": 10.0,
                "bool_a_y": 20.0,
                "bool_a_z": 30.0,
                "bool_b_x": 5.0,
                "bool_b_y": 6.0,
                "bool_b_z": 7.0,
            },
            min_confidence=0.6,
            seed=7,
            top_k=3,
            apply_autofix=True,
        )
        self.assertEqual(result.structure, "boolean")

    def test_graph_search_prefers_nest_when_parent_and_child_signatures_exist(self) -> None:
        result = search_candidate_graphs(
            "outer box 80 mm x 80 mm x 80 mm, inner lead cylinder radius 15 mm, half length 25 mm, clearance 1 mm",
            {
                "parent_x": 80.0,
                "parent_y": 80.0,
                "parent_z": 80.0,
                "child_rmax": 15.0,
                "child_hz": 25.0,
                "clearance": 1.0,
                "module_x": 80.0,
                "module_y": 80.0,
                "module_z": 80.0,
                "radius": 15.0,
            },
            min_confidence=0.6,
            seed=7,
            top_k=3,
            apply_autofix=True,
        )
        self.assertEqual(result.structure, "nest")

    def test_graph_search_prefers_grid_when_pitch_and_counts_exist(self) -> None:
        result = search_candidate_graphs(
            "4 x 4 layout, 6 mm x 6 mm x 1.5 mm modules, pitch_x 8 mm, pitch_y 8 mm, clearance 0.5 mm",
            {
                "nx": 4,
                "ny": 4,
                "module_x": 6.0,
                "module_y": 6.0,
                "module_z": 1.5,
                "pitch_x": 8.0,
                "pitch_y": 8.0,
                "clearance": 0.5,
            },
            min_confidence=0.6,
            seed=7,
            top_k=3,
            apply_autofix=True,
        )
        self.assertEqual(result.structure, "grid")


if __name__ == "__main__":
    unittest.main()

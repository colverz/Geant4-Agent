from __future__ import annotations

import unittest

from nlu.runtime_components.postprocess import merge_params


class RuntimePostprocessTest(unittest.TestCase):
    def test_merge_params_backfills_stack_fields_from_natural_language(self) -> None:
        params, notes = merge_params(
            "stack of layers along z, x 20 mm, y 20 mm, thicknesses 2 3 4 mm, clearance 0.5 mm",
            {},
        )
        self.assertEqual(params.get("stack_x"), 20.0)
        self.assertEqual(params.get("stack_y"), 20.0)
        self.assertEqual(params.get("t1"), 2.0)
        self.assertEqual(params.get("t2"), 3.0)
        self.assertEqual(params.get("t3"), 4.0)
        self.assertEqual(params.get("stack_clearance"), 0.5)
        self.assertEqual(params.get("nest_clearance"), 0.5)
        self.assertTrue(any("stack thicknesses" in note for note in notes))

    def test_merge_params_backfills_stack_footprint_and_outer_box(self) -> None:
        params, notes = merge_params(
            "Use stack footprint 25 mm x 25 mm, thicknesses 1 mm, 2 mm, 1 mm, "
            "layer clearance 0.2 mm, outer box 30 mm x 30 mm x 12 mm, container clearance 0.5 mm",
            {},
        )
        self.assertEqual(params.get("stack_x"), 25.0)
        self.assertEqual(params.get("stack_y"), 25.0)
        self.assertEqual(params.get("parent_x"), 30.0)
        self.assertEqual(params.get("parent_y"), 30.0)
        self.assertEqual(params.get("parent_z"), 12.0)
        self.assertEqual(params.get("nest_clearance"), 0.5)
        self.assertTrue(any("stack footprint" in note for note in notes))
        self.assertTrue(any("outer box" in note for note in notes))

    def test_merge_params_backfills_stack_thicknesses_when_each_value_has_unit(self) -> None:
        params, _ = merge_params(
            "stack footprint 20 mm x 20 mm, thicknesses 2 mm, 4 mm, 6 mm, outer box 30 mm x 30 mm x 20 mm",
            {},
        )
        self.assertEqual(params.get("t1"), 2.0)
        self.assertEqual(params.get("t2"), 4.0)
        self.assertEqual(params.get("t3"), 6.0)

    def test_merge_params_backfills_boolean_box_triplets_from_union_phrase(self) -> None:
        params, notes = merge_params(
            "union of two boxes, 10x20x30 mm and 5x6x7 mm",
            {},
        )
        self.assertEqual(params.get("bool_a_x"), 10.0)
        self.assertEqual(params.get("bool_a_y"), 20.0)
        self.assertEqual(params.get("bool_a_z"), 30.0)
        self.assertEqual(params.get("bool_b_x"), 5.0)
        self.assertEqual(params.get("bool_b_y"), 6.0)
        self.assertEqual(params.get("bool_b_z"), 7.0)
        self.assertNotIn("module_x", params)
        self.assertTrue(any("bool_a_*" in note for note in notes))

    def test_merge_params_backfills_boolean_box_triplets_from_by_phrase(self) -> None:
        params, _ = merge_params(
            "subtract a 5 by 6 by 7 mm box from a 10 by 20 by 30 mm box",
            {},
        )
        self.assertEqual(params.get("bool_a_x"), 10.0)
        self.assertEqual(params.get("bool_a_y"), 20.0)
        self.assertEqual(params.get("bool_a_z"), 30.0)
        self.assertEqual(params.get("bool_b_x"), 5.0)
        self.assertEqual(params.get("bool_b_y"), 6.0)
        self.assertEqual(params.get("bool_b_z"), 7.0)

    def test_merge_params_backfills_boolean_box_triplets_from_hole_phrase(self) -> None:
        params, _ = merge_params(
            "10x20x30 mm box with a 5x6x7 mm hole",
            {},
        )
        self.assertEqual(params.get("bool_a_x"), 10.0)
        self.assertEqual(params.get("bool_a_y"), 20.0)
        self.assertEqual(params.get("bool_a_z"), 30.0)
        self.assertEqual(params.get("bool_b_x"), 5.0)
        self.assertEqual(params.get("bool_b_y"), 6.0)
        self.assertEqual(params.get("bool_b_z"), 7.0)

    def test_merge_params_backfills_grid_counts_from_array_phrase(self) -> None:
        params, _ = merge_params(
            "3 x 3 module array, module 12 mm x 12 mm x 3 mm, pitch_x 15 mm, pitch_y 15 mm, clearance 1 mm",
            {},
        )
        self.assertEqual(params.get("nx"), 3)
        self.assertEqual(params.get("ny"), 3)
        self.assertEqual(params.get("module_x"), 12.0)
        self.assertEqual(params.get("pitch_x"), 15.0)
        self.assertEqual(params.get("clearance"), 1.0)

    def test_merge_params_promotes_outer_box_into_parent_for_nest_context(self) -> None:
        params, _ = merge_params(
            "nested setup: outer box 80 mm x 80 mm x 80 mm, inner cylinder radius 15 mm, half length 25 mm, clearance 1 mm",
            {},
        )
        self.assertEqual(params.get("parent_x"), 80.0)
        self.assertEqual(params.get("parent_y"), 80.0)
        self.assertEqual(params.get("parent_z"), 80.0)
        self.assertEqual(params.get("child_rmax"), 15.0)
        self.assertEqual(params.get("child_hz"), 25.0)

    def test_merge_params_backfills_grid_counts_from_modules_phrase_and_shared_pitch(self) -> None:
        params, _ = merge_params(
            "6 by 5 modules, each module 8 mm x 8 mm x 2 mm, pitch 10 mm by 10 mm, clearance 0.5 mm, grid layout",
            {},
        )
        self.assertEqual(params.get("nx"), 6)
        self.assertEqual(params.get("ny"), 5)
        self.assertEqual(params.get("pitch_x"), 10.0)
        self.assertEqual(params.get("pitch_y"), 10.0)

    def test_merge_params_backfills_grid_counts_from_loose_pair_when_pitch_present(self) -> None:
        params, _ = merge_params(
            "4 x 4 layout, 6 mm x 6 mm x 1.5 mm modules, pitch_x 8 mm, pitch_y 8 mm, clearance 0.5 mm",
            {},
        )
        self.assertEqual(params.get("nx"), 4)
        self.assertEqual(params.get("ny"), 4)
        self.assertEqual(params.get("pitch_x"), 8.0)
        self.assertEqual(params.get("pitch_y"), 8.0)

    def test_merge_params_backfills_grid_pitch_from_and_phrase(self) -> None:
        params, _ = merge_params(
            "grid layout, 10 by 3 modules, each 5 mm x 20 mm x 0.5 mm, pitch 6 mm and 22 mm, clearance 0.2 mm",
            {},
        )
        self.assertEqual(params.get("pitch_x"), 6.0)
        self.assertEqual(params.get("pitch_y"), 22.0)

    def test_merge_params_supports_chinese_multiplication_sign_and_axis_pitch(self) -> None:
        params, _ = merge_params(
            "设置为4×4网格，每个模块6毫米×6毫米×1.5毫米，x方向间距8毫米，y方向间距8毫米",
            {},
        )
        self.assertEqual(params.get("nx"), 4)
        self.assertEqual(params.get("ny"), 4)
        self.assertEqual(params.get("module_x"), 6.0)
        self.assertEqual(params.get("module_y"), 6.0)
        self.assertEqual(params.get("module_z"), 1.5)
        self.assertEqual(params.get("pitch_x"), 8.0)
        self.assertEqual(params.get("pitch_y"), 8.0)

    def test_merge_params_supports_boolean_triplets_with_chinese_multiplication_sign(self) -> None:
        params, _ = merge_params(
            "使用一个120毫米×80毫米×50毫米的大盒子，减去一个30毫米×20毫米×10毫米的盒子",
            {},
        )
        self.assertEqual(params.get("bool_a_x"), 120.0)
        self.assertEqual(params.get("bool_a_y"), 80.0)
        self.assertEqual(params.get("bool_a_z"), 50.0)
        self.assertEqual(params.get("bool_b_x"), 30.0)
        self.assertEqual(params.get("bool_b_y"), 20.0)
        self.assertEqual(params.get("bool_b_z"), 10.0)

    def test_merge_params_backfills_nested_parent_and_child_box_triplets(self) -> None:
        params, _ = merge_params(
            "nested layout: parent box 300 mm x 300 mm x 300 mm, child box 100 mm x 100 mm x 100 mm, clearance 5 mm",
            {},
        )
        self.assertEqual(params.get("parent_x"), 300.0)
        self.assertEqual(params.get("parent_y"), 300.0)
        self.assertEqual(params.get("parent_z"), 300.0)
        self.assertEqual(params.get("child_x"), 100.0)
        self.assertEqual(params.get("child_y"), 100.0)
        self.assertEqual(params.get("child_z"), 100.0)
        self.assertEqual(params.get("clearance"), 5.0)

    def test_merge_params_promotes_dual_box_prompt_to_boolean_triplets(self) -> None:
        params, notes = merge_params(
            "set geometry to box with size 60 mm x 60 mm x 30 mm; set geometry to box with size 40 mm x 80 mm x 30 mm",
            {},
        )
        self.assertEqual(params.get("bool_a_x"), 60.0)
        self.assertEqual(params.get("bool_a_y"), 60.0)
        self.assertEqual(params.get("bool_a_z"), 30.0)
        self.assertEqual(params.get("bool_b_x"), 40.0)
        self.assertEqual(params.get("bool_b_y"), 80.0)
        self.assertEqual(params.get("bool_b_z"), 30.0)
        self.assertTrue(any("dual-box heuristic" in note for note in notes))


if __name__ == "__main__":
    unittest.main()

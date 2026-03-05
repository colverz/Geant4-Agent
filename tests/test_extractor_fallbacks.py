from __future__ import annotations

import unittest

from nlu.bert.extractor import (
    _infer_material,
    _infer_source_type,
    _infer_structure_from_text,
    _parse_named_length_mm,
    _parse_vector,
    _parse_direction_shorthand,
    _parse_position_shorthand,
)


class ExtractorFallbacksTest(unittest.TestCase):
    def test_infer_structure_supports_chinese_terms(self) -> None:
        self.assertEqual(_infer_structure_from_text("\u7acb\u65b9\u4f53\u9756\u6001\u6d4b\u8bd5"), "single_box")
        self.assertEqual(_infer_structure_from_text("\u5706\u67f1\u68c0\u6d4b\u5668"), "single_tubs")
        self.assertEqual(_infer_structure_from_text("\u7403\u5f62\u76ee\u6807"), "single_sphere")

    def test_infer_material_supports_chinese_terms(self) -> None:
        self.assertEqual(_infer_material("\u94dc\u9776"), "G4_Cu")
        self.assertEqual(_infer_material("\u7845\u63a2\u6d4b\u5668"), "G4_Si")
        self.assertEqual(_infer_material("\u94dd\u677f"), "G4_Al")
        self.assertEqual(_infer_material("\u6c34\u7bb1"), "G4_WATER")

    def test_infer_source_type_supports_chinese_terms(self) -> None:
        self.assertEqual(_infer_source_type("\u70b9\u6e90"), "point")
        self.assertEqual(_infer_source_type("\u675f\u6d41"), "beam")
        self.assertEqual(_infer_source_type("\u5404\u5411\u540c\u6027\u6e90"), "isotropic")
        self.assertEqual(_infer_source_type("\u9762\u6e90"), "plane")

    def test_direction_and_position_shorthand(self) -> None:
        self.assertEqual(_parse_direction_shorthand("\u6cbf +z \u65b9\u5411"), {"type": "vector", "value": [0.0, 0.0, 1.0]})
        self.assertEqual(_parse_position_shorthand("\u6e90\u653e\u5728\u4e2d\u5fc3"), {"type": "vector", "value": [0.0, 0.0, 0.0]})

    def test_parse_vector_supports_chinese_keys(self) -> None:
        self.assertEqual(
            _parse_vector("\u4f4d\u7f6e(0, 0, -100)", "position"),
            {"type": "vector", "value": [0.0, 0.0, -100.0]},
        )
        self.assertEqual(
            _parse_vector("\u65b9\u5411: (0,0,1)", "direction"),
            {"type": "vector", "value": [0.0, 0.0, 1.0]},
        )

    def test_parse_named_lengths_for_tubs(self) -> None:
        self.assertEqual(
            _parse_named_length_mm("\u534a\u5f84 30 mm", ["\u534a\u5f84"]),
            30.0,
        )
        self.assertEqual(
            _parse_named_length_mm("\u534a\u957f 5 cm", ["\u534a\u957f"]),
            50.0,
        )


if __name__ == "__main__":
    unittest.main()

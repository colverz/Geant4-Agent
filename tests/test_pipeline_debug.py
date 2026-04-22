from __future__ import annotations

import unittest

from core.orchestrator.pipeline_debug import (
    V2PipelineDebugView,
    compile_v2_missing_paths,
    merge_v2_meta,
    merge_v2_missing_paths,
    prioritize_spatial_questions,
    prioritize_v2_compile_questions,
)


class PipelineDebugTest(unittest.TestCase):
    def test_v2_debug_view_handles_malformed_meta_without_failing(self) -> None:
        slot_debug = {
            "geometry_v2": "not-a-dict",
            "source_v2": {"missing_fields": ["particle", "energy_mev"]},
            "spatial_v2": {"source_meta": "not-a-dict"},
        }

        view = V2PipelineDebugView.from_slot_debug(slot_debug)

        self.assertEqual(view.geometry.missing_fields, ())
        self.assertEqual(view.source.missing_fields, ("particle", "energy_mev"))
        self.assertEqual(view.spatial_source.missing_fields, ())
        self.assertEqual(compile_v2_missing_paths(slot_debug), ["source.particle", "source.energy"])

    def test_v2_debug_view_owns_compile_and_spatial_question_priority(self) -> None:
        slot_debug = {
            "geometry_v2": {"missing_fields": ["radius_mm"]},
            "source_v2": {"missing_fields": ["particle"]},
            "spatial_v2": {
                "warnings": ["source_on_target_face"],
                "source_meta": {"missing_fields": ["position_mm", "source_type"]},
            },
        }

        missing_paths = merge_v2_missing_paths(["output.format"], slot_debug)
        self.assertEqual(
            missing_paths,
            [
                "output.format",
                "geometry.params.child_rmax",
                "source.particle",
                "source.position",
                "source.type",
            ],
        )
        self.assertEqual(
            prioritize_v2_compile_questions(
                ["geometry.params.child_rmax", "source.particle", "source.position"],
                missing_paths,
                slot_debug,
            ),
            ["source.position", "source.type", "source.particle"],
        )
        self.assertEqual(
            prioritize_spatial_questions(
                ["source.particle", "source.position"],
                missing_paths,
                slot_debug,
            ),
            ["source.position", "source.particle"],
        )

    def test_merge_v2_meta_prefers_ready_meta_and_dedupes_incomplete_meta(self) -> None:
        self.assertEqual(
            merge_v2_meta(
                {"compile_ok": False, "missing_fields": ["particle"], "errors": ["a"]},
                {"compile_ok": False, "missing_fields": ["particle", "energy_mev"], "errors": ["b"]},
            ),
            {
                "compile_ok": False,
                "missing_fields": ["particle", "energy_mev"],
                "errors": ["a", "b"],
                "warnings": [],
                "runtime_ready": False,
                "finalization_status": "missing",
            },
        )
        self.assertEqual(
            merge_v2_meta(
                {"compile_ok": False, "missing_fields": ["particle"]},
                {"compile_ok": True, "source_type": "point", "runtime_ready": True},
            ),
            {"compile_ok": True, "source_type": "point", "runtime_ready": True},
        )


if __name__ == "__main__":
    unittest.main()

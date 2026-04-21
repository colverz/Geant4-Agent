from __future__ import annotations

import unittest

from core.orchestrator.session_manager import (
    _apply_updates,
    _merge_v2_missing_paths,
    _prioritize_spatial_questions,
    _prioritize_v2_compile_questions,
    _v2_compile_missing_paths,
)
from core.orchestrator.types import Producer, UpdateOp


class SessionUpdateOpsTest(unittest.TestCase):
    def test_apply_updates_supports_set_and_remove(self) -> None:
        config = {
            "source": {
                "particle": "gamma",
                "energy": 1.0,
            },
            "output": {
                "format": "json",
            },
        }
        _apply_updates(
            config,
            [
                UpdateOp(
                    path="source.energy",
                    op="set",
                    value=2.5,
                    producer=Producer.USER_EXPLICIT,
                    confidence=1.0,
                    turn_id=1,
                ),
                UpdateOp(
                    path="source.particle",
                    op="remove",
                    value=None,
                    producer=Producer.USER_EXPLICIT,
                    confidence=1.0,
                    turn_id=1,
                ),
                UpdateOp(
                    path="source.missing_leaf",
                    op="remove",
                    value=None,
                    producer=Producer.USER_EXPLICIT,
                    confidence=1.0,
                    turn_id=1,
                ),
            ],
        )
        self.assertEqual(config["source"]["energy"], 2.5)
        self.assertNotIn("particle", config["source"])
        self.assertEqual(config["output"]["format"], "json")

    def test_v2_missing_paths_merge_source_and_spatial_source_meta(self) -> None:
        slot_debug = {
            "source_v2": {
                "missing_fields": ["particle"],
            },
            "spatial_v2": {
                "source_meta": {
                    "missing_fields": ["energy_mev", "source_type"],
                },
            },
        }
        missing_paths = _v2_compile_missing_paths(slot_debug)
        self.assertEqual(
            missing_paths,
            ["source.particle", "source.energy", "source.type"],
        )
        self.assertEqual(slot_debug["source_v2"]["missing_fields"], ["particle"])
        self.assertEqual(slot_debug["spatial_v2"]["source_meta"]["missing_fields"], ["energy_mev", "source_type"])

        prioritized = _prioritize_v2_compile_questions(
            ["source.particle", "source.energy", "source.type"],
            missing_paths,
            slot_debug,
        )
        self.assertEqual(
            prioritized,
            ["source.energy", "source.type", "source.particle"],
        )

    def test_spatial_review_warnings_prioritize_source_position(self) -> None:
        slot_debug = {
            "spatial_v2": {
                "warnings": ["source_inside_target"],
            },
        }
        missing_paths = _merge_v2_missing_paths(["source.direction"], slot_debug)
        self.assertEqual(missing_paths, ["source.direction", "source.position"])
        prioritized = _prioritize_spatial_questions(
            ["source.direction", "source.position"],
            missing_paths,
            slot_debug,
        )
        self.assertEqual(prioritized, ["source.position", "source.direction"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from core.orchestrator.candidate_preprocess import (
    filter_candidate_by_explicit_targets,
    partition_candidate_by_pending_paths,
)
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp


class CandidatePreprocessTest(unittest.TestCase):
    def test_path_level_filter_keeps_only_requested_updates(self) -> None:
        candidate = CandidateUpdate(
            producer=Producer.BERT_EXTRACTOR,
            intent=Intent.SET,
            target_paths=["output.format", "source.position"],
            updates=[
                UpdateOp(
                    path="source.position",
                    op="set",
                    value=[0, 0, -100],
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.7,
                    turn_id=1,
                ),
                UpdateOp(
                    path="output.format",
                    op="set",
                    value="json",
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.7,
                    turn_id=1,
                ),
            ],
            confidence=0.7,
            rationale="test",
        )

        filtered = filter_candidate_by_explicit_targets(candidate, ["output.format"])

        self.assertEqual([update.path for update in filtered.updates], ["output.format"])
        self.assertEqual(filtered.target_paths, ["output.format"])

    def test_pending_partition_blocks_only_conflicting_paths(self) -> None:
        candidate = CandidateUpdate(
            producer=Producer.BERT_EXTRACTOR,
            intent=Intent.MODIFY,
            target_paths=["materials.selected_materials", "output.format"],
            updates=[
                UpdateOp(
                    path="materials.selected_materials",
                    op="set",
                    value=["G4_Al"],
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.8,
                    turn_id=2,
                ),
                UpdateOp(
                    path="output.format",
                    op="set",
                    value="json",
                    producer=Producer.BERT_EXTRACTOR,
                    confidence=0.8,
                    turn_id=2,
                ),
            ],
            confidence=0.8,
            rationale="test",
        )

        filtered, blocked, replacements = partition_candidate_by_pending_paths(
            candidate,
            ["materials.selected_materials"],
            replace_target_paths=["materials.selected_materials"],
        )

        self.assertEqual([update.path for update in filtered.updates], ["output.format"])
        self.assertEqual([update.path for update in blocked], [])
        self.assertEqual([update.path for update in replacements], ["materials.selected_materials"])


if __name__ == "__main__":
    unittest.main()

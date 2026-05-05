from __future__ import annotations

from core.orchestrator.candidate_pipeline import (
    augment_geometry_targets,
    candidate_has_update_prefix,
    candidate_structure,
    retag_candidate,
    strip_geometry_updates,
    strip_source_updates,
)
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp


def _candidate(*updates: UpdateOp, targets: list[str] | None = None) -> CandidateUpdate:
    return CandidateUpdate(
        producer=Producer.BERT_EXTRACTOR,
        intent=Intent.MODIFY,
        target_paths=targets or [update.path for update in updates],
        updates=list(updates),
        confidence=0.7,
        rationale="test",
    )


def _update(path: str, value: object = "x") -> UpdateOp:
    return UpdateOp(
        path=path,
        op="set",
        value=value,
        producer=Producer.BERT_EXTRACTOR,
        confidence=0.7,
        turn_id=1,
    )


def test_candidate_structure_reads_geometry_structure() -> None:
    candidate = _candidate(_update("geometry.structure", "ring"))
    assert candidate_structure(candidate) == "ring"


def test_strip_geometry_updates_keeps_non_geometry_updates() -> None:
    candidate = _candidate(_update("geometry.structure", "ring"), _update("source.type", "point"))
    stripped = strip_geometry_updates(candidate)
    assert stripped is not None
    assert [update.path for update in stripped.updates] == ["source.type"]
    assert stripped.rationale.endswith("_geometry_stripped")


def test_strip_source_updates_keeps_non_source_updates() -> None:
    candidate = _candidate(_update("source.type", "point"), _update("physics.physics_list", "FTFP_BERT"))
    stripped = strip_source_updates(candidate)
    assert stripped is not None
    assert [update.path for update in stripped.updates] == ["physics.physics_list"]
    assert stripped.rationale.endswith("_source_stripped")


def test_candidate_has_update_prefix() -> None:
    candidate = _candidate(_update("geometry.params.module_x", 10.0))
    assert candidate_has_update_prefix(candidate, "geometry.")
    assert not candidate_has_update_prefix(candidate, "source.")


def test_retag_candidate_rewrites_producer_and_confidence() -> None:
    candidate = _candidate(_update("source.type", "point"))
    retagged = retag_candidate(candidate, producer=Producer.LLM_SEMANTIC_FRAME, confidence=0.9)
    assert retagged is not None
    assert retagged.producer == Producer.LLM_SEMANTIC_FRAME
    assert retagged.confidence == 0.9
    assert retagged.updates[0].producer == Producer.LLM_SEMANTIC_FRAME
    assert retagged.updates[0].confidence == 0.9


def test_augment_geometry_targets_adds_extracted_geometry_paths() -> None:
    user_candidate = _candidate(targets=["geometry.structure"])
    extracted_candidate = _candidate(_update("geometry.graph_program", {"root": "ring"}))
    augmented = augment_geometry_targets(user_candidate, extracted_candidate)
    assert augmented is not None
    assert augmented.target_paths == ["geometry.structure", "geometry.graph_program"]

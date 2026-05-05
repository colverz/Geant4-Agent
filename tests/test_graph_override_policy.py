from __future__ import annotations

from core.orchestrator.graph_override_policy import has_explicit_graph_cue, should_prefer_extracted_graph


def test_explicit_graph_cue_recognizes_boolean_minus() -> None:
    assert has_explicit_graph_cue("box target minus a small cylinder hole", "boolean")


def test_explicit_graph_cue_rejects_missing_boolean_action() -> None:
    assert not has_explicit_graph_cue("a box target near a cylinder detector", "boolean")


def test_ready_slot_geometry_blocks_graph_false_positive_without_explicit_cue() -> None:
    assert not should_prefer_extracted_graph(
        text="a detector box near the target",
        extracted_structure="boolean",
        slot_structure="single_box",
        slot_geometry_ready=True,
    )


def test_unready_slot_geometry_allows_graph_family_candidate() -> None:
    assert should_prefer_extracted_graph(
        text="arrange modules in a ring",
        extracted_structure="ring",
        slot_structure="single_box",
        slot_geometry_ready=False,
    )

from __future__ import annotations

from unittest import mock

from nlu.llm.recommender import recommend_physics_list


def test_recommender_uses_valid_llm_choice() -> None:
    with mock.patch(
        "nlu.llm.recommender.chat",
        return_value={
            "response": '{"physics_list":"QBBC","backup_physics_list":"FTFP_BERT","reasons":["general"],"covered_processes":["em"],"confidence":0.8}'
        },
    ):
        candidate = recommend_physics_list(
            "recommend a Geant4 physics list",
            "particle:gamma",
            "context",
            ["FTFP_BERT", "QBBC"],
            turn_id=1,
            config_path="",
        )

    assert candidate is not None
    values = {update.path: update.value for update in candidate.updates}
    assert values["physics.physics_list"] == "QBBC"
    assert values["physics.backup_physics_list"] == "FTFP_BERT"


def test_recommender_rejects_unknown_llm_choice_and_falls_back() -> None:
    with mock.patch(
        "nlu.llm.recommender.chat",
        return_value={
            "response": '{"physics_list":"UNKNOWN","backup_physics_list":"QBBC","reasons":["bad"],"covered_processes":[],"confidence":0.9,"tool":"run_beam"}'
        },
    ):
        candidate = recommend_physics_list(
            "recommend a Geant4 physics list for gamma no hadron",
            "particle:gamma",
            "context",
            ["FTFP_BERT", "QBBC"],
            turn_id=1,
            config_path="",
        )

    assert candidate is not None
    values = {update.path: update.value for update in candidate.updates}
    assert values["physics.physics_list"] == "FTFP_BERT"
    assert values["physics.backup_physics_list"] == "QBBC"
    assert values["physics.selection_reasons"] == [
        "Selected by rule-backed fallback recommender for current request semantics."
    ]

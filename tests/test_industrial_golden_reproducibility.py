from __future__ import annotations

from copy import deepcopy

from tools.run_industrial_golden_reproducibility import compare_industrial_golden_candidates


def _candidate() -> dict:
    return {
        "case_id": "shielding_lead_gamma_transmission",
        "runtime_fingerprint": {
            "geant4_version": "11.4",
            "runtime_payload_hash": "abc",
            "physics_list": "FTFP_BERT",
            "seed": 1337,
            "events": 10000,
            "threads": 1,
        },
        "metrics": {
            "detector_crossing_count": {"expected": 4659, "tolerance": 0},
            "transmission_factor": {"expected": 0.4659, "tolerance": 0.0},
        },
    }


def test_candidate_comparison_accepts_three_exact_repeats() -> None:
    candidate = _candidate()

    report = compare_industrial_golden_candidates([candidate, deepcopy(candidate), deepcopy(candidate)])

    assert report["ok"]
    assert report["repeat_count"] == 3
    assert report["differences"] == []


def test_candidate_comparison_rejects_metric_drift() -> None:
    baseline = _candidate()
    drifted = deepcopy(baseline)
    drifted["metrics"]["detector_crossing_count"]["expected"] = 4660

    report = compare_industrial_golden_candidates([baseline, drifted, deepcopy(baseline)])

    assert not report["ok"]
    assert report["failure_category"] == "nondeterministic"
    assert report["differences"][0]["field"] == "metrics.detector_crossing_count"


def test_candidate_comparison_rejects_runtime_fingerprint_drift() -> None:
    baseline = _candidate()
    drifted = deepcopy(baseline)
    drifted["runtime_fingerprint"]["runtime_payload_hash"] = "different"

    report = compare_industrial_golden_candidates([baseline, drifted])

    assert not report["ok"]
    assert report["differences"][0]["field"] == "runtime_fingerprint.runtime_payload_hash"

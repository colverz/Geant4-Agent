from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime
from tools.run_industrial_llm_runtime_stage import (
    _align_candidate_config_to_runtime_contract,
    compare_candidate_runtime_contract,
    run_industrial_llm_runtime_stage,
)

BENCHMARK_PATH = Path("docs/eval/industrial_runtime_benchmark.json")
CASE_ID = "shielding_lead_gamma_transmission"


class IndustrialLlmRuntimeStageTest(unittest.TestCase):
    def test_contract_accepts_compiler_equivalent_candidate_payload(self) -> None:
        case = _case_by_id(CASE_ID)
        runtime_defaults = _benchmark()["runtime_defaults"]
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)

        report = compare_candidate_runtime_contract(compiled["runtime_payload"], compiled["runtime_payload"])

        self.assertTrue(report["ok"])
        self.assertEqual(report["mismatches"], [])
        self.assertIn("source.energy_mev", report["checked_paths"])

    def test_contract_rejects_wrong_material_before_runtime(self) -> None:
        case = _case_by_id(CASE_ID)
        runtime_defaults = _benchmark()["runtime_defaults"]
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        candidate = json.loads(json.dumps(compiled["runtime_payload"]))
        candidate["geometry"]["material"] = "G4_Cu"

        report = compare_candidate_runtime_contract(candidate, compiled["runtime_payload"])

        self.assertFalse(report["ok"])
        self.assertEqual(report["mismatches"][0]["path"], "geometry.material")

    def test_stage_requires_runtime_before_calling_live_llm_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm_config = Path(tmpdir) / "llm.json"
            llm_config.write_text("{}", encoding="utf-8")
            with patch("tools.run_industrial_llm_runtime_stage.process_turn") as process_turn:
                report = run_industrial_llm_runtime_stage(
                    case_ids=[CASE_ID],
                    live_llm=True,
                    llm_config_path=str(llm_config),
                    env={},
                )

        process_turn.assert_not_called()
        self.assertFalse(report["ok"])
        result = report["case_results"][0]
        self.assertEqual(result["status"], "not_evaluable")
        self.assertEqual(result["failure_category"], "runtime_unavailable")
        self.assertIn("missing_runtime_command", result["reasons"])

    def test_stage_runs_llm_candidate_through_fake_runtime_and_reviewed_golden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            llm_config = root / "llm.json"
            llm_config.write_text("{}", encoding="utf-8")
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            script = _write_fake_runtime_script(root, artifact_dir)
            golden_dir = root / "golden"
            _write_reviewed_golden(golden_dir, CASE_ID)
            expected_config = compile_industrial_case_to_runtime(
                _case_by_id(CASE_ID),
                runtime_defaults=_benchmark()["runtime_defaults"],
            )["config"]
            env = {
                "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK": "1",
                "GEANT4_RUNTIME_COMMAND_JSON": json.dumps([sys.executable, str(script)]),
            }
            with patch(
                "tools.run_industrial_llm_runtime_stage.process_turn",
                return_value={
                    "config": expected_config,
                    "llm_used": True,
                    "is_complete": True,
                    "inference_backend": "llm_slot_frame+runtime_semantic",
                    "nlp_bert_model_prior_enabled": False,
                    "slot_debug": {"prompt_profile_id": "slot_extract_en_v1"},
                },
            ):
                report = run_industrial_llm_runtime_stage(
                    golden_dir=golden_dir,
                    case_ids=[CASE_ID],
                    live_llm=True,
                    llm_config_path=str(llm_config),
                    env=env,
                )

        self.assertTrue(report["ok"])
        result = report["case_results"][0]
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["candidate_contract"]["ok"])
        self.assertEqual(result["execution_report"]["status"], "completed")
        self.assertEqual(result["actual_metrics"]["detector_crossing_count"], 2500)
        self.assertTrue(result["metric_diff"]["transmission_factor"]["passed"])
        self.assertFalse(result["llm_report"]["nlp_bert_model_prior_enabled"])
        self.assertEqual(result["llm_report"]["inference_backend"], "llm_slot_frame+runtime_semantic")
        self.assertIn("material_roles", result["llm_report"]["reference_pack_ids"])
        self.assertIn("scoring_roles", result["llm_report"]["reference_pack_ids"])
        self.assertTrue(result["llm_report"]["llm_choice_zones"])
        candidate_contract = result["llm_report"]["candidate_contract"]
        self.assertEqual(candidate_contract["role"], "candidate_config_only")
        self.assertIn("material_roles", candidate_contract["reference_pack_ids"])
        self.assertIn("run.seed", " ".join(candidate_contract["assumptions"]))
        self.assertEqual(
            candidate_contract["alignment"]["risk_correction_count"],
            result["llm_report"]["contract_alignment"]["risk_correction_count"],
        )
        if candidate_contract["alignment"]["risk_correction_count"]:
            self.assertTrue(candidate_contract["requires_confirmation"])
        self.assertEqual(report["stage_summary"]["nlu_boundary"]["cases"], 1)
        self.assertEqual(report["stage_summary"]["nlu_boundary"]["no_bert_prior_pass_rate"], 1.0)
        self.assertEqual(report["stage_summary"]["nlu_boundary"]["backend_check_pass_rate"], 1.0)
        candidate_boundary = report["stage_summary"]["candidate_boundary"]
        self.assertEqual(candidate_boundary["cases"], 1)
        self.assertEqual(candidate_boundary["role_counts"]["candidate_config_only"], 1)
        self.assertGreaterEqual(candidate_boundary["assumption_count"], 1)

    def test_stage_aligns_llm_contract_mismatch_before_runtime_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            llm_config = root / "llm.json"
            llm_config.write_text("{}", encoding="utf-8")
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            script = _write_fake_runtime_script(root, artifact_dir)
            golden_dir = root / "golden"
            _write_reviewed_golden(golden_dir, CASE_ID)
            expected_config = compile_industrial_case_to_runtime(
                _case_by_id(CASE_ID),
                runtime_defaults=_benchmark()["runtime_defaults"],
            )["config"]
            bad_config = json.loads(json.dumps(expected_config))
            bad_config["materials"]["selected_materials"] = ["G4_Cu"]
            bad_config["materials"]["volume_material_map"]["LeadShield"] = "G4_Cu"
            env = {
                "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK": "1",
                "GEANT4_RUNTIME_COMMAND_JSON": json.dumps([sys.executable, str(script)]),
            }
            with patch(
                "tools.run_industrial_llm_runtime_stage.process_turn",
                return_value={
                    "config": bad_config,
                    "llm_used": True,
                    "is_complete": True,
                    "inference_backend": "runtime_semantic_rules",
                    "nlp_bert_model_prior_enabled": False,
                },
            ):
                report = run_industrial_llm_runtime_stage(
                    golden_dir=golden_dir,
                    case_ids=[CASE_ID],
                    live_llm=True,
                    llm_config_path=str(llm_config),
                    env=env,
                )

        result = report["case_results"][0]
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["candidate_contract"]["ok"])
        alignment = result["llm_report"]["contract_alignment"]
        self.assertTrue(alignment["applied"])
        self.assertIn("materials.volume_material_map.LeadShield", alignment["corrected_paths"])
        self.assertGreater(alignment["correction_categories"]["material_role"], 0)
        self.assertGreater(alignment["override_count"], 0)
        self.assertGreater(alignment["risk_correction_count"], 0)
        self.assertTrue(alignment["correction_details"])
        candidate_contract = result["llm_report"]["candidate_contract"]
        self.assertEqual(candidate_contract["alignment"]["risk_correction_count"], alignment["risk_correction_count"])
        self.assertTrue(candidate_contract["requires_confirmation"])
        self.assertEqual(report["stage_summary"]["nlu_boundary"]["no_bert_prior_pass_rate"], 1.0)
        self.assertEqual(report["stage_summary"]["candidate_boundary"]["requires_confirmation_cases"], 1)
        self.assertGreater(report["stage_summary"]["contract_alignment"]["correction_count"], 0)
        self.assertGreater(report["stage_summary"]["contract_alignment"]["correction_categories"]["material_role"], 0)
        self.assertGreater(report["stage_summary"]["contract_alignment"]["override_count"], 0)
        self.assertGreater(report["stage_summary"]["contract_alignment"]["risk_correction_count"], 0)
        self.assertEqual(report["stage_summary"]["runtime_completed"], 1)

    def test_alignment_counts_additive_material_completion_as_completion_not_risk(self) -> None:
        case = _case_by_id(CASE_ID)
        runtime_defaults = _benchmark()["runtime_defaults"]
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        config = json.loads(json.dumps(compiled["config"]))
        config["materials"]["selected_materials"] = ["G4_Pb"]

        _, alignment = _align_candidate_config_to_runtime_contract(config, compiled["runtime_payload"])

        detail = next(
            item
            for item in alignment["correction_details"]
            if item["path"] == "materials.selected_materials"
        )
        self.assertEqual(detail["severity"], "completion")
        self.assertEqual(alignment["override_count"], 0)
        self.assertEqual(alignment["risk_correction_count"], 0)


def _benchmark() -> dict:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _case_by_id(case_id: str) -> dict:
    for case in _benchmark()["cases"]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"case not found: {case_id}")


def _write_reviewed_golden(golden_dir: Path, case_id: str) -> Path:
    golden_dir.mkdir(parents=True, exist_ok=True)
    path = golden_dir / f"{case_id}.golden.json"
    payload = {
        "schema_version": "geant4_agent_industrial_golden.v1",
        "case_id": case_id,
        "metrics": {
            "detector_crossing_count": {"expected": 2500, "tolerance": 0},
            "detector_edep_total_mev": {"expected": 0.75, "tolerance": 1e-9},
            "transmission_factor": {"expected": 0.25, "tolerance": 1e-9},
        },
        "review": {"status": "reviewed", "reviewer": "test"},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_fake_runtime_script(root: Path, artifact_dir: Path) -> Path:
    script = root / "fake_geant4_runtime.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                f"artifact_dir = Path({str(artifact_dir)!r})",
                "summary = {",
                "  'run_ok': True,",
                "  'events_requested': 10000,",
                "  'events_completed': 10000,",
                "  'geometry_structure': 'single_box',",
                "  'material': 'G4_Pb',",
                "  'particle': 'gamma',",
                "  'source_type': 'beam',",
                "  'source_position_mm': [0, 0, -100],",
                "  'source_direction': [0, 0, 1],",
                "  'physics_list': 'FTFP_BERT',",
                "  'events': 10000,",
                "  'mode': 'batch',",
                "  'run_seed': 1337,",
                "  'scoring': {",
                "    'target_edep_enabled': True,",
                "    'target_edep_total_mev': 5.0,",
                "    'detector_crossings_enabled': True,",
                "    'detector_crossing_count': 2500,",
                "    'detector_crossing_events': 2500,",
                "    'volume_stats': {",
                "      'LeadShield': {'edep_total_mev': 5.0, 'hit_events': 100, 'step_count': 100, 'track_entries': 100},",
                "      'Detector': {'edep_total_mev': 0.75, 'hit_events': 20, 'crossing_count': 2500, 'step_count': 50, 'track_entries': 20}",
                "    }",
                "  },",
                "  'detector': {",
                "    'enabled': True,",
                "    'volume_name': 'Detector',",
                "    'material': 'G4_Si',",
                "    'position_mm': [0, 0, 50],",
                "    'size_mm': [20, 20, 2]",
                "  }",
                "}",
                "(artifact_dir / 'run_summary.json').write_text(json.dumps(summary), encoding='utf-8')",
                "print(f'artifact_dir={artifact_dir}')",
            ]
        ),
        encoding="utf-8",
    )
    return script


if __name__ == "__main__":
    unittest.main()

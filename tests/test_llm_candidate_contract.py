from __future__ import annotations

import unittest

from core.agent.llm_candidate_contract import (
    LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION,
    LLM_CANDIDATE_ROLE,
    build_llm_candidate_contract,
    build_workflow_llm_candidate_report,
)


class LlmCandidateContractTest(unittest.TestCase):
    def test_candidate_contract_separates_llm_candidate_from_runtime_truth(self) -> None:
        alignment = {
            "applied": True,
            "corrected_paths": ["materials.volume_material_map.Detector"],
            "correction_count": 1,
            "completion_count": 0,
            "override_count": 1,
            "risk_correction_count": 1,
            "correction_categories": {"material_role": 1},
            "correction_details": [
                {
                    "path": "materials.volume_material_map.Detector",
                    "category": "material_role",
                    "severity": "override",
                    "before_present": True,
                    "before": "G4_AIR",
                    "after": "G4_Si",
                }
            ],
        }

        contract = build_llm_candidate_contract(
            user_goal="estimate silicon detector response",
            raw_config={"materials": {"volume_material_map": {"Detector": "G4_AIR"}}},
            aligned_config={"materials": {"volume_material_map": {"Detector": "G4_Si"}}},
            reference_pack_ids=["material_roles", "scoring_roles"],
            choice_zones=[{"field": "run.seed", "policy": "May use runtime default 1337."}],
            alignment_report=alignment,
            is_complete=True,
            runtime_contract_present=True,
        )
        report = contract.to_report()

        self.assertEqual(report["schema_version"], LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION)
        self.assertEqual(report["role"], LLM_CANDIDATE_ROLE)
        self.assertEqual(report["alignment"]["risk_correction_count"], 1)
        self.assertIn("material_roles", report["reference_pack_ids"])
        self.assertTrue(report["assumptions"])
        self.assertTrue(report["requires_confirmation"])
        self.assertNotIn("raw_config", report)

    def test_candidate_contract_records_uncertainty_for_fallback_or_missing_runtime_contract(self) -> None:
        contract = build_llm_candidate_contract(
            user_goal="draft unsupported CAD import",
            raw_config={},
            aligned_config={},
            fallback_reason="llm_unavailable",
            is_complete=False,
            runtime_contract_present=False,
        )
        report = contract.to_report(include_configs=True)

        self.assertGreaterEqual(len(report["uncertainties"]), 3)
        self.assertIn("raw_config", report)
        self.assertEqual(report["alignment"]["correction_count"], 0)

    def test_workflow_candidate_report_tracks_resolution_without_runtime_claims(self) -> None:
        report = build_workflow_llm_candidate_report(
            user_goal="change source energy to 2 MeV",
            llm_used=True,
            fallback_reason=None,
            prompt_profile_id="slot_extract_en_strict_slot_v2",
            inference_backend="llm_slot_frame+runtime_semantic",
            candidate_patch_paths=["source.energy", "source.energy"],
            applied_paths=["source.energy"],
            rejected_paths=[],
            pending_confirmation_paths=[],
        )

        self.assertEqual(report["schema_version"], LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION)
        self.assertEqual(report["role"], LLM_CANDIDATE_ROLE)
        self.assertEqual(report["source"], "process_turn")
        self.assertEqual(report["candidate_boundary"]["proposed_paths"], ["source.energy"])
        self.assertTrue(report["resolution"]["applied_to_session"])
        self.assertIn("Runtime execution", report["assumptions"][0])

    def test_workflow_candidate_report_marks_pending_confirmation(self) -> None:
        report = build_workflow_llm_candidate_report(
            user_goal="replace copper with lead",
            llm_used=False,
            fallback_reason="E_LLM_DISABLED",
            prompt_profile_id=None,
            inference_backend="runtime_semantic_rules",
            candidate_patch_paths=["materials.selected_materials"],
            applied_paths=[],
            rejected_paths=["materials.selected_materials"],
            pending_confirmation_paths=["materials.selected_materials"],
        )

        self.assertFalse(report["resolution"]["applied_to_session"])
        self.assertTrue(report["resolution"]["confirmation_required"])
        self.assertTrue(report["requires_confirmation"])
        self.assertTrue(report["uncertainties"])


if __name__ == "__main__":
    unittest.main()

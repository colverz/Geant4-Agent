from __future__ import annotations

import unittest

from core.agent.evidence_grounding import EvidenceGroundingContext, check_candidate_update_grounding


class EvidenceGroundingCheckerTest(unittest.TestCase):
    def test_accepts_user_grounded_numeric_update_with_unit(self) -> None:
        context = EvidenceGroundingContext.from_mapping(
            {"user_text": "Set source energy to 10 MeV."},
            allowed_paths={"source.energy_mev"},
        )

        result = check_candidate_update_grounding(
            {
                "path": "source.energy_mev",
                "op": "set",
                "value": 10,
                "evidence": [{"text": "10 MeV", "source": "user", "role": "energy"}],
            },
            context=context,
        )

        self.assertTrue(result.ok, result.errors)

    def test_rejects_llm_invented_number(self) -> None:
        context = EvidenceGroundingContext.from_mapping(
            {"user_text": "Set source energy to 10 MeV."},
            allowed_paths={"source.energy_mev"},
        )

        result = check_candidate_update_grounding(
            {
                "path": "source.energy_mev",
                "op": "set",
                "value": 99,
                "evidence": [{"text": "10 MeV", "source": "user", "role": "energy"}],
            },
            context=context,
        )

        self.assertFalse(result.ok)
        self.assertIn("ungrounded_numeric_value:candidate_updates[0].value", result.errors)

    def test_rejects_user_numeric_value_without_unit_for_unit_path(self) -> None:
        context = EvidenceGroundingContext.from_mapping(
            {"user_text": "Set source energy to 10."},
            allowed_paths={"source.energy_mev"},
        )

        result = check_candidate_update_grounding(
            {
                "path": "source.energy_mev",
                "op": "set",
                "value": 10,
                "evidence": [{"text": "10", "source": "user", "role": "energy"}],
            },
            context=context,
        )

        self.assertFalse(result.ok)
        self.assertIn("missing_unit_for_numeric_value:candidate_updates[0].value", result.errors)

    def test_existing_stable_context_number_can_be_preserved(self) -> None:
        context = EvidenceGroundingContext.from_mapping(
            {"stable_context_text": "source.energy_mev=10"},
            allowed_paths={"source.energy_mev"},
        )

        result = check_candidate_update_grounding(
            {
                "path": "source.energy_mev",
                "op": "keep",
                "value": 10,
                "evidence": [{"text": "source.energy_mev=10", "source": "context", "role": "stable_value"}],
            },
            context=context,
        )

        self.assertTrue(result.ok, result.errors)

    def test_capability_kb_can_ground_enum_but_not_numeric_value(self) -> None:
        enum_context = EvidenceGroundingContext.from_mapping(
            {"user_text": "Use gamma."},
            allowed_paths={"source.particle", "source.energy_mev"},
        )
        enum_result = check_candidate_update_grounding(
            {
                "path": "source.particle",
                "op": "set",
                "value": "gamma",
                "evidence": [{"text": "supported_particles includes gamma", "source": "capability_kb", "role": "enum"}],
            },
            context=enum_context,
        )
        self.assertTrue(enum_result.ok, enum_result.errors)

        numeric_context = EvidenceGroundingContext.from_mapping(
            {"user_text": "Set source energy."},
            allowed_paths={"source.energy_mev"},
        )
        numeric_result = check_candidate_update_grounding(
            {
                "path": "source.energy_mev",
                "op": "set",
                "value": 10,
                "evidence": [{"text": "supported source energy is numeric", "source": "capability_kb", "role": "range"}],
            },
            context=numeric_context,
        )

        self.assertFalse(numeric_result.ok)
        self.assertIn("ungrounded_numeric_value:candidate_updates[0].value", numeric_result.errors)
        self.assertIn("capability_kb_cannot_ground_numeric_value:candidate_updates[0].value", numeric_result.errors)

    def test_rejects_internal_path_and_unsupported_kb_grounding(self) -> None:
        context = EvidenceGroundingContext.from_mapping(
            {
                "user_text": "Import a CT scanner.",
                "unsupported_terms": ["ct_scanner_from_free_text"],
            },
            allowed_paths={"geometry.kind"},
        )

        result = check_candidate_update_grounding(
            {
                "path": "runtime.command",
                "op": "set",
                "value": "ct_scanner",
                "evidence": [
                    {
                        "text": "ct_scanner_from_free_text",
                        "source": "capability_kb",
                        "role": "unsupported_geometry",
                    }
                ],
            },
            context=context,
        )

        self.assertFalse(result.ok)
        self.assertIn("value_not_allowed:candidate_updates[0].path", result.errors)
        self.assertIn("internal_path:candidate_updates[0].path", result.errors)
        self.assertIn("unsupported_kb_grounding:candidate_updates[0].evidence[0].text", result.errors)

    def test_rejects_evidence_text_not_present_in_declared_source(self) -> None:
        context = EvidenceGroundingContext.from_mapping(
            {"user_text": "Set source energy to 10 MeV."},
            allowed_paths={"source.energy_mev"},
        )

        result = check_candidate_update_grounding(
            {
                "path": "source.energy_mev",
                "op": "set",
                "value": 10,
                "evidence": [{"text": "20 MeV", "source": "user", "role": "energy"}],
            },
            context=context,
        )

        self.assertFalse(result.ok)
        self.assertIn("evidence_text_not_found:candidate_updates[0].evidence[0].text", result.errors)


if __name__ == "__main__":
    unittest.main()

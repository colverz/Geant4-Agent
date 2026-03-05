from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from core.config.llm_prompt_registry import STRICT_SLOT_PROMPT_PROFILE
from core.slots.slot_mapper import slot_frame_to_candidates
from nlu.llm.slot_frame import build_llm_slot_frame, parse_slot_payload


class LlmSlotFrameTest(unittest.TestCase):
    def test_parse_slot_payload_canonicalizes_common_values(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.88,
            "normalized_text": "geometry.size:1m,1m,1m; materials.primary:copper; source.particle:photon; source.energy:1 MeV; source.position:(0,0,-100 mm); source.direction:+z; output.format:root",
            "target_slots": [
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
                "output.format",
            ],
            "slots": {
                "geometry": {"kind": "cube", "size_triplet_mm": "1 m x 1 m x 1 m"},
                "materials": {"primary": "copper"},
                "source": {
                    "particle": "photon",
                    "energy_mev": "1 MeV",
                    "position_mm": [0, 0, -100],
                    "direction_vec": "+z",
                },
                "output": {"format": "root"},
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.kind, "box")
        self.assertEqual(frame.geometry.size_triplet_mm, [1000.0, 1000.0, 1000.0])
        self.assertEqual(frame.materials.primary, "G4_Cu")
        self.assertEqual(frame.source.particle, "gamma")
        self.assertEqual(frame.source.energy_mev, 1.0)
        self.assertEqual(frame.source.position_mm, [0.0, 0.0, -100.0])
        self.assertEqual(frame.source.direction_vec, [0.0, 0.0, 1.0])
        self.assertEqual(frame.output.format, "root")

    def test_slot_mapper_translates_slot_targets_to_config_targets(self) -> None:
        payload = {
            "intent": "MODIFY",
            "confidence": 0.75,
            "normalized_text": "physics.recommendation_intent:gamma_attenuation; output.format:json",
            "target_slots": ["physics.recommendation_intent", "output.format"],
            "slots": {
                "physics": {"recommendation_intent": "gamma_attenuation"},
                "output": {"format": "json"},
            },
        }
        frame, _ = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        content_candidate, user_candidate = slot_frame_to_candidates(frame, turn_id=4)
        self.assertIsNotNone(content_candidate)
        assert content_candidate is not None
        self.assertEqual({u.path: u.value for u in content_candidate.updates}["output.format"], "json")
        self.assertIn("physics.physics_list", user_candidate.target_paths)
        self.assertIn("output.path", user_candidate.target_paths)

    def test_parse_slot_payload_rejects_placeholder_strings(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.5,
            "normalized_text": "materials.primary:null",
            "target_slots": ["materials.primary"],
            "slots": {"materials": {"primary": "null"}},
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNone(frame)
        self.assertEqual(meta.get("schema_errors"), [])

    def test_parse_slot_payload_canonicalizes_hdf5_alias(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.5,
            "normalized_text": "output.format:h5",
            "target_slots": ["output.format"],
            "slots": {"output": {"format": "h5"}},
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.output.format, "hdf5")

    def test_cylinder_half_length_maps_to_child_hz(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "geometry.kind:cylinder; geometry.radius_mm:30 mm; geometry.half_length_mm:50 mm",
            "target_slots": ["geometry.kind", "geometry.radius_mm", "geometry.half_length_mm"],
            "slots": {
                "geometry": {
                    "kind": "cylinder",
                    "radius_mm": "30 mm",
                    "half_length_mm": "50 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.radius_mm, 30.0)
        self.assertEqual(frame.geometry.half_length_mm, 50.0)
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=5)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_tubs")
        self.assertEqual(mapped["geometry.params.child_rmax"], 30.0)
        self.assertEqual(mapped["geometry.params.child_hz"], 50.0)

    def test_build_llm_slot_frame_backfills_common_chinese_geometry(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "source.particle:gamma; source.energy:1 MeV",
            "target_slots": ["source.particle", "source.energy_mev"],
            "slots": {
                "source": {
                    "particle": "gamma",
                    "energy_mev": "1 MeV",
                }
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "我想做一个10厘米见方的铜立方体靶，用1MeV gamma照射。",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.geometry.kind, "box")
        self.assertEqual(result.frame.geometry.size_triplet_mm, [100.0, 100.0, 100.0])
        self.assertEqual(result.frame.materials.primary, "G4_Cu")
        self.assertIn("geometry.kind", result.stage_trace.get("raw_text_backfill_fields", []))

    def test_build_llm_slot_frame_recovers_from_malformed_slot_payload(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": "high",
            "normalized_text": "",
            "target_slots": ["geometry.kind", "geometry.size_triplet_mm", "materials.primary"],
            "slots": {
                "geometry": "broken",
                "materials": {"primary": "null"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Set up a 1 m x 1 m x 1 m copper box target.",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.geometry.kind, "box")
        self.assertEqual(result.frame.geometry.size_triplet_mm, [1000.0, 1000.0, 1000.0])
        self.assertEqual(result.frame.materials.primary, "G4_Cu")
        self.assertTrue(result.stage_trace.get("repair_used"))
        self.assertEqual(result.stage_trace.get("prompt_profile"), STRICT_SLOT_PROMPT_PROFILE)
        self.assertIn("confidence_not_number", result.stage_trace.get("initial_schema_errors", []))

    def test_build_llm_slot_frame_cylinder_prompt_backfill(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "geometry.kind:cylinder",
            "target_slots": ["geometry.kind", "geometry.radius_mm", "geometry.half_length_mm"],
            "slots": {"geometry": {"kind": "cylinder"}},
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Create a copper cylinder with radius 30 mm and half-length 50 mm.",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.geometry.radius_mm, 30.0)
        self.assertEqual(result.frame.geometry.half_length_mm, 50.0)
        self.assertIn("geometry.radius_mm", result.stage_trace.get("raw_text_backfill_fields", []))

    def test_build_llm_slot_frame_backfills_english_source_vectors(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.82,
            "normalized_text": "geometry.size:1m,1m,1m; materials.primary:copper; source.kind:point; source.particle:gamma",
            "target_slots": [
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.position_mm",
                "source.direction_vec",
            ],
            "slots": {
                "geometry": {"kind": "box", "size_triplet_mm": "1 m x 1 m x 1 m"},
                "materials": {"primary": "copper"},
                "source": {"kind": "point", "particle": "gamma"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1).",
                context_summary="phase=source",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.source.position_mm, [0.0, 0.0, -100.0])
        self.assertEqual(result.frame.source.direction_vec, [0.0, 0.0, 1.0])
        self.assertIn("source.position_mm", result.stage_trace.get("raw_text_backfill_fields", []))
        self.assertIn("source.direction_vec", result.stage_trace.get("raw_text_backfill_fields", []))

    def test_build_llm_slot_frame_prefers_beam_over_pointing_token(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "source.particle:gamma",
            "target_slots": ["source.particle", "source.kind"],
            "slots": {"source": {"particle": "gamma"}},
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Set gamma beam from z- to z+, pointing +z.",
                context_summary="phase=source",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.source.kind, "beam")

    def test_parse_slot_payload_inferrs_cylinder_kind_from_dimensions(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "",
            "target_slots": ["geometry.radius_mm", "geometry.half_length_mm"],
            "slots": {
                "geometry": {
                    "radius_mm": "30 mm",
                    "half_length_mm": "50 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.kind, "cylinder")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from core.orchestrator.types import Producer
from nlu.llm.semantic_frame import build_llm_semantic_frame, parse_semantic_frame_payload


class LlmSemanticFrameTest(unittest.TestCase):
    def test_parse_valid_payload(self) -> None:
        payload = {
            "intent": "SET",
            "target_paths": ["source.type", "source.energy", "source.position", "source.direction"],
            "normalized_text": "source_type: point; energy: 1 MeV; position: (0,0,0); direction: +z",
            "structure_hint": "single_box",
            "confidence": 0.91,
            "updates": [
                {"path": "source.type", "op": "set", "value": "point"},
                {"path": "source.energy", "op": "set", "value": "1 MeV"},
                {"path": "source.position", "op": "set", "value": [0, 0, 0]},
                {"path": "source.direction", "op": "set", "value": "+z"},
            ],
        }
        candidate, user_candidate, meta = parse_semantic_frame_payload(payload, turn_id=3)
        self.assertIsNotNone(candidate)
        self.assertIsNotNone(user_candidate)
        self.assertEqual(meta.get("schema_errors"), [])
        assert candidate is not None
        assert user_candidate is not None
        self.assertEqual(candidate.producer, Producer.LLM_SEMANTIC_FRAME)
        self.assertEqual(user_candidate.producer, Producer.USER_EXPLICIT)
        mapped = {u.path: u.value for u in candidate.updates}
        self.assertEqual(mapped["source.type"], "point")
        self.assertEqual(mapped["source.energy"], 1.0)
        self.assertEqual(mapped["source.position"], {"type": "vector", "value": [0.0, 0.0, 0.0]})
        self.assertEqual(mapped["source.direction"], {"type": "vector", "value": [0.0, 0.0, 1.0]})

    def test_parse_reject_scope_leak(self) -> None:
        payload = {
            "intent": "SET",
            "target_paths": ["planner.hidden"],
            "normalized_text": "planner_hidden: true",
            "structure_hint": "unknown",
            "confidence": 0.5,
            "updates": [{"path": "planner.hidden", "op": "set", "value": True}],
        }
        candidate, user_candidate, meta = parse_semantic_frame_payload(payload, turn_id=1)
        self.assertIsNone(candidate)
        self.assertIsNone(user_candidate)
        errors = " ".join(meta.get("schema_errors", []))
        self.assertIn("out_of_scope", errors)

    def test_parse_canonicalizes_common_aliases(self) -> None:
        payload = {
            "intent": "SET",
            "target_paths": ["geometry.size", "materials.target", "physics.list", "output.type"],
            "normalized_text": "geometry.size:1m,1m,1m; materials.target:copper; physics.list:FTFP_BERT; output.type:root",
            "structure_hint": "single_box",
            "confidence": 0.9,
            "updates": [
                {"path": "geometry.size", "op": "set", "value": "1m,1m,1m"},
                {"path": "materials.target", "op": "set", "value": "copper"},
                {"path": "source.type", "op": "set", "value": "gamma"},
                {"path": "physics.list", "op": "set", "value": "FTFP_BERT"},
                {"path": "output.type", "op": "set", "value": "root"},
            ],
        }
        candidate, user_candidate, meta = parse_semantic_frame_payload(payload, turn_id=7)
        self.assertIsNotNone(candidate)
        self.assertIsNotNone(user_candidate)
        assert candidate is not None
        mapped = {u.path: u.value for u in candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_box")
        self.assertEqual(mapped["geometry.params.module_x"], 1000.0)
        self.assertEqual(mapped["geometry.params.module_y"], 1000.0)
        self.assertEqual(mapped["geometry.params.module_z"], 1000.0)
        self.assertEqual(mapped["materials.selected_materials"], ["G4_Cu"])
        self.assertEqual(mapped["source.particle"], "gamma")
        self.assertEqual(mapped["physics.physics_list"], "FTFP_BERT")
        self.assertEqual(mapped["output.format"], "root")
        self.assertNotIn("source.type", mapped)
        self.assertIn("geometry.params.module_x", user_candidate.target_paths)
        self.assertIn("materials.selected_materials", user_candidate.target_paths)
        self.assertIn("physics.physics_list", user_candidate.target_paths)
        self.assertIn("output.format", user_candidate.target_paths)
        self.assertNotIn("update_invalid_source_type:gamma", meta.get("schema_errors", []))

    def test_build_llm_semantic_frame_rejects_prompt_contract_escape(self) -> None:
        payload = {
            "intent": "SET",
            "target_paths": ["source.type"],
            "normalized_text": "source type: point",
            "structure_hint": "unknown",
            "confidence": 0.8,
            "updates": [{"path": "source.type", "op": "set", "value": "point", "tool": "run_beam"}],
        }
        with patch("nlu.llm.semantic_frame.chat", return_value={"response": json.dumps(payload)}):
            result = build_llm_semantic_frame(
                "Set source type to point.",
                context_summary="phase=source",
                config_path="",
                turn_id=1,
            )

        self.assertFalse(result.ok)
        self.assertIn("unknown_json_key:updates[0].tool", result.schema_errors)


if __name__ == "__main__":
    unittest.main()

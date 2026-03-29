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

    def test_build_llm_slot_frame_prefers_explicit_material_from_user_text(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 1.0,
            "normalized_text": "set material to copper; set geometry to grid",
            "target_slots": ["geometry.kind", "materials.primary"],
            "slots": {
                "geometry": {"kind": "grid"},
                "materials": {"primary": "G4_Cu"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": f"```json\n{json.dumps(payload)}\n```"}):
            result = build_llm_slot_frame(
                "设成 4 x 3 的盒体阵列，材料 硅；点源 gamma 0.511 MeV，方向 +z。",
                context_summary="",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.materials.primary, "G4_Si")

    def test_build_llm_slot_frame_prefers_explicit_beam_from_user_text(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 1.0,
            "normalized_text": "set source kind to point",
            "target_slots": ["source.kind"],
            "slots": {
                "source": {"kind": "point"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": f"```json\n{json.dumps(payload)}\n```"}):
            result = build_llm_slot_frame(
                "请创建完整布尔配置，材料 铝；束流 gamma 0.5 MeV，位置 (0,0,-70) mm，方向 +z。",
                context_summary="",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.source.kind, "beam")

    def test_build_llm_slot_frame_clears_primitive_geometry_when_graph_family_is_explicit(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 1.0,
            "normalized_text": "geometry kind is box; geometry size triplet mm is [12,12,3]",
            "target_slots": ["geometry.kind", "geometry.size_triplet_mm"],
            "slots": {
                "geometry": {"kind": "box", "size_triplet_mm": [12, 12, 3]},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": f"```json\n{json.dumps(payload)}\n```"}):
            result = build_llm_slot_frame(
                "请创建完整的阵列配置：3 x 3 盒体阵列，每个模块 12 mm x 12 mm x 3 mm。",
                context_summary="",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertIsNone(result.frame.geometry.kind)
        self.assertEqual(result.frame.geometry.size_triplet_mm, [12.0, 12.0, 3.0])

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

    def test_parse_slot_payload_applies_geometry_candidate_side_length(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "10 mm copper box",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "box"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "box",
                    "side_length_mm": 10,
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.kind, "box")
        self.assertEqual(frame.geometry.size_triplet_mm, [10.0, 10.0, 10.0])

    def test_parse_slot_payload_applies_relative_source_candidate(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "20 mm outside target center along -z",
            "target_slots": ["source.kind", "source.position_mm", "source.direction_vec"],
            "slots": {"source": {"kind": "point"}},
            "candidates": {
                "source": {
                    "relation": "outside_target_center",
                    "offset_mm": 20,
                    "axis": "-z",
                    "direction_mode": "toward_target_center",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.source.kind, "point")
        self.assertEqual(frame.source.position_mm, [0.0, 0.0, -20.0])
        self.assertEqual(frame.source.direction_vec, [0.0, 0.0, 1.0])

    def test_parse_slot_payload_applies_cylinder_candidate_full_length(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "radius 5 mm height 20 mm copper cylinder",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "cylinder"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "cylinder",
                    "radius_mm": 5,
                    "full_length_mm": 20,
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.kind, "cylinder")
        self.assertEqual(frame.geometry.radius_mm, 5.0)
        self.assertEqual(frame.geometry.half_length_mm, 10.0)

    def test_parse_slot_payload_applies_orb_candidate_radius(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "5 cm lead sphere",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "orb"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "orb",
                    "radius_mm": 50,
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.kind, "orb")
        self.assertEqual(frame.geometry.radius_mm, 50.0)

    def test_parse_slot_payload_applies_orb_candidate_diameter(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "20 mm sphere",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "orb"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "orb",
                    "diameter_mm": 20,
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.radius_mm, 10.0)

    def test_parse_slot_payload_applies_cylinder_candidate_diameter(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "20 mm diameter copper cylinder",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "cylinder"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "cylinder",
                    "diameter_mm": 20,
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.radius_mm, 10.0)

    def test_parse_slot_payload_applies_source_in_front_candidate(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "5 mm in front of the target along -z",
            "target_slots": ["source.kind", "source.position_mm", "source.direction_vec"],
            "slots": {"source": {"kind": "point"}},
            "candidates": {
                "source": {
                    "relation": "in_front_of_target",
                    "offset_mm": 5,
                    "axis": "-z",
                    "direction_mode": "toward_target_face_normal",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.source.position_mm, [0.0, 0.0, -5.0])
        self.assertEqual(frame.source.direction_vec, [0.0, 0.0, 1.0])

    def test_parse_slot_payload_applies_direction_relation_toward_center(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "point source at (0,0,-20) toward target center",
            "target_slots": ["source.kind", "source.position_mm"],
            "slots": {
                "source": {
                    "kind": "point",
                    "position_mm": [0, 0, -20],
                }
            },
            "candidates": {
                "source": {
                    "direction_relation": "toward_target_center",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.source.direction_vec, [0.0, 0.0, 1.0])

    def test_parse_slot_payload_applies_direction_relation_normal_to_face(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "beam normal to target face along -z",
            "target_slots": ["source.kind"],
            "slots": {"source": {"kind": "beam"}},
            "candidates": {
                "source": {
                    "axis": "-z",
                    "direction_relation": "normal_to_target_face",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.source.direction_vec, [0.0, 0.0, 1.0])

    def test_parse_slot_payload_applies_direction_relation_toward_face(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "point source toward target face along -z",
            "target_slots": ["source.kind"],
            "slots": {"source": {"kind": "point"}},
            "candidates": {
                "source": {
                    "axis": "-z",
                    "direction_relation": "toward_target_face",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.source.direction_vec, [0.0, 0.0, 1.0])

    def test_parse_slot_payload_applies_box_thickness_candidate(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "2 mm thick lead slab",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "box"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "box",
                    "thickness_mm": 2,
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.size_triplet_mm, [10.0, 10.0, 2.0])

    def test_parse_slot_payload_applies_plate_size_candidate(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.7,
            "normalized_text": "10 mm x 20 mm plate",
            "target_slots": ["geometry.kind"],
            "slots": {"geometry": {"kind": "box"}},
            "candidates": {
                "geometry": {
                    "kind_candidate": "box",
                    "plate_size_xy_mm": [10, 20],
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        self.assertEqual(frame.geometry.size_triplet_mm, [10.0, 20.0, 1.0])

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

    def test_cons_slots_map_to_single_cons(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "geometry.kind:cons; geometry.radius1_mm:20 mm; geometry.radius2_mm:40 mm; geometry.half_length_mm:60 mm",
            "target_slots": ["geometry.kind", "geometry.radius1_mm", "geometry.radius2_mm", "geometry.half_length_mm"],
            "slots": {
                "geometry": {
                    "kind": "cons",
                    "radius1_mm": "20 mm",
                    "radius2_mm": "40 mm",
                    "half_length_mm": "60 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=6)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_cons")
        self.assertEqual(mapped["geometry.params.rmax1"], 20.0)
        self.assertEqual(mapped["geometry.params.rmax2"], 40.0)
        self.assertEqual(mapped["geometry.params.child_hz"], 60.0)

    def test_orb_slots_map_to_single_orb(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "geometry.kind:orb; geometry.radius_mm:25 mm",
            "target_slots": ["geometry.kind", "geometry.radius_mm"],
            "slots": {
                "geometry": {
                    "kind": "orb",
                    "radius_mm": "25 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=6)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_orb")
        self.assertEqual(mapped["geometry.params.child_rmax"], 25.0)

    def test_trd_slots_map_to_single_trd(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.82,
            "normalized_text": "geometry.kind:trd; geometry.x1_mm:10 mm; geometry.x2_mm:20 mm; geometry.y1_mm:30 mm; geometry.y2_mm:40 mm; geometry.z_mm:50 mm",
            "target_slots": ["geometry.kind", "geometry.x1_mm", "geometry.x2_mm", "geometry.y1_mm", "geometry.y2_mm", "geometry.z_mm"],
            "slots": {
                "geometry": {
                    "kind": "trd",
                    "x1_mm": "10 mm",
                    "x2_mm": "20 mm",
                    "y1_mm": "30 mm",
                    "y2_mm": "40 mm",
                    "z_mm": "50 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=7)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_trd")
        self.assertEqual(mapped["geometry.params.x1"], 10.0)
        self.assertEqual(mapped["geometry.params.x2"], 20.0)
        self.assertEqual(mapped["geometry.params.y1"], 30.0)
        self.assertEqual(mapped["geometry.params.y2"], 40.0)
        self.assertEqual(mapped["geometry.params.module_z"], 50.0)

    def test_parse_slot_payload_maps_trap_slots(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.84,
            "normalized_text": (
                "geometry.kind:trap; geometry.trap_x1_mm:10 mm; geometry.trap_x2_mm:12 mm; "
                "geometry.trap_x3_mm:14 mm; geometry.trap_x4_mm:16 mm; "
                "geometry.trap_y1_mm:20 mm; geometry.trap_y2_mm:24 mm; geometry.trap_z_mm:30 mm"
            ),
            "target_slots": [
                "geometry.kind",
                "geometry.trap_x1_mm",
                "geometry.trap_x2_mm",
                "geometry.trap_x3_mm",
                "geometry.trap_x4_mm",
                "geometry.trap_y1_mm",
                "geometry.trap_y2_mm",
                "geometry.trap_z_mm",
            ],
            "slots": {
                "geometry": {
                    "kind": "trap",
                    "trap_x1_mm": "10 mm",
                    "trap_x2_mm": "12 mm",
                    "trap_x3_mm": "14 mm",
                    "trap_x4_mm": "16 mm",
                    "trap_y1_mm": "20 mm",
                    "trap_y2_mm": "24 mm",
                    "trap_z_mm": "30 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=8)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_trap")
        self.assertEqual(mapped["geometry.params.trap_x1"], 10.0)
        self.assertEqual(mapped["geometry.params.trap_x4"], 16.0)
        self.assertEqual(mapped["geometry.params.trap_y2"], 24.0)
        self.assertEqual(mapped["geometry.params.trap_z"], 30.0)

    def test_parse_slot_payload_maps_torus_slots(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.79,
            "normalized_text": (
                "geometry.kind:torus; geometry.torus_major_radius_mm:60 mm; "
                "geometry.torus_minor_radius_mm:8 mm"
            ),
            "target_slots": [
                "geometry.kind",
                "geometry.torus_major_radius_mm",
                "geometry.torus_minor_radius_mm",
            ],
            "slots": {
                "geometry": {
                    "kind": "torus",
                    "torus_major_radius_mm": "60 mm",
                    "torus_minor_radius_mm": "8 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=9)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_torus")
        self.assertEqual(mapped["geometry.params.torus_rtor"], 60.0)
        self.assertEqual(mapped["geometry.params.torus_rmax"], 8.0)

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
                "\u6211\u60f3\u505a\u4e00\u4e2a10\u5398\u7c73\u89c1\u65b9\u7684\u94dc\u7acb\u65b9\u4f53\u9776\uff0c\u75281MeV gamma\u7167\u5c04\u3002",
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

    def test_build_llm_slot_frame_backfills_para_from_user_text(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.76,
            "normalized_text": "geometry.kind:para",
            "target_slots": [
                "geometry.kind",
                "geometry.para_x_mm",
                "geometry.para_y_mm",
                "geometry.para_z_mm",
                "geometry.para_alpha_deg",
                "geometry.para_theta_deg",
                "geometry.para_phi_deg",
            ],
            "slots": {"geometry": {"kind": "para"}},
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Use a skewed box with para_x 20 mm, para_y 30 mm, para_z 40 mm, para_alpha 10 deg, para_theta 15 deg, para_phi 25 deg.",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.geometry.kind, "para")
        self.assertEqual(result.frame.geometry.para_x_mm, 20.0)
        self.assertEqual(result.frame.geometry.para_y_mm, 30.0)
        self.assertEqual(result.frame.geometry.para_z_mm, 40.0)
        self.assertEqual(result.frame.geometry.para_alpha_deg, 10.0)
        self.assertEqual(result.frame.geometry.para_theta_deg, 15.0)
        self.assertEqual(result.frame.geometry.para_phi_deg, 25.0)

    def test_build_llm_slot_frame_backfills_ellipsoid_from_user_text(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": "geometry.kind:ellipsoid",
            "target_slots": [
                "geometry.kind",
                "geometry.ellipsoid_ax_mm",
                "geometry.ellipsoid_by_mm",
                "geometry.ellipsoid_cz_mm",
            ],
            "slots": {"geometry": {"kind": "ellipsoid"}},
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Build an ellipsoid target with ax 12 mm, by 18 mm, cz 24 mm.",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.geometry.kind, "ellipsoid")
        self.assertEqual(result.frame.geometry.ellipsoid_ax_mm, 12.0)
        self.assertEqual(result.frame.geometry.ellipsoid_by_mm, 18.0)
        self.assertEqual(result.frame.geometry.ellipsoid_cz_mm, 24.0)

    def test_parse_slot_payload_maps_elltube_slots(self) -> None:
        payload = {
            "intent": "SET",
            "confidence": 0.8,
            "normalized_text": (
                "geometry.kind:elltube; geometry.elltube_ax_mm:12 mm; "
                "geometry.elltube_by_mm:18 mm; geometry.elltube_hz_mm:24 mm"
            ),
            "target_slots": [
                "geometry.kind",
                "geometry.elltube_ax_mm",
                "geometry.elltube_by_mm",
                "geometry.elltube_hz_mm",
            ],
            "slots": {
                "geometry": {
                    "kind": "elltube",
                    "elltube_ax_mm": "12 mm",
                    "elltube_by_mm": "18 mm",
                    "elltube_hz_mm": "24 mm",
                }
            },
        }
        frame, meta = parse_slot_payload(payload)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(meta.get("schema_errors"), [])
        content_candidate, _ = slot_frame_to_candidates(frame, turn_id=10)
        assert content_candidate is not None
        mapped = {u.path: u.value for u in content_candidate.updates}
        self.assertEqual(mapped["geometry.structure"], "single_elltube")
        self.assertEqual(mapped["geometry.params.elltube_ax"], 12.0)
        self.assertEqual(mapped["geometry.params.elltube_by"], 18.0)
        self.assertEqual(mapped["geometry.params.elltube_hz"], 24.0)

    def test_build_llm_slot_frame_backfills_polyhedra_from_user_text(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.74,
            "normalized_text": (
                "geometry.kind:polyhedra; geometry.polyhedra_sides:8; "
                "geometry.z_planes_mm:(-20,0,20); geometry.radii_mm:(10,12,14)"
            ),
            "target_slots": [
                "geometry.kind",
                "geometry.polyhedra_sides",
                "geometry.z_planes_mm",
                "geometry.radii_mm",
            ],
            "slots": {
                "geometry": {
                    "kind": "polyhedra",
                    "polyhedra_sides": 8,
                    "z_planes_mm": [-20, 0, 20],
                    "radii_mm": [10, 12, 14],
                }
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Build a polyhedra with polyhedra_sides 8, z planes -20 0 20 mm, radii 10 12 14 mm.",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.geometry.kind, "polyhedra")
        self.assertEqual(result.frame.geometry.polyhedra_sides, 8)
        self.assertEqual(result.frame.geometry.z_planes_mm, [-20.0, 0.0, 20.0])
        self.assertEqual(result.frame.geometry.radii_mm, [10.0, 12.0, 14.0])

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

    def test_build_llm_slot_frame_backfills_relative_source_from_center(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.85,
            "normalized_text": (
                "geometry.kind:box; geometry.size_triplet_mm:(10,10,10); materials.primary:copper; "
                "source.kind:point; source.particle:gamma; source.energy_mev:1000; output.format:json"
            ),
            "target_slots": [
                "geometry.kind",
                "geometry.size_triplet_mm",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
                "output.format",
            ],
            "slots": {
                "geometry": {"kind": "box", "size_triplet_mm": [10, 10, 10]},
                "materials": {"primary": "copper"},
                "source": {"kind": "point", "particle": "gamma", "energy_mev": 1000},
                "output": {"format": "json"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "铜靶，10mm立方体，gamma粒子从中心点20mm外入射，点源，1GeV，输出格式为json。",
                context_summary="phase=source",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.source.position_mm, [0.0, 0.0, -20.0])
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

    def test_build_llm_slot_frame_strips_llm_invented_beam_direction_without_text_evidence(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.9,
            "normalized_text": "source.kind:beam; source.particle:gamma; source.energy_mev:5; source.position_mm:(0,0,-250)",
            "target_slots": [
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
            ],
            "slots": {
                "source": {
                    "kind": "beam",
                    "particle": "gamma",
                    "energy_mev": 5.0,
                    "position_mm": [0, 0, -250],
                    "direction_vec": "+z",
                }
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "gamma beam 5 MeV from (0,0,-250) mm",
                context_summary="phase=source",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.source.kind, "beam")
        self.assertIsNone(result.frame.source.direction_vec)
        self.assertIn("source.direction_vec", result.stage_trace.get("unsupported_llm_fields", []))

    def test_build_llm_slot_frame_strips_llm_invented_source_energy_without_text_evidence(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.9,
            "normalized_text": "source.kind:point; source.particle:gamma; source.energy_mev:5; source.position_mm:(0,0,-20); source.direction_vec:+z",
            "target_slots": [
                "source.kind",
                "source.particle",
                "source.position_mm",
                "source.direction_vec",
            ],
            "slots": {
                "source": {
                    "kind": "point",
                    "particle": "gamma",
                    "energy_mev": 5.0,
                    "position_mm": [0, 0, -20],
                    "direction_vec": "+z",
                }
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "在靶前表面外5 mm放一个gamma点源，朝靶心入射。",
                context_summary="phase=source",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.source.kind, "point")
        self.assertEqual(result.frame.source.particle, "gamma")
        self.assertIsNone(result.frame.source.energy_mev)
        self.assertIn("source.energy_mev", result.stage_trace.get("unsupported_llm_fields", []))

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

    def test_build_llm_slot_frame_prunes_unresolved_targets_from_uncertain_turn(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.92,
            "normalized_text": (
                "geometry.kind:box; geometry.size:1m,1m,1m; materials.primary:copper; "
                "source.kind:beam; source.particle:gamma; source.energy_mev:1; "
                "source.position_mm:(0,0,-100); source.direction_vec:+z; output.format:csv"
            ),
            "target_slots": [
                "geometry.kind",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "source.position_mm",
                "source.direction_vec",
                "output.format",
            ],
            "slots": {
                "geometry": {"kind": "box", "size_triplet_mm": "1 m x 1 m x 1 m"},
                "materials": {"primary": "copper"},
                "source": {
                    "kind": "beam",
                    "particle": "gamma",
                    "energy_mev": 1.0,
                    "position_mm": [0, 0, -100],
                    "direction_vec": "+z",
                },
                "output": {"format": "csv"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "Need a quick cargo-screening toy setup with forward photons, but I have not fixed the target shape, material, source energy, or output format yet.",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.intent.value, "QUESTION")
        self.assertIsNone(result.frame.geometry.kind)
        self.assertIsNone(result.frame.geometry.size_triplet_mm)
        self.assertIsNone(result.frame.materials.primary)
        self.assertIsNone(result.frame.source.kind)
        self.assertEqual(result.frame.source.particle, "gamma")
        self.assertIsNone(result.frame.source.energy_mev)
        self.assertIsNone(result.frame.source.position_mm)
        self.assertIsNone(result.frame.source.direction_vec)
        self.assertIsNone(result.frame.output.format)
        self.assertIn("geometry.kind", result.stage_trace.get("pruned_unresolved_targets", []))
        self.assertIn("materials.primary", result.stage_trace.get("pruned_unresolved_targets", []))
        self.assertIn("source.energy_mev", result.stage_trace.get("pruned_unresolved_targets", []))
        self.assertIn("output.format", result.stage_trace.get("pruned_unresolved_targets", []))

    def test_build_llm_slot_frame_prunes_general_missing_start_but_keeps_grounded_particle(self) -> None:
        llm_payload = {
            "intent": "SET",
            "confidence": 0.9,
            "normalized_text": (
                "geometry.kind:box; materials.primary:lead; source.kind:point; "
                "source.particle:neutron; source.energy_mev:14; output.format:xml"
            ),
            "target_slots": [
                "geometry.kind",
                "materials.primary",
                "source.kind",
                "source.particle",
                "source.energy_mev",
                "output.format",
            ],
            "slots": {
                "geometry": {"kind": "box", "size_triplet_mm": "200 mm x 200 mm x 200 mm"},
                "materials": {"primary": "lead"},
                "source": {"kind": "point", "particle": "neutron", "energy_mev": 14.0},
                "output": {"format": "xml"},
            },
        }
        with patch("nlu.llm.slot_frame.chat", return_value={"response": json.dumps(llm_payload)}):
            result = build_llm_slot_frame(
                "我想做一个中子屏蔽的概念验证，现在先给一个缺参开头。",
                context_summary="phase=geometry",
                config_path="",
            )
        self.assertTrue(result.ok)
        assert result.frame is not None
        self.assertEqual(result.frame.intent.value, "QUESTION")
        self.assertIsNone(result.frame.geometry.kind)
        self.assertIsNone(result.frame.materials.primary)
        self.assertIsNone(result.frame.source.kind)
        self.assertEqual(result.frame.source.particle, "neutron")
        self.assertIsNone(result.frame.source.energy_mev)
        self.assertIsNone(result.frame.output.format)


if __name__ == "__main__":
    unittest.main()

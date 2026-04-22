from __future__ import annotations

import unittest
from unittest.mock import patch

import core.slots.slot_mapper as slot_mapper_module
from core.contracts.slots import GeometrySlots, MaterialsSlots, OutputSlots, PhysicsSlots, SlotFrame, SourceSlots
from core.orchestrator.session_manager import process_turn, reset_session
from core.orchestrator.types import CandidateUpdate, Intent, Producer
from core.simulation import build_simulation_spec
from mcp.geant4.runtime_payload import build_runtime_payload
from nlu.llm.slot_frame import LlmSlotBuildResult


def _complete_frame() -> SlotFrame:
    return SlotFrame(
        intent=Intent.SET,
        confidence=1.0,
        normalized_text="complete switch contract frame",
        target_slots=[
            "geometry.kind",
            "geometry.size_triplet_mm",
            "materials.primary",
            "source.kind",
            "source.particle",
            "source.energy_mev",
            "source.position_mm",
            "source.direction_vec",
            "physics.explicit_list",
            "output.format",
        ],
        geometry=GeometrySlots(kind="box", size_triplet_mm=[10.0, 20.0, 30.0]),
        materials=MaterialsSlots(primary="G4_Cu"),
        source=SourceSlots(
            kind="point",
            particle="gamma",
            energy_mev=1.0,
            position_mm=[0.0, 0.0, -20.0],
            direction_vec=[0.0, 0.0, 1.0],
        ),
        physics=PhysicsSlots(explicit_list="FTFP_BERT"),
        output=OutputSlots(format="json"),
    )


def _slot_result(frame: SlotFrame) -> LlmSlotBuildResult:
    return LlmSlotBuildResult(
        ok=True,
        frame=frame,
        normalized_text=frame.normalized_text,
        confidence=1.0,
        llm_raw="{}",
        fallback_reason=None,
        schema_errors=[],
        stage_trace={"final_status": "ok"},
    )


def _empty_semantic_candidate() -> tuple[CandidateUpdate, dict]:
    return (
        CandidateUpdate(
            producer=Producer.BERT_EXTRACTOR,
            intent=Intent.SET,
            target_paths=[],
            updates=[],
            confidence=0.0,
            rationale="empty_switch_contract_sidecar",
        ),
        {"graph_candidates": [], "graph_choice": {}, "inference_backend": "test_empty_sidecar"},
    )


class PipelineSwitchContractTest(unittest.TestCase):
    def tearDown(self) -> None:
        for suffix in ("legacy-legacy", "v2-legacy", "legacy-v2", "v2-v2"):
            reset_session(f"switch-contract-{suffix}")

    def _run_case(self, *, geometry: str, source: str) -> dict:
        frame = _complete_frame()
        with (
            patch("core.orchestrator.session_manager.build_llm_slot_frame", return_value=_slot_result(frame)),
            patch("core.orchestrator.session_manager.extract_candidates_from_normalized_text", return_value=_empty_semantic_candidate()),
            patch(
                "core.slots.slot_mapper.build_legacy_geometry_updates",
                wraps=slot_mapper_module.build_legacy_geometry_updates,
            ) as legacy_geometry,
            patch(
                "core.slots.slot_mapper.build_v2_geometry_updates",
                wraps=slot_mapper_module.build_v2_geometry_updates,
            ) as v2_geometry,
            patch(
                "core.slots.slot_mapper.build_legacy_source_updates",
                wraps=slot_mapper_module.build_legacy_source_updates,
            ) as legacy_source,
            patch(
                "core.slots.slot_mapper.build_v2_source_updates",
                wraps=slot_mapper_module.build_v2_source_updates,
            ) as v2_source,
            patch(
                "core.slots.slot_mapper.build_v2_spatial_updates",
                wraps=slot_mapper_module.build_v2_spatial_updates,
            ) as v2_spatial,
        ):
            out = process_turn(
                {
                    "session_id": f"switch-contract-{geometry}-{source}",
                    "text": "10 x 20 x 30 mm copper box, 1 MeV gamma point source at (0,0,-20) mm along +z",
                    "llm_router": True,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": geometry,
                    "source_pipeline": source,
                    "enable_compare": False,
                    "enable_interpreter": False,
                },
                ollama_config_path="",
            )

        out["_pipeline_calls"] = {
            "legacy_geometry": legacy_geometry.call_count,
            "v2_geometry": v2_geometry.call_count,
            "legacy_source": legacy_source.call_count,
            "v2_source": v2_source.call_count,
            "v2_spatial": v2_spatial.call_count,
        }
        return out

    def test_all_legacy_v2_switch_combinations_keep_pipeline_boundaries(self) -> None:
        cases = {
            ("legacy", "legacy"): {
                "calls": {"legacy_geometry": 1, "v2_geometry": 0, "legacy_source": 1, "v2_source": 0, "v2_spatial": 0},
                "debug_keys": set(),
            },
            ("v2", "legacy"): {
                "calls": {"legacy_geometry": 0, "v2_geometry": 1, "legacy_source": 1, "v2_source": 0, "v2_spatial": 0},
                "debug_keys": {"geometry_v2"},
            },
            ("legacy", "v2"): {
                "calls": {"legacy_geometry": 1, "v2_geometry": 0, "legacy_source": 0, "v2_source": 1, "v2_spatial": 0},
                "debug_keys": {"source_v2"},
            },
            ("v2", "v2"): {
                "calls": {"legacy_geometry": 0, "v2_geometry": 0, "legacy_source": 0, "v2_source": 0, "v2_spatial": 1},
                "debug_keys": {"geometry_v2", "source_v2", "spatial_v2"},
            },
        }

        for (geometry, source), expected in cases.items():
            with self.subTest(geometry=geometry, source=source):
                out = self._run_case(geometry=geometry, source=source)

                self.assertEqual(out["pipelines"], {"geometry": geometry, "source": source})
                self.assertEqual(out["_pipeline_calls"], expected["calls"])

                slot_debug = out["slot_debug"]
                for key in ("geometry_v2", "source_v2", "spatial_v2"):
                    if key in expected["debug_keys"]:
                        self.assertIn(key, slot_debug)
                    else:
                        self.assertNotIn(key, slot_debug)

                self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
                self.assertEqual(out["config"]["geometry"]["params"]["module_x"], 10.0)
                self.assertEqual(out["config"]["source"]["type"], "point")
                self.assertEqual(out["config"]["source"]["particle"], "gamma")
                self.assertEqual(out["config"]["physics"]["physics_list"], "FTFP_BERT")
                self.assertEqual(out["config"]["output"]["format"], "json")

                runtime_payload = build_runtime_payload(build_simulation_spec(out["config"], events=1))
                self.assertEqual(runtime_payload["structure"], "single_box")
                self.assertEqual(runtime_payload["source_type"], "point")
                self.assertEqual(runtime_payload["particle"], "gamma")
                self.assertEqual(runtime_payload["physics_list"], "FTFP_BERT")


if __name__ == "__main__":
    unittest.main()

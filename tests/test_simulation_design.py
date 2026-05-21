from __future__ import annotations

import json
import unittest
from unittest import mock

from core.agent.simulation_design import (
    ALLOWED_NEXT_ACTIONS,
    build_simulation_design_candidate,
    build_simulation_design_reference_pack,
    check_simulation_design_capability,
    load_simulation_design_annotations,
    validate_simulation_design_annotations,
)
from core.agent.simulation_design_llm import normalize_simulation_design_candidate
from core.agent.simulation_design_llm import build_simulation_design_prompt
from core.orchestrator.session_manager import process_turn, reset_session
from ui.web.request_router import handle_post_request


class SimulationDesignKnowledgeTest(unittest.TestCase):
    def test_required_materials_have_usage_tags(self) -> None:
        annotations = load_simulation_design_annotations()

        self.assertEqual(validate_simulation_design_annotations(annotations), [])
        for material in (
            "G4_Pb",
            "G4_WATER",
            "G4_Si",
            "G4_AIR",
            "G4_Galactic",
            "G4_POLYETHYLENE",
            "G4_PLASTIC_SC_VINYLTOLUENE",
        ):
            with self.subTest(material=material):
                self.assertTrue(annotations["materials"][material]["tags"])
        self.assertIn("beam_for_transmission", annotations["sources"]["beam"]["tags"])
        self.assertIn("isotropic_sampling_supported", annotations["sources"]["isotropic"]["tags"])
        self.assertIn("target_edep_supported", annotations["scoring"]["target_edep"]["tags"])
        self.assertIn("depth_bins_supported", annotations["scoring"]["depth_bins"]["tags"])
        self.assertIn("single_box_supported", annotations["geometry"]["single_box"]["tags"])
        self.assertIn("step_wedge_supported", annotations["geometry"]["step_wedge"]["tags"])
        self.assertIn("slab_approximation_requires_user_approval", annotations["geometry"]["pipe"]["tags"])
        self.assertIn("vacuum", annotations["materials"]["G4_Galactic"]["tags"])

    def test_reference_pack_provides_full_catalog_with_non_binding_query_hints(self) -> None:
        pack = build_simulation_design_reference_pack(
            "Design a polyethylene neutron moderation benchmark with plane crossing scoring."
        )

        self.assertEqual(pack["schema_version"], "geant4_agent_simulation_design_reference_pack.v1")
        self.assertIn("single_box", pack["runtime_capabilities"]["supported_geometry"])
        self.assertIn("beam", pack["runtime_capabilities"]["supported_sources"])
        self.assertIn("plane_crossing_count", pack["runtime_capabilities"]["supported_scoring"])
        material_ids = {item["id"] for item in pack["materials"]}
        scoring_ids = {item["id"] for item in pack["scoring"]}
        hinted_material_ids = {item["id"] for item in pack["query_hints"]["materials"]}
        hinted_scoring_ids = {item["id"] for item in pack["query_hints"]["scoring"]}
        self.assertEqual(pack["selection_policy"]["catalog_scope"], "full_catalog")
        self.assertTrue(pack["selection_policy"]["query_hints_are_non_binding"])
        self.assertIn("G4_POLYETHYLENE", material_ids)
        self.assertIn("G4_Pb", material_ids)
        self.assertIn("G4_Galactic", material_ids)
        self.assertIn("plane_crossing_count", scoring_ids)
        self.assertIn("G4_POLYETHYLENE", hinted_material_ids)
        self.assertIn("plane_crossing_count", hinted_scoring_ids)

    def test_simulation_design_prompt_uses_full_catalog_not_extraction_subset(self) -> None:
        pack = build_simulation_design_reference_pack("真空环境中的 gamma 传输模拟")
        prompt = build_simulation_design_prompt("真空环境中的 gamma 传输模拟", pack, lang="zh")

        self.assertIn("full catalogs", prompt)
        self.assertIn("query_hints are only orientation hints", prompt)
        self.assertIn("Do not behave like a keyword extractor", prompt)
        self.assertIn("G4_Galactic", prompt)
        self.assertIn("G4_Pb", prompt)

    def test_supported_lead_gamma_transmission_design_can_build_candidate_config(self) -> None:
        candidate = build_simulation_design_candidate(
            "Use a 1 MeV gamma beam through lead shielding and measure transmission at a silicon detector."
        ).to_dict()

        self.assertEqual(candidate["schema_version"], "geant4_agent_simulation_design_candidate.v1")
        self.assertEqual(candidate["next_action"], "build_candidate_config")
        self.assertEqual(candidate["recommended_setup"]["geometry"], "single_box")
        self.assertEqual(candidate["recommended_setup"]["material"], "G4_Pb")
        self.assertIn("detector_crossing_count", candidate["observables"])
        self.assertEqual(candidate["unsupported_capabilities"], [])

    def test_pipe_corrosion_requires_user_approved_approximation(self) -> None:
        candidate = build_simulation_design_candidate(
            "Compare detector transmission for a steel pipe section with corrosion thinning."
        ).to_dict()

        self.assertEqual(candidate["next_action"], "ask_user_to_choose_approximation")
        self.assertTrue(candidate["simplifications"])
        self.assertTrue(candidate["user_decisions_required"])
        self.assertEqual(candidate["capability_check"]["requires_user_approval"], True)

    def test_water_phantom_depth_dose_can_build_supported_depth_bins_candidate(self) -> None:
        candidate = build_simulation_design_candidate(
            "Send a proton beam into a water phantom and score depth-binned dose."
        ).to_dict()

        self.assertEqual(candidate["next_action"], "build_candidate_config")
        self.assertIn("depth_bins", candidate["observables"])
        self.assertEqual(candidate["unsupported_capabilities"], [])
        self.assertTrue(candidate["capability_check"]["supported"])

    def test_detector_response_prefers_detector_observables_not_region_contrast(self) -> None:
        candidate = build_simulation_design_candidate(
            "Place a silicon detector behind an air gap and measure gamma detector response."
        ).to_dict()

        self.assertEqual(candidate["next_action"], "build_candidate_config")
        self.assertIn("detector_crossing_count", candidate["observables"])
        self.assertIn("detector_edep", candidate["observables"])
        self.assertNotIn("region_contrast", candidate["observables"])

    def test_vacuum_environment_uses_g4_galactic_not_air(self) -> None:
        candidate = build_simulation_design_candidate("真空环境中的 1 MeV gamma 点源传输模拟").to_dict()
        pack = build_simulation_design_reference_pack("真空环境中的 1 MeV gamma 点源传输模拟")

        self.assertEqual(candidate["recommended_setup"]["material"], "G4_Galactic")
        self.assertEqual(candidate["recommended_setup"]["environment_material"], "G4_Galactic")
        self.assertNotEqual(candidate["recommended_setup"]["material"], "G4_AIR")
        self.assertIn("materials:G4_Galactic", candidate["knowledge_references"])
        self.assertIn("G4_Galactic", {item["id"] for item in pack["materials"]})

    def test_llm_normalization_rejects_air_as_vacuum_substitute(self) -> None:
        raw = {
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_AIR",
                "source": "beam",
                "scoring": ["target_edep"],
            },
            "observables": ["target_edep"],
            "knowledge_references": ["materials:G4_AIR"],
            "unsupported_capabilities": [],
        }

        normalized = normalize_simulation_design_candidate(raw, "vacuum beamline transport benchmark")

        self.assertEqual(normalized["recommended_setup"]["material"], "G4_Galactic")
        self.assertEqual(normalized["recommended_setup"]["environment_material"], "G4_Galactic")
        self.assertIn("materials:G4_Galactic", normalized["knowledge_references"])

    def test_llm_normalization_preserves_design_rationale_and_alternatives(self) -> None:
        raw = {
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "scoring": ["detector_crossing_count"],
                "design_rationale": "Lead is selected because the goal is gamma attenuation.",
                "alternatives_considered": ["G4_WATER rejected because it is a phantom material."],
            },
            "observables": ["detector_crossing_count"],
        }

        normalized = normalize_simulation_design_candidate(raw, "gamma shielding benchmark")

        self.assertIn("gamma attenuation", normalized["recommended_setup"]["design_rationale"])
        self.assertEqual(len(normalized["recommended_setup"]["alternatives_considered"]), 1)

    def test_chinese_void_contrast_selects_void_references_and_supported_region_scoring(self) -> None:
        pack = build_simulation_design_reference_pack("铝块内部空洞缺陷的区域 contrast 模拟")
        candidate = build_simulation_design_candidate("铝块内部空洞缺陷的区域 contrast 模拟").to_dict()

        material_ids = {item["id"] for item in pack["materials"]}
        geometry_ids = {item["id"] for item in pack["geometry"]}
        scoring_ids = {item["id"] for item in pack["scoring"]}
        self.assertIn("G4_Al", material_ids)
        self.assertIn("void", geometry_ids)
        self.assertIn("region_contrast", scoring_ids)
        self.assertEqual(candidate["next_action"], "build_candidate_config")
        self.assertEqual(candidate["unsupported_capabilities"], [])
        self.assertTrue(candidate["capability_check"]["supported"])

    def test_capability_checker_accepts_region_contrast_observable(self) -> None:
        report = check_simulation_design_capability(
            {
                "recommended_setup": {"geometry": "single_box", "source": "beam"},
                "observables": ["region_contrast"],
                "simplifications": [],
                "unsupported_capabilities": [],
            }
        )

        self.assertTrue(report["supported"])
        self.assertEqual(report["unsupported_capabilities"], [])


class SimulationDesignWorkflowTest(unittest.TestCase):
    def test_process_turn_simulation_design_is_read_only(self) -> None:
        session_id = "simulation-design-read-only"
        reset_session(session_id)
        try:
            first = process_turn(
                {
                    "session_id": session_id,
                    "text": "10 mm copper box target; gamma point source 1 MeV; output json.",
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                },
                ollama_config_path="",
                lang="en",
            )
            before_config = first["config"]
            design = process_turn(
                {
                    "session_id": session_id,
                    "text": "Can you design a steel pipe corrosion simulation approach?",
                    "enable_simulation_design": True,
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": False,
                },
                ollama_config_path="",
                lang="en",
            )

            self.assertEqual(design["action_safety_class"], "read_only")
            self.assertEqual(design["dialogue_action"], "simulation_design")
            self.assertEqual(design["config"], before_config)
            if design["simulation_design"]["next_action"] == "build_candidate_config":
                self.assertIn("geometry", design["recommended_config"])
                self.assertEqual(design["recommended_config"]["physics"]["physics_list"], "FTFP_BERT")
            self.assertEqual(design["audit_size"], first["audit_size"])
            self.assertEqual(design["nlu_turn_trace"]["tool_calls_allowed"], [])
            self.assertIn("run_beam", design["nlu_turn_trace"]["tool_calls_blocked"])
            self.assertIn(design["simulation_design"]["next_action"], ALLOWED_NEXT_ACTIONS)
        finally:
            reset_session(session_id)

    def test_web_api_simulation_design_is_read_only_and_does_not_create_step_job(self) -> None:
        session_id = "simulation-design-web-api"
        reset_session(session_id)
        try:
            called = {"step": False}

            def fail_step(_payload: dict, **_kwargs) -> dict:
                called["step"] = True
                raise AssertionError("simulation design must not create step workflow")

            status, body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "Design a steel pipe corrosion transmission simulation.",
                    "lang": "en",
                    "llm_router": False,
                },
                legacy_sessions={},
                solve_fn=lambda payload: {"unexpected": "solve"},
                step_fn=fail_step,
            )

            self.assertEqual(status, 200)
            self.assertFalse(called["step"])
            self.assertEqual(body["dialogue_action"], "simulation_design")
            self.assertEqual(body["action_safety_class"], "read_only")
            self.assertEqual(body["simulation_design"]["next_action"], "ask_user_to_choose_approximation")
            self.assertEqual(body["recommended_config"], {})
            self.assertEqual(body["nlu_turn_trace"]["tool_calls_allowed"], [])
            self.assertIn("run_beam", body["nlu_turn_trace"]["tool_calls_blocked"])
        finally:
            reset_session(session_id)

    def test_web_api_simulation_design_uses_llm_when_enabled(self) -> None:
        session_id = "simulation-design-web-api-llm"
        reset_session(session_id)
        llm_payload = {
            "schema_version": "geant4_agent_simulation_design_candidate.v1",
            "goal": "Design a 1 MeV gamma transmission study through lead shielding.",
            "recommended_setup": {
                "geometry": "single_box",
                "material": "G4_Pb",
                "source": "beam",
                "detector": {"enabled": True, "material": "G4_Si"},
                "scoring": ["detector_crossing_count", "detector_edep", "transmission_factor"],
            },
            "observables": ["detector_crossing_count", "detector_edep", "transmission_factor"],
            "assumptions": [],
            "simplifications": [],
            "unsupported_capabilities": [],
            "user_decisions_required": [],
            "knowledge_references": ["materials:G4_Pb", "scoring:detector_crossing_count"],
            "capability_check": {},
            "next_action": "build_candidate_config",
        }
        try:
            with mock.patch(
                "core.agent.simulation_design_llm.ollama_client.chat",
                return_value={"response": json.dumps(llm_payload)},
            ):
                status, body = handle_post_request(
                    "/api/simulation/design",
                    {
                        "session_id": session_id,
                        "text": "Design a 1 MeV gamma transmission study through lead shielding.",
                        "lang": "en",
                        "llm_router": True,
                    },
                    legacy_sessions={},
                    solve_fn=lambda payload: {"unexpected": "solve"},
                    step_fn=lambda payload: {"unexpected": "step"},
                )

            self.assertEqual(status, 200)
            self.assertEqual(body["dialogue_action"], "simulation_design")
            self.assertEqual(body["simulation_design_source"], "llm")
            self.assertTrue(body["simulation_design_llm"]["used"])
            self.assertEqual(body["nlu_turn_trace"]["llm_used"], True)
            self.assertEqual(body["simulation_design"]["next_action"], "build_candidate_config")
            self.assertEqual(body["simulation_design"]["recommended_setup"]["material"], "G4_Pb")
            self.assertEqual(body["recommended_config"]["geometry"]["params"]["module_x"], 10.0)
            self.assertEqual(body["recommended_config"]["materials"]["selected_materials"], ["G4_Pb"])
            self.assertEqual(body["recommended_config"]["source"]["energy"], 1.0)
            self.assertEqual(body["recommended_config"]["physics"]["physics_list"], "FTFP_BERT")
        finally:
            reset_session(session_id)

    def test_web_api_simulation_design_uses_previous_design_context(self) -> None:
        session_id = "simulation-design-memory"
        reset_session(session_id)
        try:
            first_status, first_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV along +z; output json",
                    "lang": "en",
                    "llm_router": False,
                },
                legacy_sessions={},
                solve_fn=lambda payload: {"unexpected": "solve"},
                step_fn=lambda payload: {"unexpected": "step"},
            )
            second_status, second_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "Change the target material to lead, keep the rest.",
                    "lang": "en",
                    "llm_router": False,
                },
                legacy_sessions={},
                solve_fn=lambda payload: {"unexpected": "solve"},
                step_fn=lambda payload: {"unexpected": "step"},
            )

            self.assertEqual(first_status, 200)
            self.assertEqual(second_status, 200)
            self.assertEqual(second_body["recommended_config"]["materials"]["selected_materials"], ["G4_Pb"])
            self.assertEqual(second_body["recommended_config"]["geometry"]["params"]["module_x"], 10.0)
            self.assertEqual(second_body["recommended_config"]["geometry"]["params"]["module_y"], 20.0)
            self.assertEqual(second_body["recommended_config"]["geometry"]["params"]["module_z"], 30.0)
        finally:
            reset_session(session_id)

    def test_web_api_accept_candidate_commits_design_to_runtime_session(self) -> None:
        session_id = "simulation-design-accept-runtime"
        reset_session(session_id)
        common = {"legacy_sessions": {}, "solve_fn": lambda payload: {}, "step_fn": lambda payload: {}}
        try:
            design_status, design_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": (
                        "10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV "
                        "at (0,0,-20) mm along +z; physics FTFP_BERT; output json"
                    ),
                    "lang": "en",
                    "llm_router": False,
                },
                **common,
            )
            accept_status, accept_body = handle_post_request(
                "/api/simulation/accept",
                {"session_id": session_id},
                **common,
            )
            summary_status, summary_body = handle_post_request(
                "/api/config/summary",
                {"session_id": session_id, "lang": "en"},
                **common,
            )
            validate_status, validate_body = handle_post_request(
                "/api/geant4/validate",
                {"session_id": session_id, "events": 1},
                **common,
            )
            run_status, run_body = handle_post_request(
                "/api/geant4/run",
                {"session_id": session_id, "events": 1, "action_id": "simulation-design-accept-runtime-run"},
                **common,
            )

            self.assertEqual(design_status, 200)
            self.assertTrue(design_body["recommended_config"])
            self.assertEqual(design_body["candidate_status"]["status"], "proposed")
            self.assertEqual(accept_status, 200)
            self.assertTrue(accept_body["ok"])
            self.assertEqual(accept_body["candidate_status"]["status"], "committed")
            self.assertIn("geometry", accept_body["committed_paths"])
            self.assertIn("source", accept_body["committed_paths"])
            self.assertEqual(summary_status, 200)
            self.assertEqual(summary_body["config"]["geometry"]["structure"], "single_box")
            self.assertEqual(summary_body["config"]["source"]["particle"], "gamma")
            self.assertEqual(validate_status, 200)
            self.assertTrue(validate_body["payload"]["ok"])
            self.assertEqual(validate_body["payload"]["missing_paths"], [])
            self.assertEqual(run_status, 200)
            self.assertEqual(run_body["runtime_smoke_report"]["events_completed"], 1)
        finally:
            reset_session(session_id)

    def test_web_api_vacuum_design_commits_galactic_and_passes_runtime_preflight(self) -> None:
        session_id = "simulation-design-vacuum-runtime"
        reset_session(session_id)
        common = {"legacy_sessions": {}, "solve_fn": lambda payload: {}, "step_fn": lambda payload: {}}
        try:
            design_status, design_body = handle_post_request(
                "/api/simulation/design",
                {
                    "session_id": session_id,
                    "text": "真空环境中的 1 MeV gamma 点源传输模拟，输出 json",
                    "lang": "zh",
                    "llm_router": False,
                },
                **common,
            )
            accept_status, accept_body = handle_post_request(
                "/api/simulation/accept",
                {"session_id": session_id},
                **common,
            )
            validate_status, validate_body = handle_post_request(
                "/api/geant4/validate",
                {"session_id": session_id, "events": 1},
                **common,
            )

            self.assertEqual(design_status, 200)
            self.assertEqual(design_body["simulation_design"]["recommended_setup"]["material"], "G4_Galactic")
            self.assertEqual(design_body["recommended_config"]["materials"]["selected_materials"], ["G4_Galactic"])
            self.assertEqual(accept_status, 200)
            self.assertTrue(accept_body["ok"])
            self.assertEqual(validate_status, 200)
            self.assertTrue(validate_body["payload"]["ok"])
            self.assertEqual(validate_body["payload"]["missing_paths"], [])
        finally:
            reset_session(session_id)


if __name__ == "__main__":
    unittest.main()

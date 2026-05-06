from __future__ import annotations

import unittest

from core.agent.context_pack import build_context_pack, capability_kb, unsupported_kb
from core.agent.intent_router import route_user_turn
from core.orchestrator.session_manager import process_turn, reset_session


class AgentContextPackTest(unittest.TestCase):
    def test_capability_kb_separates_supported_and_unsupported_capabilities(self) -> None:
        supported = capability_kb()
        unsupported = unsupported_kb()

        self.assertIn("single_box", supported["supported_geometry"])
        self.assertIn("single_tubs", supported["supported_geometry"])
        self.assertIn("ct_scanner_from_free_text", unsupported["unsupported_geometry"])
        self.assertNotIn("ct_scanner_from_free_text", supported["supported_geometry"])
        self.assertIn("implicit_chat_run", unsupported["unsupported_runtime_actions"])

    def test_context_pack_for_config_mutation_retrieves_capability_and_unsupported_snippets(self) -> None:
        decision = route_user_turn("Create a water phantom with a proton beam", "en")
        pack = build_context_pack(
            user_turn="Create a water phantom with a proton beam",
            intent_decision=decision,
            config={
                "geometry": {"structure": "single_box"},
                "materials": {"selected_materials": ["G4_WATER"]},
                "source": {"type": "beam", "particle": "proton"},
                "physics": {"physics_list": "FTFP_BERT"},
            },
        )

        self.assertEqual(pack.intent, "config_mutation")
        self.assertTrue(pack.context_pack_hash)
        self.assertEqual(pack.stable_slots["geometry_structure"], "single_box")
        source_types = {snippet.source_type for snippet in pack.retrieved_knowledge}
        self.assertIn("capability", source_types)
        self.assertIn("unsupported", source_types)
        self.assertIn("geometry.", pack.allowed_config_paths)

    def test_context_pack_for_normal_chat_does_not_retrieve_geant4_kb(self) -> None:
        decision = route_user_turn("hello there", "en")
        pack = build_context_pack(user_turn="hello there", intent_decision=decision, config={})

        self.assertEqual(pack.intent, "normal_chat")
        self.assertEqual(pack.retrieved_knowledge, [])

    def test_process_turn_exposes_context_pack_without_changing_parser_behavior(self) -> None:
        session_id = "agent-context-pack-process-turn"
        reset_session(session_id)
        out = process_turn(
            {
                "session_id": session_id,
                "text": (
                    "10 mm x 20 mm x 30 mm copper box target; "
                    "gamma point source 1 MeV at (0,0,-20) mm along +z; "
                    "physics FTFP_BERT; output json."
                ),
                "llm_router": False,
                "llm_question": False,
                "normalize_input": True,
                "geometry_pipeline": "v2",
                "source_pipeline": "v2",
                "enable_compare": False,
            },
            ollama_config_path="",
            lang="en",
        )
        try:
            self.assertTrue(out["is_complete"])
            pack = out["context_pack"]
            trace = out["nlu_turn_trace"]
            self.assertEqual(pack["intent"], "config_mutation")
            self.assertEqual(trace["context_pack_hash"], pack["context_pack_hash"])
            self.assertIn("unsupported_geometry", pack["unsupported_capabilities"])
            self.assertEqual(out["internal_trace"]["agent"]["context_pack"], pack)
        finally:
            reset_session(session_id)


if __name__ == "__main__":
    unittest.main()

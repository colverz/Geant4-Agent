from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.agent_v3.dialogue_composer import compose_v3_dialogue
from core.agent_v3.service import V3AgentTurnService


class V3DialogueComposerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._sessions_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_service(self) -> V3AgentTurnService:
        return V3AgentTurnService(sessions_dir=self._sessions_dir)

    def test_waiting_confirmation_has_human_display_message_and_raw_message(self) -> None:
        service = self._make_service()
        result = service.run_turn(
            {
                "session_id": "dialogue-confirm",
                "text": "run a default lead shielding gamma simulation",
                "events": 5,
                "allow_in_memory": True,
            }
        )

        self.assertEqual(result["dialogue_act"], "action_needs_confirmation")
        self.assertIn("确认运行", result["display_message"])
        self.assertIn("5 events", result["display_message"])
        self.assertEqual(result["raw_message"], result["answer"]["message"])
        self.assertEqual(result["answer"]["display_message"], result["display_message"])
        self.assertNotEqual(result["display_message"], result["raw_message"])
        self.assertIsInstance(result["dialogue"]["next_suggestions"][0], dict)
        self.assertIn("prefill", result["dialogue"]["next_suggestions"][0])
        self.assertIn("answer_parts", result)
        self.assertEqual(result["answer_parts"][0]["kind"], "summary")
        self.assertTrue(any(part["kind"] == "evidence" for part in result["answer_parts"]))
        self.assertTrue(any(part["kind"] == "next_step" for part in result["answer_parts"]))
        self.assertEqual(result["answer"]["answer_parts"], result["answer_parts"])

    def test_cancellation_turn_acknowledges_cancelled_runtime_action(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "dialogue-cancel",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        result = service.run_turn(
            {
                "session_id": "dialogue-cancel",
                "text": "cancel run, keep the design",
                "allow_in_memory": True,
            }
        )

        self.assertEqual(result["dialogue_act"], "action_cancelled")
        self.assertIn("已取消运行", result["display_message"])
        self.assertIn({"source": "pending_action", "status": "cancelled"}, result["evidence_used"])
        self.assertIsNone(result["pending_action"])
        self.assertIn("cancelled_pending_action", result)

    def test_design_only_turn_speaks_as_design_collaborator(self) -> None:
        service = self._make_service()
        result = service.run_turn(
            {
                "session_id": "dialogue-design",
                "text": "design a 1 MeV gamma lead shielding setup, do not run",
                "events": 3,
            }
        )

        self.assertEqual(result["dialogue_act"], "design_presented")
        self.assertIn("Geant4 方案", result["display_message"])
        self.assertIn("还没有开始运行", result["display_message"])
        self.assertTrue(result["evidence_used"])
        kinds = {part["kind"] for part in result["answer_parts"]}
        self.assertIn("summary", kinds)
        self.assertIn("evidence", kinds)
        self.assertIn("next_step", kinds)

    def test_design_dialogue_cleans_repeated_assumption_punctuation(self) -> None:
        response = {
            "terminated_reason": "final_answer",
            "answer": {"message": "raw"},
            "observations": [
                {
                    "source": "geant4_llm_design_tool",
                    "status": "ok",
                    "data": {
                        "design": {
                            "goal": "gamma shielding",
                            "recommended_setup": {
                                "geometry": "single_box",
                                "material": "G4_Pb",
                                "source": "beam",
                                "user_explanation": "铅适合先做屏蔽基线。",
                            },
                            "observables": ["detector_crossing_count"],
                            "assumptions": ["厚度先用 50 mm。", "探测器放在下游。"],
                        }
                    },
                }
            ],
        }

        dialogue = compose_v3_dialogue(response)

        self.assertNotIn("。。", dialogue.display_message)
        self.assertNotIn("。；", dialogue.display_message)
        self.assertIn("铅适合先做屏蔽基线。", dialogue.display_message)

    def test_waiting_user_with_design_shows_design_summary_not_generic_question(self) -> None:
        response = {
            "terminated_reason": "waiting_user",
            "answer": {
                "message": "I have drafted a Geant4 design. Accept defaults and generate config?",
                "next_options": ["accept current design and generate payload", "modify material or energy"],
            },
            "observations": [],
            "state": {
                "observations": [
                    {
                        "source": "geant4_llm_design_tool",
                        "status": "ok",
                        "data": {
                            "design": {
                                "goal": "100 MeV proton in G4_WATER",
                                "recommended_setup": {
                                    "geometry": "single_box",
                                    "material": "G4_WATER",
                                    "source": "beam",
                                    "source_particle": "proton",
                                    "source_energy_mev": 100.0,
                                    "design_rationale": "Water is tissue-equivalent for this proton setup.",
                                },
                                "observables": ["target_edep", "depth_bins"],
                                "assumptions": ["Use FTFP_BERT unless the user specifies otherwise."],
                            }
                        },
                    }
                ]
            },
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "needs_user_input")
        self.assertIn("G4_WATER", dialogue.display_message)
        self.assertIn("proton", dialogue.display_message)
        self.assertIn("100 MeV", dialogue.display_message)
        self.assertIn("no payload has been generated", dialogue.display_message)
        self.assertEqual(dialogue.next_suggestions[0]["prefill"], "accept current design and generate payload")

    def test_invalid_patch_waiting_user_keeps_patch_error_even_with_existing_design(self) -> None:
        response = {
            "terminated_reason": "waiting_user",
            "answer": {
                "message": "I could not apply that change: `run_confirmed` is not editable.",
                "evidence": [{"source": "state_patch", "status": "failed"}],
                "next_options": ["confirm run"],
            },
            "observations": [],
            "state": {
                "metadata": {
                    "last_state_patch": {
                        "ok": False,
                        "errors": ["unsupported_patch_field:run_confirmed"],
                    }
                },
                "observations": [
                    {
                        "source": "geant4_design_template_tool",
                        "status": "ok",
                        "data": {
                            "design": {
                                "recommended_setup": {"geometry": "single_box", "material": "G4_Pb", "source": "beam"},
                                "observables": ["target_edep"],
                            }
                        },
                    }
                ],
            },
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "needs_user_input")
        self.assertIn("run_confirmed", dialogue.display_message)
        self.assertNotIn("Here is the simulation design", dialogue.display_message)

    def test_payload_state_without_current_observation_is_not_labeled_result_answer(self) -> None:
        response = {
            "terminated_reason": "final_answer",
            "answer": {
                "message": "已按默认假设把方案推进为 Geant4 runtime payload 草案。"
                "当前还没有执行 Geant4，下一步需要确认运行。"
            },
            "observations": [],
            "state": {
                "observations": [
                    {
                        "source": "geant4_payload_builder_tool",
                        "status": "ok",
                        "data": {
                            "simulation_spec": {
                                "source": {"energy_mev": 1.0},
                                "geometry": {"material": "G4_Pb"},
                                "run": {"events": 10},
                            },
                            "runtime_payload": {"schema_version": "runtime_dsl.v1"},
                        },
                    }
                ]
            },
        }

        dialogue = compose_v3_dialogue(response)

        self.assertEqual(dialogue.dialogue_act, "payload_draft_presented")

    def test_no_runtime_result_answer_has_next_step(self) -> None:
        response = {
            "terminated_reason": "final_answer",
            "answer": {"message": "No runtime result is available yet."},
            "observations": [],
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "final_answer")
        self.assertTrue(dialogue.next_suggestions)
        self.assertTrue(any(part["kind"] == "next_step" for part in dialogue.to_dict()["answer_parts"]))

    def test_state_observations_count_as_dialogue_evidence(self) -> None:
        response = {
            "terminated_reason": "final_answer",
            "answer": {"message": "Explain current result."},
            "observations": [],
            "state": {
                "observations": [
                    {"source": "geant4_runtime_tool", "status": "ok", "data": {"result_summary": {}}},
                ]
            },
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.evidence_used, [{"source": "geant4_runtime_tool", "status": "ok"}])

    def test_current_configuration_answer_is_not_rewritten_as_design_prompt(self) -> None:
        response = {
            "terminated_reason": "final_answer",
            "answer": {
                "message": "Current design draft: geometry=single_box, material=G4_Pb, source=beam, particle=TBD, observables=target_edep. No runtime payload has been generated yet."
            },
            "observations": [],
            "state": {
                "observations": [
                    {
                        "source": "geant4_design_template_tool",
                        "status": "ok",
                        "data": {
                            "design": {
                                "recommended_setup": {"geometry": "single_box", "material": "G4_Pb", "source": "beam"},
                                "observables": ["target_edep"],
                            }
                        },
                    }
                ]
            },
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "configuration_answered")
        self.assertIn("Current design draft", dialogue.display_message)
        self.assertNotIn("Want me to adjust anything", dialogue.display_message)

    def test_blocked_runtime_without_payload_gives_actionable_recovery(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "answer": {"message": "blocked"},
            "observations": [
                {
                    "source": "proposal_critic",
                    "status": "blocked",
                    "data": {"reason": "runtime_without_payload"},
                    "message": "Runtime execution needs a drafted payload before Geant4 can run.",
                }
            ],
        }

        dialogue = compose_v3_dialogue(response)

        self.assertEqual(dialogue.dialogue_act, "blocked")
        self.assertIn("payload", dialogue.display_message)
        self.assertIsInstance(dialogue.next_suggestions[0], dict)
        self.assertEqual(dialogue.next_suggestions[0]["text"], "生成运行配置")
        self.assertIn("runtime payload", dialogue.next_suggestions[0]["prefill"])

    def test_blocked_runtime_without_preflight_is_not_trace_only(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "answer": {"message": "blocked"},
            "observations": [
                {
                    "source": "proposal_critic",
                    "status": "blocked",
                    "data": {"reason": "runtime_without_preflight"},
                    "message": "Runtime execution needs a preflight observation before confirmation.",
                }
            ],
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "blocked")
        self.assertIn("preflight", dialogue.display_message.lower())
        self.assertEqual(dialogue.next_suggestions[0]["text"], "Run preflight again")

    def test_blocked_schema_invalid_has_repair_suggestion(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "answer": {"message": "blocked"},
            "observations": [
                {
                    "source": "proposal_critic",
                    "status": "blocked",
                    "data": {
                        "reason": "tool_schema_invalid",
                        "schema_errors": ["missing_required:payload_builder_observation"],
                        "repair_suggestions": ["Provide required field `payload_builder_observation` from current context."],
                    },
                }
            ],
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "blocked")
        self.assertIn("registered schema", dialogue.display_message)
        self.assertIn("Repair hint", dialogue.display_message)
        self.assertIn("prefill", dialogue.next_suggestions[0])

    def test_blocked_unknown_tool_guides_back_to_catalog(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "answer": {"message": "blocked"},
            "observations": [
                {
                    "source": "proposal_critic",
                    "status": "blocked",
                    "data": {"reason": "unknown_tool", "tool_name": "made_up_tool"},
                }
            ],
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "blocked")
        self.assertIn("registered v3 tool contract", dialogue.display_message)
        self.assertTrue(dialogue.next_suggestions)

    def test_blocked_ungrounded_context_guides_rebuild(self) -> None:
        response = {
            "terminated_reason": "blocked",
            "answer": {"message": "blocked"},
            "observations": [
                {
                    "source": "proposal_critic",
                    "status": "blocked",
                    "data": {
                        "reason": "context_fact_not_grounded",
                        "grounding_errors": ["ungrounded_payload_argument"],
                        "repair_suggestions": ["Use the latest payload observation from the current session."],
                    },
                }
            ],
        }

        dialogue = compose_v3_dialogue(response, locale="en-US")

        self.assertEqual(dialogue.dialogue_act, "blocked")
        self.assertIn("not grounded", dialogue.display_message)
        self.assertIn("rebuild", dialogue.display_message.lower())
        self.assertTrue(dialogue.next_suggestions)


if __name__ == "__main__":
    unittest.main()

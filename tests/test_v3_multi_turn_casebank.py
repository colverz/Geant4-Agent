from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.agent_v3.service import V3AgentTurnService


def _make_service_with_temp_dir() -> tuple[V3AgentTurnService, tempfile.TemporaryDirectory]:
    tmpdir = tempfile.TemporaryDirectory()
    service = V3AgentTurnService(sessions_dir=Path(tmpdir.name))
    return service, tmpdir


class MultiTurnCasebankTest(unittest.TestCase):
    """Multi-turn end-to-end behavior tests for the v3 agent main chain.

    These tests verify that the agent maintains correct state across multiple
    conversation turns: design, modify, confirm, cancel, run, and explain.
    """

    # ── Case 1: design-only (no run) ──────────────────────────────────

    def test_design_only_then_ask_then_confirm(self) -> None:
        """User: "design a shielding setup, don't run" → design presented.
        Then: "accept defaults and generate payload" → payload drafted.
        Then: "confirm run" → waiting for confirmation."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            # Turn 1: design only
            r1 = service.run_turn({
                "session_id": "case-01",
                "text": "设计一个铅屏蔽 gamma 透射方案，不要运行",
            })
            self.assertTrue(r1["ok"])
            self.assertIn("design_presented", r1.get("dialogue_act", ""))

            # Turn 2: accept defaults (triggers payload generation)
            r2 = service.run_turn({
                "session_id": "case-01",
                "text": "接受默认近似并生成配置",
                "accept_defaults": True,
                "allow_in_memory": True,
            })
            self.assertTrue(r2["ok"])
            self.assertIn(r2.get("dialogue_act", ""), ("payload_draft_presented", "design_presented"))
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 2: modify parameters and run ─────────────────────────────

    def test_design_modify_confirm_run(self) -> None:
        """User designs, then modifies energy and material, confirms run."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            # Turn 1: design + run request
            r1 = service.run_turn({
                "session_id": "case-02",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "accept_defaults": True,
                "run": True,
                "allow_in_memory": True,
            })
            self.assertTrue(r1["ok"])
            self.assertEqual(r1["terminated_reason"], "waiting_confirmation")

            # Turn 2: modify energy and material, then re-run
            r2 = service.run_turn({
                "session_id": "case-02",
                "text": "把能量改成 2 MeV，材料换成铜，再跑",
                "allow_in_memory": True,
            })
            self.assertTrue(r2["ok"])
            self.assertEqual(r2["terminated_reason"], "waiting_confirmation")

            # Turn 3: confirm the modified run
            r3 = service.run_turn({
                "session_id": "case-02",
                "text": "确认运行",
            })
            self.assertTrue(r3["ok"])
            self.assertEqual(r3["terminated_reason"], "observed")
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 3: cancel pending run ────────────────────────────────────

    def test_design_run_cancel(self) -> None:
        """User requests run, then cancels."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            # Turn 1: request run
            r1 = service.run_turn({
                "session_id": "case-03",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "accept_defaults": True,
                "run": True,
                "allow_in_memory": True,
            })
            self.assertEqual(r1["terminated_reason"], "waiting_confirmation")

            # Turn 2: cancel
            r2 = service.run_turn({
                "session_id": "case-03",
                "text": "取消运行，只保留方案",
            })
            self.assertTrue(r2["ok"])
            self.assertEqual(r2.get("dialogue_act"), "action_cancelled")
            self.assertIsNone(r2.get("pending_action"))
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 4: result follow-up question ─────────────────────────────

    def test_result_followup_after_run(self) -> None:
        """User runs a simulation, then asks about the result."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            # Turn 1: design + run
            service.run_turn({
                "session_id": "case-04",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "accept_defaults": True,
                "run": True,
                "allow_in_memory": True,
            })
            # Turn 2: confirm
            r2 = service.run_turn({
                "session_id": "case-04",
                "text": "确认运行",
            })
            self.assertEqual(r2["terminated_reason"], "observed")

            # Turn 3: ask about result
            r3 = service.run_turn({
                "session_id": "case-04",
                "text": "target_edep 这个值说明了什么？",
            })
            self.assertTrue(r3["ok"])
            self.assertIn(
                r3.get("dialogue_act", ""),
                ("runtime_result_answered", "final_answer"),
            )
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 5: result question without prior run ─────────────────────

    def test_result_question_without_runtime(self) -> None:
        """User asks about a result before any simulation ran."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            r = service.run_turn({
                "session_id": "case-05",
                "text": "detector_crossing_count 是什么意思？",
            })
            self.assertTrue(r["ok"])
            self.assertEqual(r["terminated_reason"], "final_answer")
            # Should NOT start a new design just because user asked a question
            self.assertNotIn("design", r.get("dialogue_act", ""))
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 6: short confirmation ("确认") ────────────────────────────

    def test_short_confirmation_word(self) -> None:
        """A single word "确认" should confirm the pending action."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            service.run_turn({
                "session_id": "case-06",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "accept_defaults": True,
                "run": True,
                "allow_in_memory": True,
            })
            r2 = service.run_turn({
                "session_id": "case-06",
                "text": "确认",
            })
            self.assertTrue(r2["ok"])
            self.assertIn(r2["terminated_reason"], ("observed", "waiting_confirmation"))
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 7: session persistence across service instances ──────────

    def test_session_survives_new_service_instance(self) -> None:
        """State should persist to disk and be reloaded by a new service."""
        tmpdir = tempfile.TemporaryDirectory()
        try:
            sessions_dir = Path(tmpdir.name)
            sa = V3AgentTurnService(sessions_dir=sessions_dir)
            sa.run_turn({
                "session_id": "case-07",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "accept_defaults": True,
                "run": True,
                "allow_in_memory": True,
            })

            sb = V3AgentTurnService(sessions_dir=sessions_dir)
            r2 = sb.run_turn({
                "session_id": "case-07",
                "text": "确认运行",
            })
            self.assertEqual(r2["terminated_reason"], "observed")
        finally:
            try:
                sa.reset()
                sb.reset()
            except Exception:
                pass
            tmpdir.cleanup()

    # ── Case 8: LLM intent classifier falls back gracefully ────────────

    # ── Case 9: preflight rejects in-memory without allow flag ─────────

    def test_preflight_rejects_in_memory_by_default(self) -> None:
        """Without allow_in_memory, preflight returns not_evaluable.
        The agent presents the payload draft and informs the user
        that a real runtime is required."""
        service, tmpdir = _make_service_with_temp_dir()
        try:
            r1 = service.run_turn({
                "session_id": "case-09",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "accept_defaults": True,
                "run": True,
            })
            self.assertTrue(r1["ok"])
            # Without allow_in_memory, the flow stops at payload or preflight stage
            self.assertEqual(r1["terminated_reason"], "final_answer")
        finally:
            service.reset()
            tmpdir.cleanup()

    # ── Case 10: LLM reasoner fallback ────────────────────────────────

    def test_llm_reasoner_fallback_preserves_chain(self) -> None:
        """When LLM reasoner fails, fallback to BasicGeant4Reasoner
        and the multi-turn chain still works."""
        from core.agent_v3.controller import AgentController
        from core.agent_v3.reasoners import LLMGeant4Reasoner
        from core.agent_v3.tools import build_default_geant4_tool_registry

        reasoner = LLMGeant4Reasoner(llm_config_path="")  # empty → fallback
        controller = AgentController(
            reasoner=reasoner,
            tools=build_default_geant4_tool_registry(),
        )
        from core.agent_v3.contracts import V3TurnInput

        # Should run through the fallback (BasicGeant4Reasoner)
        result = controller.run(V3TurnInput(
            session_id="case-09",
            user_text="设计一个铅屏蔽方案",
        ))
        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertTrue(len(result.trace) >= 2)

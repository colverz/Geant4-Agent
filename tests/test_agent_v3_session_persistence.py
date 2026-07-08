from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.agent_v3.contracts import V3AgentState, V3Observation, V3ObservationStatus
from core.agent_v3.service import V3AgentTurnService
from core.agent_v3.session import V3_SESSION_SCHEMA_VERSION


class SessionPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self._tmpdir.name)
        self.service = V3AgentTurnService(sessions_dir=self.sessions_dir)

    def tearDown(self) -> None:
        self.service.reset()
        self._tmpdir.cleanup()

    def test_saves_state_to_disk_after_turn(self) -> None:
        result = self.service.run_turn({
            "session_id": "sess-001",
            "text": "设计铅屏蔽方案",
        })

        state_path = self.service._state_path("sess-001")
        self.assertTrue(state_path.exists())
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["schema_version"], V3_SESSION_SCHEMA_VERSION)
        self.assertEqual(saved["session_id"], "sess-001")
        self.assertIn("updated_at", saved)
        self.assertIn("last_turn_id", saved)
        self.assertIsInstance(saved["state"], dict)

    def test_loads_state_from_disk_on_next_turn(self) -> None:
        result1 = self.service.run_turn({
            "session_id": "sess-002",
            "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
            "accept_defaults": True,
            "run": True,
            "allow_in_memory": True,
        })
        self.assertEqual(result1.get("terminated_reason"), "waiting_confirmation")
        result2 = self.service.run_turn({
            "session_id": "sess-002",
            "text": "确认运行",
        })

        self.assertEqual(result2.get("terminated_reason"), "observed")

    def test_reset_deletes_session_file(self) -> None:
        self.service.run_turn({
            "session_id": "sess-003",
            "text": "设计铅屏蔽方案",
        })
        state_path = self.service._state_path("sess-003")
        self.assertTrue(state_path.exists())

        self.service.reset("sess-003")
        self.assertFalse(state_path.exists())

    def test_reset_all_clears_sessions_dir(self) -> None:
        self.service.run_turn({"session_id": "sess-a", "text": "hello"})
        self.service.run_turn({"session_id": "sess-b", "text": "hello"})

        self.service.reset()
        self.assertEqual(len(list(self.sessions_dir.glob("*.json"))), 0)

    def test_loads_state_from_disk_on_fresh_service_instance(self) -> None:
        service_a = V3AgentTurnService(sessions_dir=self.sessions_dir)
        service_a.run_turn({
            "session_id": "sess-004",
            "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
            "accept_defaults": True,
            "run": True,
            "allow_in_memory": True,
        })

        service_b = V3AgentTurnService(sessions_dir=self.sessions_dir)
        result = service_b.run_turn({
            "session_id": "sess-004",
            "text": "确认运行",
        })

        self.assertEqual(result.get("terminated_reason"), "observed")

    def test_corrupted_disk_file_is_gracefully_ignored(self) -> None:
        state_path = self.service._state_path("corrupt-sess")
        state_path.write_text("not valid json", encoding="utf-8")

        result = self.service.run_turn({
            "session_id": "corrupt-sess",
            "text": "设计铅屏蔽方案",
        })

        self.assertIn(result.get("terminated_reason"), ("final_answer", "waiting_confirmation"))
        self.assertTrue(list(self.sessions_dir.glob("corrupt-sess.json.corrupt*")))
        self.assertTrue(state_path.exists())

    def test_loads_legacy_bare_state_file(self) -> None:
        state_path = self.service._state_path("legacy-sess")
        state_path.write_text(
            json.dumps({"session_id": "legacy-sess", "goal": "legacy goal", "observations": []}),
            encoding="utf-8",
        )

        result = self.service.run_turn({
            "session_id": "legacy-sess",
            "text": "hello",
        })

        self.assertTrue(result["ok"])
        self.assertEqual(result["state"]["session_id"], "legacy-sess")

    def test_state_from_dict_roundtrip_preserves_observations(self) -> None:
        state = V3AgentState(session_id="test-rtt")
        state.goal = "lead shielding"
        state.add_observation(V3Observation(source="tool_a", status=V3ObservationStatus.OK, message="done"))
        state.metadata["pending_action"] = {"kind": "run_simulation"}
        state.artifacts["design"] = {"material": "G4_Pb"}

        d = state.to_dict()
        restored = V3AgentState.from_dict(d)

        self.assertEqual(restored.session_id, "test-rtt")
        self.assertEqual(restored.goal, "lead shielding")
        self.assertEqual(len(restored.observations), 1)
        self.assertEqual(restored.observations[0].source, "tool_a")
        self.assertEqual(restored.observations[0].status, V3ObservationStatus.OK)
        self.assertEqual(restored.metadata["pending_action"], {"kind": "run_simulation"})

    def test_state_from_dict_handles_partial_data(self) -> None:
        restored = V3AgentState.from_dict({"session_id": "minimal"})
        self.assertEqual(restored.session_id, "minimal")
        self.assertEqual(restored.goal, "")
        self.assertEqual(restored.observations, [])
        self.assertEqual(restored.metadata, {})

    def test_unsafe_session_ids_have_collision_resistant_paths(self) -> None:
        slash_path = self.service._state_path("client/a")
        underscore_path = self.service._state_path("client_a")

        self.assertNotEqual(slash_path, underscore_path)
        self.service.run_turn({"session_id": "client/a", "text": "hello"})
        self.service.run_turn({"session_id": "client_a", "text": "hello"})

        self.assertTrue(slash_path.exists())
        self.assertTrue(underscore_path.exists())
        self.assertEqual(self.service._load_state("client/a").session_id, "client/a")
        self.assertEqual(self.service._load_state("client_a").session_id, "client_a")

    def test_session_store_rejects_mismatched_state_identity(self) -> None:
        state_path = self.service._state_path("expected-session")
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": V3_SESSION_SCHEMA_VERSION,
                    "session_id": "other-session",
                    "state": {"session_id": "other-session", "observations": []},
                }
            ),
            encoding="utf-8",
        )

        self.assertIsNone(self.service._load_state("expected-session"))
        self.assertTrue(list(self.sessions_dir.glob("expected-session.json.corrupt*")))

from __future__ import annotations

import os
import unittest

from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import InMemoryGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.server import Geant4McpServer


def _live_smoke_enabled() -> bool:
    return str(os.getenv("GEANT4_LIVE_SMOKE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_command_configured() -> bool:
    return bool(os.getenv("GEANT4_RUNTIME_COMMAND_JSON") or os.getenv("GEANT4_RUNTIME_COMMAND"))


def _smoke_patch() -> dict:
    return {
        "geometry": {
            "structure": "single_box",
            "root_name": "Target",
            "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
        },
        "materials": {
            "selected_materials": ["G4_Cu"],
            "volume_material_map": {"Target": "G4_Cu"},
        },
        "source": {
            "type": "point",
            "particle": "gamma",
            "energy": 1.0,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "physics_list": {"name": "FTFP_BERT"},
        "scoring": {"target_edep": True},
    }


@unittest.skipUnless(
    _live_smoke_enabled() and _runtime_command_configured(),
    "Set GEANT4_LIVE_SMOKE=1 and GEANT4_RUNTIME_COMMAND_JSON or GEANT4_RUNTIME_COMMAND to run live Geant4 smoke.",
)
class Geant4LiveSmokeTest(unittest.TestCase):
    def test_local_geant4_runtime_executes_and_returns_summary(self) -> None:
        adapter = build_geant4_adapter_from_env()
        self.assertNotIsInstance(adapter, InMemoryGeant4Adapter)

        server = Geant4McpServer(adapter=adapter)
        apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": _smoke_patch()}))
        self.assertEqual(apply_obs.status, RuntimeActionStatus.COMPLETED)
        runtime_payload = apply_obs.payload["runtime_payload"]
        self.assertEqual(runtime_payload["structure"], "single_box")
        self.assertEqual(runtime_payload["source_type"], "point")
        self.assertEqual(runtime_payload["particle"], "gamma")

        init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
        self.assertEqual(init_obs.status, RuntimeActionStatus.COMPLETED)

        run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": 1}))
        self.assertEqual(run_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertIsInstance(run_obs.payload.get("result_summary"), dict)
        result = run_obs.payload.get("simulation_result")
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertEqual(result["geometry_structure"], "single_box")
        self.assertEqual(result["source_type"], "point")
        self.assertEqual(result["particle"], "gamma")
        self.assertIn("run_summary_path", result)
        self.assertIn("artifact_dir", result)
        summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
        self.assertEqual(summary_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(summary_obs.payload["result_summary"]["run"]["events_completed"], 1)
        self.assertEqual(summary_obs.payload["result_summary"]["configuration"]["geometry_structure"], "single_box")


if __name__ == "__main__":
    unittest.main()

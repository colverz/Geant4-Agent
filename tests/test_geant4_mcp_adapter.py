from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from core.runtime.types import Geant4RuntimePhase, RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import LocalProcessGeant4Adapter
from mcp.geant4.server import Geant4McpServer


class Geant4McpAdapterTest(unittest.TestCase):
    def test_server_lists_minimal_tools(self) -> None:
        server = Geant4McpServer()
        names = [item["name"] for item in server.list_tools()]
        self.assertEqual(
            names,
            [
                "get_runtime_state",
                "apply_config_patch",
                "initialize_run",
                "run_beam",
                "get_last_log",
            ],
        )

    def test_run_requires_initialization(self) -> None:
        server = Geant4McpServer()
        obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": 10}))
        self.assertEqual(obs.status, RuntimeActionStatus.REJECTED)
        self.assertEqual(obs.runtime_phase, Geant4RuntimePhase.IDLE)

    def test_configure_initialize_and_run(self) -> None:
        server = Geant4McpServer()
        server.call_tool(
            ToolCallRequest(
                tool_name="apply_config_patch",
                arguments={
                    "patch": {
                        "geometry": {"structure": "single_box"},
                        "source": {"particle": "gamma"},
                        "physics_list": {"name": "FTFP_BERT"},
                    }
                },
            )
        )
        init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
        run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": 4}))
        self.assertEqual(init_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(run_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(run_obs.payload["events"], 4)

    def test_local_process_adapter_executes_command(self) -> None:
        adapter = LocalProcessGeant4Adapter(
            [
                sys.executable,
                "-c",
                "import sys; print('geant4 wrapper ok'); print('argv=' + ' '.join(sys.argv[1:]))",
            ]
        )
        server = Geant4McpServer(adapter=adapter)
        server.call_tool(
            ToolCallRequest(
                tool_name="apply_config_patch",
                arguments={
                    "patch": {
                        "geometry": {"structure": "single_box"},
                        "source": {"particle": "gamma"},
                        "physics_list": {"name": "FTFP_BERT"},
                    }
                },
            )
        )
        init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
        run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": 2}))
        log_obs = server.call_tool(ToolCallRequest(tool_name="get_last_log", arguments={}))
        self.assertEqual(init_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(run_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(run_obs.payload["returncode"], 0)
        self.assertIn("geant4 wrapper ok", "\n".join(log_obs.payload["lines"]))

    def test_local_process_adapter_loads_simulation_result_from_artifact_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            summary_path = artifact_dir / "run_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_ok": True,
                        "events_requested": 3,
                        "events_completed": 3,
                        "geometry_structure": "single_box",
                        "material": "G4_Cu",
                        "particle": "gamma",
                        "source_type": "point",
                        "source_position_mm": [0, 0, -20],
                        "source_direction": [0, 0, 1],
                        "physics_list": "FTFP_BERT",
                        "events": 3,
                        "mode": "batch",
                        "scoring": {
                            "target_edep_enabled": True,
                            "target_edep_total_mev": 1.5,
                            "target_edep_mean_mev_per_event": 0.5,
                            "target_hit_events": 2,
                            "target_step_count": 12,
                            "target_track_entries": 3,
                            "volume_stats": {
                                "Target": {
                                    "edep_total_mev": 1.5,
                                    "edep_mean_mev_per_event": 0.5,
                                    "hit_events": 2,
                                    "step_count": 12,
                                    "track_entries": 3,
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter = LocalProcessGeant4Adapter(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "import sys; "
                        "print('geant4 wrapper ok'); "
                        f"print('artifact_dir={artifact_dir.as_posix()}')"
                    ),
                ]
            )
            server = Geant4McpServer(adapter=adapter)
            server.call_tool(
                ToolCallRequest(
                    tool_name="apply_config_patch",
                    arguments={
                        "patch": {
                            "geometry": {"structure": "single_box"},
                            "source": {"particle": "gamma"},
                            "physics_list": {"name": "FTFP_BERT"},
                        }
                    },
                )
            )
            server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
            run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": 3}))
        self.assertEqual(run_obs.status, RuntimeActionStatus.COMPLETED)
        result = run_obs.payload["simulation_result"]
        self.assertEqual(result["geometry_structure"], "single_box")
        self.assertEqual(result["scoring"]["target_hit_events"], 2)
        self.assertEqual(result["scoring"]["volume_stats"]["Target"]["step_count"], 12)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from core.runtime.types import Geant4RuntimePhase, RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import (
    InMemoryGeant4Adapter,
    LocalProcessGeant4Adapter,
    _load_run_summary_payload,
    build_geant4_adapter_from_env,
)
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
                "validate_config",
                "initialize_run",
                "run_beam",
                "summarize_last_result",
                "get_last_log",
            ],
        )

    def test_summarize_last_result_requires_completed_run(self) -> None:
        server = Geant4McpServer()
        obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
        self.assertEqual(obs.status, RuntimeActionStatus.REJECTED)
        self.assertIn("no_result_summary_available", obs.errors)

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
        self.assertEqual(run_obs.payload["result_summary"]["run"]["events_completed"], 4)
        summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
        self.assertEqual(summary_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(summary_obs.payload["result_summary"]["configuration"]["particle"], "gamma")
        self.assertEqual(summary_obs.payload["result_summary"]["configuration"]["physics_list"], "FTFP_BERT")

    def test_validate_config_reports_missing_runtime_fields(self) -> None:
        server = Geant4McpServer()
        obs = server.call_tool(
            ToolCallRequest(
                tool_name="validate_config",
                arguments={
                    "config": {
                        "geometry": {"structure": "single_box", "params": {"module_x": 10.0}},
                    }
                },
            )
        )
        self.assertEqual(obs.status, RuntimeActionStatus.COMPLETED)
        self.assertFalse(obs.payload["ok"])
        self.assertIn("geometry.params.module_y", obs.payload["missing_paths"])
        self.assertIn("source.type", obs.payload["missing_paths"])
        self.assertIn("physics.physics_list", obs.payload["missing_paths"])
        self.assertFalse(obs.payload["readiness"]["source"])
        self.assertIn("runtime_payload_preview_omitted_until_required_fields_are_present", obs.warnings)

    def test_validate_config_returns_runtime_payload_preview_for_complete_config(self) -> None:
        server = Geant4McpServer()
        obs = server.call_tool(
            ToolCallRequest(
                tool_name="validate_config",
                arguments={
                    "events": 5,
                    "config": {
                        "geometry": {
                            "structure": "single_box",
                            "root_name": "Target",
                            "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
                        },
                        "materials": {"selected_materials": ["G4_Cu"]},
                        "source": {
                            "type": "beam",
                            "particle": "gamma",
                            "energy": 1.0,
                            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
                            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
                        },
                        "physics_list": {"name": "FTFP_BERT"},
                        "scoring": {"target_edep": True},
                    },
                },
            )
        )
        self.assertEqual(obs.status, RuntimeActionStatus.COMPLETED)
        self.assertTrue(obs.payload["ok"])
        self.assertEqual(obs.errors, [])
        self.assertEqual(obs.payload["readiness"], {"geometry": True, "source": True, "physics": True})
        preview = obs.payload["runtime_payload_preview"]
        self.assertEqual(preview["structure"], "single_box")
        self.assertEqual(preview["source_type"], "beam")
        self.assertEqual(preview["particle"], "gamma")
        self.assertEqual(preview["physics_list"], "FTFP_BERT")
        self.assertEqual(preview["run"]["events"], 5)
        self.assertIn("scoring", preview)

    def test_validate_config_can_preflight_current_config_plus_patch(self) -> None:
        server = Geant4McpServer()
        server.call_tool(
            ToolCallRequest(
                tool_name="apply_config_patch",
                arguments={
                    "patch": {
                        "geometry": {
                            "structure": "single_box",
                            "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
                        },
                        "source": {
                            "type": "point",
                            "particle": "gamma",
                            "energy": 1.0,
                            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
                        },
                    }
                },
            )
        )
        obs = server.call_tool(
            ToolCallRequest(
                tool_name="validate_config",
                arguments={"patch": {"physics": {"physics_list": "FTFP_BERT"}}},
            )
        )
        self.assertEqual(obs.status, RuntimeActionStatus.COMPLETED)
        self.assertTrue(obs.payload["ok"])
        self.assertEqual(obs.payload["runtime_payload_preview"]["source_type"], "point")
        self.assertEqual(obs.payload["runtime_payload_preview"]["physics_list"], "FTFP_BERT")

    def test_default_adapter_uses_in_memory_without_runtime_env(self) -> None:
        adapter = build_geant4_adapter_from_env({})
        self.assertIsInstance(adapter, InMemoryGeant4Adapter)

    def test_default_adapter_rejects_malformed_runtime_command_json(self) -> None:
        with self.assertRaises(ValueError):
            build_geant4_adapter_from_env({"GEANT4_RUNTIME_COMMAND_JSON": "[not-json"})

    def test_default_adapter_can_be_configured_from_runtime_env(self) -> None:
        adapter = build_geant4_adapter_from_env(
            {
                "GEANT4_RUNTIME_COMMAND_JSON": json.dumps([sys.executable, "-c", "print('ok')"]),
                "GEANT4_ROOT": "F:\\Geant4Test",
                "GEANT4_WORKING_DIR": "F:\\geant4agent",
            }
        )
        self.assertIsInstance(adapter, LocalProcessGeant4Adapter)

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

    def test_local_process_adapter_preview_preserves_config_run_events(self) -> None:
        adapter = LocalProcessGeant4Adapter([sys.executable, "-c", "print('ok')"])
        server = Geant4McpServer(adapter=adapter)

        apply_obs = server.call_tool(
            ToolCallRequest(
                tool_name="apply_config_patch",
                arguments={
                    "patch": {
                        "geometry": {"structure": "single_box"},
                        "source": {"particle": "gamma", "energy": 1.0},
                        "physics_list": {"name": "FTFP_BERT"},
                        "run": {"events": 12},
                    }
                },
            )
        )

        self.assertEqual(apply_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(apply_obs.payload["runtime_payload"]["run"]["events"], 12)

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
                                "target": {
                                    "edep_total_mev": 1.5,
                                    "edep_mean_mev_per_event": 0.5,
                                    "hit_events": 2,
                                    "step_count": 12,
                                    "track_entries": 3,
                                },
                                "Detector": {
                                    "edep_total_mev": 0.2,
                                    "edep_mean_mev_per_event": 0.0667,
                                    "hit_events": 1,
                                    "step_count": 4,
                                    "track_entries": 1,
                                }
                            },
                        },
                        "detector": {
                            "enabled": True,
                            "volume_name": "Detector",
                            "material": "G4_Si",
                            "position_mm": [0, 0, 50],
                            "size_mm": [12, 12, 1.5],
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
                            "geometry": {"structure": "single_box", "root_name": "target"},
                            "simulation": {
                                "detector": {
                                    "enabled": True,
                                    "name": "Detector",
                                    "material": "G4_Si",
                                    "position": {"type": "vector", "value": [0.0, 0.0, 50.0]},
                                    "size_triplet_mm": [12.0, 12.0, 1.5],
                                }
                            },
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
        self.assertEqual(result["scoring"]["volume_stats"]["target"]["step_count"], 12)
        self.assertEqual(result["scoring"]["role_stats"]["target"]["track_entries"], 3)
        self.assertEqual(result["scoring"]["role_stats"]["detector"]["track_entries"], 1)
        self.assertEqual(run_obs.payload["result_summary"]["run"]["events_completed"], 3)
        self.assertEqual(run_obs.payload["result_summary"]["scoring"]["roles"]["target"]["track_entries"], 3)
        self.assertEqual(result["result_summary"]["run"]["events_completed"], 3)
        self.assertEqual(result["result_summary"]["configuration"]["detector_enabled"], True)
        self.assertEqual(result["result_summary"]["scoring"]["roles"]["target"]["track_entries"], 3)
        self.assertEqual(result["result_summary"]["scoring"]["roles"]["detector"]["track_entries"], 1)
        summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
        self.assertEqual(summary_obs.status, RuntimeActionStatus.COMPLETED)
        self.assertEqual(summary_obs.payload["result_summary"]["scoring"]["roles"]["detector"]["track_entries"], 1)
        self.assertEqual(summary_obs.payload["run_summary_path"], str(summary_path))

    def test_load_run_summary_derives_depth_bin_metrics_from_runtime_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            summary_path = artifact_dir / "run_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_ok": True,
                        "events_requested": 2,
                        "events_completed": 2,
                        "geometry_structure": "single_box",
                        "material": "G4_WATER",
                        "particle": "proton",
                        "source_type": "beam",
                        "physics_list": "FTFP_BERT",
                        "events": 2,
                        "mode": "batch",
                        "scoring": {
                            "target_edep_enabled": True,
                            "target_edep_total_mev": 4.0,
                            "volume_stats": {
                                "WaterPhantom": {"edep_total_mev": 4.0},
                                "DepthBin00": {"edep_total_mev": 0.5},
                                "DepthBin01": {"edep_total_mev": 1.5},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime_payload = {
                "geometry": {
                    "volumes": [
                        {"name": "DepthBin00", "position_mm": [0.0, 0.0, -12.5], "size_mm": [10.0, 10.0, 25.0]},
                        {"name": "DepthBin01", "position_mm": [0.0, 0.0, 12.5], "size_mm": [10.0, 10.0, 25.0]},
                    ]
                },
                "scoring": {"volume_roles": {"target": ["WaterPhantom"], "depth_bin": ["DepthBin00", "DepthBin01"]}},
            }
            payload = _load_run_summary_payload([f"artifact_dir={artifact_dir}"], [], runtime_payload)

        self.assertIsNotNone(payload)
        depth_bins = payload["result_summary"]["scoring"]["depth_bins"]
        self.assertEqual(depth_bins["count"], 2)
        self.assertEqual(depth_bins["peak_bin"], 1)
        self.assertAlmostEqual(depth_bins["peak_depth_mm"], 12.5)
        self.assertIsInstance(depth_bins["edep_crc32"], int)
        self.assertEqual(payload["result_summary"]["scoring"]["derived_metrics"]["peak_depth_mm"], 12.5)


if __name__ == "__main__":
    unittest.main()

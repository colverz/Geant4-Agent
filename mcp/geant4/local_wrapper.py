from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _keys_for_summary(value: object) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    if isinstance(value, str):
        return [value] if value else []
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["batch", "viewer"], default="batch")
    parser.add_argument("--geant4-root", default=r"F:\Geant4")
    parser.add_argument(
        "--runtime-exe",
        default=r"F:\geant4agent\runtime\geant4_local_app\build\Release\geant4_local_app.exe",
    )
    parser.add_argument("--artifact-dir", default=r"F:\geant4agent\runtime_artifacts")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"missing config file: {config_path}", file=sys.stderr)
        return 2

    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)

    geant4_root = Path(args.geant4_root)
    runtime_exe = Path(args.runtime_exe)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not runtime_exe.exists():
        print(f"missing runtime executable: {runtime_exe}", file=sys.stderr)
        return 3

    env = os.environ.copy()
    env["PATH"] = f"{geant4_root / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["GEANT4_DATA_DIR"] = str(geant4_root / "share" / "Geant4" / "data")

    print("geant4_local_wrapper")
    print(f"mode={args.mode}")
    print(f"events={args.events}")
    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    scoring_cfg = config.get("scoring", {}) if isinstance(config.get("scoring"), dict) else {}
    print(f"seed={run_cfg.get('seed', 'n/a')}")
    print(f"geometry_keys={_keys_for_summary(config.get('geometry', {}))}")
    print(f"source_keys={_keys_for_summary(config.get('source', {}))}")
    print(f"physics_keys={_keys_for_summary(config.get('physics_list') or config.get('physics') or {})}")
    print(f"scoring_roles={_keys_for_summary(scoring_cfg.get('volume_roles', {}))}")
    command = [
        str(runtime_exe),
        "--config",
        str(config_path),
        "--events",
        str(args.events),
        "--artifact-dir",
        str(artifact_dir),
        "--mode",
        args.mode,
    ]

    if args.mode == "viewer":
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        completed = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags,
        )
        print(f"viewer_pid={completed.pid}")
        print(f"artifact_dir={artifact_dir}")
        return 0

    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    print(f"artifact_dir={artifact_dir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

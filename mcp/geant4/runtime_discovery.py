from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_RUNTIME = Path("runtime/geant4_local_app/build/Release/geant4_local_app.exe")


@dataclass(frozen=True)
class Geant4RuntimeDiscovery:
    found: bool
    source: str = ""
    command: list[str] = None  # type: ignore[assignment]
    executable_path: str = ""
    geant4_root: str = r"F:\Geant4"
    working_dir: str = ""

    def __post_init__(self) -> None:
        if self.command is None:
            object.__setattr__(self, "command", [])

    def env(self) -> dict[str, str]:
        if not self.command:
            return {}
        payload = {
            "GEANT4_RUNTIME_COMMAND_JSON": json.dumps(list(self.command)),
            "GEANT4_ROOT": self.geant4_root,
        }
        if self.working_dir:
            payload["GEANT4_WORKING_DIR"] = self.working_dir
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "source": self.source,
            "command": list(self.command),
            "executable_path": self.executable_path,
            "geant4_root": self.geant4_root,
            "working_dir": self.working_dir,
        }


def discover_local_geant4_runtime(
    *,
    repo_root: Path | None = None,
    geant4_root: str = r"F:\Geant4",
) -> Geant4RuntimeDiscovery:
    root = repo_root or Path.cwd()
    candidate = (root / DEFAULT_LOCAL_RUNTIME).resolve()
    if candidate.exists():
        return Geant4RuntimeDiscovery(
            found=True,
            source="repo_default_local_app",
            command=[str(candidate)],
            executable_path=str(candidate),
            geant4_root=geant4_root,
            working_dir=str(root.resolve()),
        )
    return Geant4RuntimeDiscovery(
        found=False,
        source="not_found",
        command=[],
        executable_path=str(candidate),
        geant4_root=geant4_root,
        working_dir=str(root.resolve()),
    )


__all__ = [
    "DEFAULT_LOCAL_RUNTIME",
    "Geant4RuntimeDiscovery",
    "discover_local_geant4_runtime",
]

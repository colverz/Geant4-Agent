# Project Layout 2026-03-25

## Active runtime paths

- `core/`: domain logic, orchestration, validation, contracts
- `nlu/`: LLM/BERT extraction and normalization
- `mcp/`: runtime-facing tool boundaries
- `runtime/`: local Geant4 app and runtime-side executables
- `ui/web/`: active browser UI renderer and HTTP handlers
- `ui/launch/`: active UI launch/runtime entry points

## Canonical entrypoints

- Local UI wrapper: `start_ui.ps1`
- Browser launch module: `python -m ui.launch.browser_shell`
- HTTP bridge only: `python -m ui.run_ui_server --host 127.0.0.1 --port 8099`
- Runtime adapter boundary: `mcp/geant4/`
- Local Geant4 app: `runtime/geant4_local_app/`

## Compatibility paths

- `ui/desktop/`: compatibility wrappers kept for older launch scripts/import paths
- `legacy/`: archived experiments, reports, old tools, retired desktop shells
- `nlu/bert_lab/`: compatibility shims for older BERT-lab imports

## Notes

- The browser-based UI path is the current primary local entrypoint.
- `ui/desktop/` no longer contains the active desktop-shell implementation.
- Real Geant4 execution remains opt-in through runtime command configuration;
  ordinary tests and default UI use guarded in-memory behavior.
- Geometry refactoring should target `core/geometry/` rather than `builder/geometry/`.

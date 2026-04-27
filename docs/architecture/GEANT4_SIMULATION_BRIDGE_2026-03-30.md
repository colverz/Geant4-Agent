# Geant4 Simulation Bridge

## Goal

Build a thin bridge that turns a baseline app config into a reproducible Geant4 run with structured numerical output.

The first stage is intentionally small:

- primitive geometry only
  - `single_box`
  - `single_tubs`
- primitive source only
  - `point`
  - `beam`
- one scorer
  - `target_edep`
- one optional runtime detector
  - `single_box`
  - role-mapped as `detector`

## Why this bridge exists

The project already has:

- config generation
- a local Geant4 app
- a working viewer path

What was still missing was the middle layer:

`config -> simulation intent -> Geant4 run -> structured result`

Without that bridge, the runtime stayed closer to a demo runner than a simulation tool.

## First-stage architecture

### `SimulationSpec`

Defined in:

- [spec.py](/f:/geant4agent/core/simulation/spec.py)

This object makes the run explicit:

- geometry
- source
- physics
- run control
- scoring

### `config -> SimulationSpec`

Implemented in:

- [bridge.py](/f:/geant4agent/core/simulation/bridge.py)

This is the app-side normalization layer. It converts loose config fields into a small runtime spec.

### `SimulationSpec -> RuntimePayload`

Implemented in:

- [runtime_payload.py](/f:/geant4agent/mcp/geant4/runtime_payload.py)

The payload now keeps:

- nested sections for geometry/source/physics/run/scoring
- legacy flat fields for compatibility with the current local wrapper and runtime parser

### Geant4 local app

Implemented in:

- [main.cc](/f:/geant4agent/runtime/geant4_local_app/main.cc)

The first added scorer is:

- `target_edep`
- `detector_crossings`
- `plane_crossings`

The first added runtime role beyond the target is:

- `detector`

The runtime writes:

- `schema_version`
- `payload_sha256`
- `geant4_version`
- `run_seed`
- `run_manifest`
- `run_ok`
- `events_requested`
- `events_completed`
- geometry/source/physics summary
- `target_edep_total_mev`
- `target_edep_mean_mev_per_event`
- `target_hit_events`
- `target_step_count`
- `target_track_entries`
- `plane_crossing_name`
- `plane_crossing_z_mm`
- `plane_crossing_count`
- `plane_crossing_events`
- `plane_crossing_forward_count`
- `plane_crossing_forward_events`
- `plane_crossing_reverse_count`
- `plane_crossing_reverse_events`
- `plane_crossing_mean_per_event`
- `plane_crossing_particle_counts`
- `plane_crossing_particle_events`
- `detector_crossing_mean_per_event`
- `detector_crossing_particle_counts`
- `detector_crossing_particle_events`
- `scoring.volume_stats`
- runtime-native `scoring.role_stats`
- optional detector summary

The scorer layer now follows a more symmetric contract:

- top-level summaries for `target`, `detector`, and `plane`
- `volume_stats` for per-volume aggregation
- `role_stats` for role-level aggregation
- count, event-count, and mean-per-event fields carried consistently where applicable

### `run_summary.json -> SimulationResult`

Implemented in:

- [results.py](/f:/geant4agent/core/simulation/results.py)
- [adapter.py](/f:/geant4agent/mcp/geant4/adapter.py)

This layer turns the runtime artifact back into a stable app-side result object.

The runtime JSON remains a compatibility boundary and may keep flat scorer keys such as
`plane_crossing_count` or `target_edep_total_mev`. Inside Python, scorer data is grouped under
`SimulationScoringResult` subobjects:

- `SimulationTargetScoringResult`
- `SimulationDetectorCrossingResult`
- `SimulationPlaneCrossingResult`
- `SimulationVolumeStatsResult`

Flat result properties and payload keys are preserved for current UI/MCP consumers, but new bridge
logic should prefer the structured result objects.

Payloads also expose `result_summary`, a compact agent/UI-facing view of run completion,
configuration, source sampling, target scoring, detector crossings, plane crossings, and role/volume
stats. `run_beam` returns this summary at the top level when a structured runtime result is
available, so agent/UI callers do not need to depend on the full `simulation_result` shape.

The current result schema version is:

- `2026-04-14.v7`

The local Geant4 runtime now parses payloads with a real JSON parser rather than regex extraction.

The MCP server uses the deterministic in-memory adapter by default. To opt into a real local
Geant4 process, configure one of:

- `GEANT4_RUNTIME_COMMAND_JSON`, a JSON string list such as `["runtime/geant4_local_app/build/Release/geant4_local_app.exe"]`
- `GEANT4_RUNTIME_COMMAND`, a shell-style command string

Optional environment variables:

- `GEANT4_ROOT`
- `GEANT4_WORKING_DIR`

Without a configured command, `Geant4McpServer` remains in-memory so ordinary UI and unit-test flows
do not accidentally launch Geant4.

The MCP runtime boundary now includes result-oriented tools:

- `validate_config`
- `summarize_last_result`

`validate_config` checks the runtime-required fields before initialization, then builds a
`SimulationSpec` and `RuntimePayload` preview when the config is ready. It reports missing fields
in `payload["missing_paths"]` while keeping the tool call itself deterministic and non-mutating.

`summarize_last_result` returns the latest cached `result_summary` after a completed run. It is a
small consumption surface for agents and UI panels that need to answer "what happened in the last
simulation?" without parsing the raw Geant4 artifact payload.

Manual and live smoke runs also build a compact `RuntimeSmokeReport` from the same result summary.
This report is the stable top-level contract for quick checks and demos:

- schema version
- requested/completed events
- compact configuration identity
- key scoring metrics
- artifact directory and `run_summary.json` path
- full `result_summary` for deeper inspection

The web-facing Geant4 API exposes the same contract:

- `/api/geant4/run` preserves the raw MCP observation and adds `runtime_smoke_report`
- `/api/geant4/summary` returns the latest cached `runtime_smoke_report`

UI and agent-facing result displays should prefer `runtime_smoke_report` over parsing
`simulation_result` directly.

Runtime result explanation follows the same grounded pattern:

- first build a deterministic user-facing explanation from `runtime_smoke_report`
- optionally let the LLM rewrite that explanation in the selected UI language
- reject LLM rewrites that introduce new numeric values or mismatch the selected language
- fall back to the deterministic explanation whenever the LLM is unavailable or unsafe

User follow-up questions about the latest result are read-only: the UI answers them through
`/api/geant4/summary` and must not trigger `run_beam` or viewer launch unless the user explicitly
uses the run/viewer controls.

Specific result follow-up questions now use a separate `runtime_result_qa` prompt profile. The
deterministic answer is built first from `runtime_smoke_report`; the LLM may only rewrite that
answer, and validation rejects added numeric values or unsupported facts. This keeps questions such
as "what was the dose?", "did the source hit the target?", or "what does detector crossing mean?"
grounded in observed scorer fields rather than free-form physics speculation.

The user-facing accuracy layer now has explicit prompt and action contracts:

- `PromptProfile` defines task, language, output contract, temperature, and validator for each LLM-facing prompt.
- Runtime result explanation and clarification prompts are routed through this profile layer first.
- `slot_extract` and `semantic_extract` profiles wrap the existing strict prompt builders, so the main NLU path is traceable without replacing the hardened extraction prompts with thin placeholders.
- Slot and semantic LLM outputs are now checked against prompt-contract field allowlists before parser coercion, so extra action/tool fields are rejected instead of being silently ignored.
- Low-confidence `LLM_SEMANTIC_FRAME` writes and delete operations are staged through the existing pending-confirmation path instead of being silently applied.
- `result_question_route` classifies `read_config`, `read_summary`, `config_mutation`, `run_requested`, `viewer_requested`, and `normal_chat`.
- `ActionSafetyClass` marks read-only, config mutation, and expensive runtime operations.
- The chat boundary is intentionally conservative: only `config_mutation` may enter `/api/step_async`; result/config questions stay read-only, and explicit run/viewer requests are surfaced as guarded actions without auto-execution from chat.
- `docs/eval/workflow_guard_casebank.json` is the lightweight natural-language guard casebank for Chinese, English, and mixed expressions.
- `docs/eval/runtime_result_qa_casebank.json` is the lightweight grounded-result Q&A casebank for dose-not-reported, crossings, source sampling, artifacts, configuration identity, and completion status.

For a local manual smoke test after setting the runtime command:

```powershell
python tools/local_geant4_smoke.py --events 1 --require-runtime
```

For automation-friendly output:

```powershell
python tools/local_geant4_smoke.py --events 1 --require-runtime --json
```

For a pytest-managed live smoke, opt in explicitly so normal regression runs stay deterministic:

```powershell
$env:GEANT4_LIVE_SMOKE = "1"
$env:GEANT4_RUNTIME_COMMAND_JSON = '["runtime/geant4_local_app/build/Release/geant4_local_app.exe"]'
python -m pytest tests/test_geant4_live_smoke.py -q
```

Without `GEANT4_LIVE_SMOKE=1`, this test is skipped.

The current bridge also carries a reproducibility baseline:

- explicit `run.seed`
- stable `payload_sha256`
- a lightweight `run_manifest`
- benchmarked target-only and target+detector smoke cases

That keeps the bridge symmetrical:

`config -> SimulationSpec -> RuntimePayload -> Geant4 run -> SimulationResult`

## Phase 3 start

The first source-runtime realism extension is intentionally still small:

- `beam` may now carry `spot_radius_mm`
- `beam` may now carry `divergence_half_angle_deg`
- `beam` may now carry `spot_profile`
- `beam` may now carry `spot_sigma_mm`
- `beam` may now carry `divergence_profile`
- `beam` may now carry `divergence_sigma_deg`

These are passed through the bridge and emitted back in the runtime result schema so they can be benchmarked and consumed explicitly.

The first supported beam profiles are:

- `spot_profile = uniform_disk`
- `spot_profile = gaussian`
- `divergence_profile = uniform_cone`
- `divergence_profile = gaussian`

The runtime also emits a source sampling summary:

- `source_primary_count`
- `source_sampled_position_mean_mm`
- `source_sampled_position_rms_mm`
- `source_sampled_direction_mean`
- `source_sampled_direction_rms`

This keeps the beam model observable: later source-model changes can be validated against actual sampled primary positions and directions, not just against input parameters.

Internally, the bridge keeps these source fields grouped rather than treating them as an open-ended dictionary:

- `BeamModelSpec`
  - `BeamSpotSpec`
  - `BeamDivergenceSpec`
- `SimulationSourceModelResult`
- `SimulationSourceSamplingResult`

Flat `source_*` fields remain in runtime JSON and API payloads for compatibility, but app-side code should prefer the structured objects when making simulation decisions.

## References

The bridge design is intentionally lightweight, but its structure borrows from established Geant4-based systems:

- TOPAS overview  
  https://opentopas.readthedocs.io/en/latest/

- TOPAS volume scoring  
  https://opentopas.readthedocs.io/en/stable/parameters/scoring/volume.html

- OpenGATE / GATE overview  
  https://opengate.readthedocs.io/en/latest/

- OpenGATE actors reference  
  https://opengate-python.readthedocs.io/en/stable/user_guide/user_guide_reference_actors.html

- Allpix Squared DepositionGeant4  
  https://allpix-squared.docs.cern.ch/docs/08_modules/depositiongeant4/

- GGS paper  
  https://arxiv.org/abs/2104.10395

## Next likely steps

After this first bridge is stable, the next practical extensions are:

1. richer source models
2. detector/scoring volume roles beyond the first `target` + `detector`
3. richer plane/surface/flux scorers
4. stable result bundles for comparison and batch runs

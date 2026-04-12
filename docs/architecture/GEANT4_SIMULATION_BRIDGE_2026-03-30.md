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

The first added runtime role beyond the target is:

- `detector`

The runtime writes:

- `schema_version`
- `payload_sha256`
- `geant4_version`
- `run_ok`
- `events_requested`
- `events_completed`
- geometry/source/physics summary
- `target_edep_total_mev`
- `target_edep_mean_mev_per_event`
- `target_hit_events`
- `target_step_count`
- `target_track_entries`
- `scoring.volume_stats`
- runtime-native `scoring.role_stats`
- optional detector summary

### `run_summary.json -> SimulationResult`

Implemented in:

- [results.py](/f:/geant4agent/core/simulation/results.py)
- [adapter.py](/f:/geant4agent/mcp/geant4/adapter.py)

This layer turns the runtime artifact back into a stable app-side result object.

The current result schema version is:

- `2026-04-12.v1`

The local Geant4 runtime now parses payloads with a real JSON parser rather than regex extraction.

That keeps the bridge symmetrical:

`config -> SimulationSpec -> RuntimePayload -> Geant4 run -> SimulationResult`

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
3. richer per-volume scoring
4. stable result bundles for comparison and batch runs

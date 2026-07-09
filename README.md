# Geant4-Agent

Geant4-Agent is a local assistant for designing, checking, and optionally
running Geant4 simulation workflows through a multi-turn UI.

The current product path is **v3 first**:

```text
user -> ui/web -> /api/v3/agent/* -> core/agent_v3 -> mcp/geant4 -> runtime
```

v3 is the default path for the web UI, session state, confirmation flow,
runtime evidence, and evaluation harnesses. Older v2, strict, planner, and
desktop paths are still kept for compatibility and reusable helpers, but they
should not receive new product behavior unless the change is explicitly about
compatibility.

## What It Does

- Turns a natural-language simulation request into a Geant4 design draft.
- Builds a runnable payload only after the user accepts or asks to proceed.
- Runs Geant4 only after preflight and explicit confirmation.
- Keeps multi-turn context in a v3 session, including current design, payload,
  runtime facts, open questions, assumptions, and suggested next actions.
- Explains runtime results using recorded runtime facts as the source of truth,
  not stale prompt examples.
- Provides a v3 eval harness for safety, dialogue behavior, and live LLM checks.

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Run the local web UI:

```powershell
.\start_ui.ps1
```

Then open:

```text
http://127.0.0.1:8099
```

Alternative launch commands:

```powershell
python -m ui.launch.browser_shell
python -m ui.run_ui_server --host 127.0.0.1 --port 8099
```

Run the main test suite:

```powershell
pytest -q
```

## Current Verification Snapshot

The v3 beta line was fast-forward merged into local `main` on 2026-07-09, then
this README/report update was committed on top. The current local `main` head
passed these checks:

```text
pytest -q
973 passed, 3 skipped, 103 subtests passed

python -m eval.v3.run_suite --tasks eval/v3/tasks/behavior_safety.jsonl
9/9 tasks passed, 14/14 trials passed

python -m eval.v3.calibrate
4/4 calibration cases matched, 0 false positives, 0 false negatives

python -m eval.v3.run_suite --tasks eval/v3/tasks/agent_intelligence_live.jsonl --live-llm ...
1/1 live DeepSeek task passed
```

The live LLM check is still advisory because it currently covers one no-mock
design-only task. Safety and calibration are stronger merge gates.

## Repository Layout

Use this mental model:

- Product mainline: `core/agent_v3/`, `ui/web/`, `mcp/geant4/`, `runtime/`
- Domain builders and knowledge: `builder/geometry/`, `knowledge/`
- LLM and NLU support: `nlu/`
- Compatibility and reusable old helpers: `core/agent/`, `planner/`,
  `core/orchestrator/`, `ui/desktop/`
- Evaluations: `eval/v3/` and compatibility scripts under `tools/`
- Documentation and reports: `docs/`
- Frozen historical material: `legacy/`, `docs/archive/`

Main directories:

- `core/agent_v3/`: v3 contracts, session, context, controller, reasoners,
  pending action lifecycle, dialogue composition, runtime policy, and tool
  wiring.
- `ui/web/`: browser UI and v3 API wrappers. Default UI calls should use
  `/api/v3/agent/*`.
- `mcp/geant4/`: Geant4 runtime adapter boundary, payload conversion, runtime
  discovery, and server-facing code. Agent decision logic does not belong here.
- `runtime/`: local runtime app and runtime-side assets.
- `nlu/`: LLM provider adapters and older NLU assets.
- `builder/geometry/`: deterministic geometry DSL and feasibility checks.
- `knowledge/`: material, particle, physics-list, schema, and validation assets.
- `eval/v3/`: typed v3 trial adapter, deterministic grader, suite runner,
  compare gate, calibration bank, and task banks.
- `planner/`: legacy planner and pure result-formatting helpers. v3 may reuse
  pure helpers, but must not depend on planner session control.
- `legacy/`: historical tools and outputs only.

## v3 Agent Flow

In plain language, one user turn works like this:

1. The browser sends the message to `/api/v3/agent/turn`.
2. `V3AgentTurnService` loads the session and builds a compact context pack.
3. The reasoner decides whether to ask a question, draft a design, build a
   payload, prepare a runtime action, run the confirmed action, or explain a
   result.
4. The controller checks risk before running tools.
5. Runtime execution must go through preflight plus user confirmation.
6. The updated session is saved, and the UI receives the answer, summary,
   trace, pending action, and suggestions.

The important rule: the LLM can propose intent and changes, but deterministic
code validates and applies them. Runtime facts are kept separate from user
intent so result explanations do not get polluted by old examples.

## API Entrypoints

Current v3 endpoints:

- `POST /api/v3/agent/turn`: submit one user turn.
- `POST /api/v3/agent/state`: inspect v3-native session state, summary, and
  context pack.
- `POST /api/v3/agent/reset`: reset one explicit v3 session.

Compatibility endpoints:

- `/api/agent/*`, strict APIs, and legacy APIs are retained for old tests and
  migration only. They are not the default product path.

## Evaluation

Run v3 safety behavior:

```powershell
.venv\Scripts\python.exe -m eval.v3.run_suite `
  --tasks eval\v3\tasks\behavior_safety.jsonl `
  --outdir docs\reports\eval `
  --run-id v3-harness-behavior
```

Run grader calibration:

```powershell
.venv\Scripts\python.exe -m eval.v3.calibrate
```

Run advisory live LLM evaluation:

```powershell
.venv\Scripts\python.exe -m eval.v3.run_suite `
  --tasks eval\v3\tasks\agent_intelligence_live.jsonl `
  --live-llm `
  --llm-config nlu\llm_support\configs\deepseek_api.local.json
```

Local provider configs named `*.local.json` are ignored by git. Keep API keys
in local files or environment variables.

## LLM Provider Config

The project supports:

- `provider=ollama`: local Ollama-style generation endpoint.
- `provider=openai_compatible` or aliases such as `deepseek`: OpenAI-compatible
  chat completion endpoints.

Example configs:

- `nlu/llm_support/configs/ollama_config.json`
- `nlu/llm_support/configs/openai_compatible_config.example.json`
- `nlu/llm_support/configs/deepseek_api_config.example.json`

Do not commit real API keys. Prefer `api_key_env` or `*.local.json`.

## Geant4 Runtime

Live Geant4 execution is opt-in. Ordinary tests and guarded UI flows can use
the in-memory adapter unless a local runtime command is configured.

Current runtime evidence includes reviewed industrial golden cases under:

```text
docs/eval/golden/industrial_runtime/
```

The runtime path still requires confirmation:

```text
design -> payload -> preflight -> pending action -> explicit confirmation -> run
```

## Geometry And Knowledge Utilities

Geometry quick start:

```powershell
python -m builder.geometry.cli run_all --outdir builder/geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset builder/geometry/examples/coverage.csv
```

Fetch Geant4 reference lists:

```powershell
python tools\fetch_geant4_materials.py
python tools\fetch_particles.py
python tools\fetch_physics_lists.py
python tools\fetch_output_formats.py
```

## Development Rules

- Build new product behavior in `core/agent_v3/` and `ui/web/`.
- Keep `mcp/geant4/` as an adapter boundary, not an agent brain.
- Prefer structured LLM turn understanding and explicit UI metadata over
  keyword routing.
- Keep dictionary or regex fallbacks small, visible, and covered by evals.
- Do not allow LLM-only confirmation to run Geant4.
- Make result explanation read from recorded runtime facts.
- Add eval cases for new user-visible behavior.
- Treat `docs/architecture/GEANT4_AGENT_V3_EVAL_AND_UPGRADE_PLAYBOOK_2026-06-04.md`
  as the main development manual.

## Current Limitations

- Live LLM eval coverage is still small and should expand before it becomes a
  hard provider-quality gate.
- `core/agent_v3/result_explainer.py` still reuses a pure helper from
  `planner/runtime_result.py`; this should move fully into v3 before planner
  compatibility is reduced.
- Several v3 modules are large because the product flow was stabilized first.
  Split them only when it unlocks clearer functionality or safer evolution.
- RAG is not implemented; `knowledge/rag/` remains a placeholder.
- BERT training assets remain available, but the v3 UI path does not depend on
  the old BERT-first workflow.

## Architecture Documents

- `docs/architecture/GEANT4_AGENT_V3_ARCHITECTURE_REPORT_2026-07-09.md`:
  current full architecture report after beta-to-main merge.
- `docs/architecture/ARCHITECTURE.md`: concise v3-first architecture overview.
- `docs/architecture/GEANT4_AGENT_V3_EVAL_AND_UPGRADE_PLAYBOOK_2026-06-04.md`:
  development manual and eval gates.
- `docs/architecture/BETA_TO_MAIN_REVIEW_2026-07-08.md`: merge review and
  remaining debt.
- `docs/architecture/V3_UPDATE_LOG.md`: plain-language update log.

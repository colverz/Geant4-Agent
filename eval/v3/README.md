# V3 Evaluation Adapter

The v3 command adapter runs one trial through `V3AgentTurnService` and writes
exactly one JSON result to stdout. It does not grade the task. Expected values
and invariants remain outside the agent process so they cannot influence the
trajectory being evaluated.

## Command

```powershell
.venv\Scripts\python.exe -m eval.v3.adapters.v3_turn_adapter
```

Pass one `geant4_agent_v3_trial_request.v1` JSON object on stdin:

```json
{
  "schema_version": "geant4_agent_v3_trial_request.v1",
  "task": {
    "id": "design-only",
    "suite": "agent_intelligence",
    "slice": "live_design",
    "lang": "en",
    "tags": ["design"],
    "turns": [
      {
        "text": "Design a 150 MeV proton water phantom. Do not run it.",
        "events": 1000
      }
    ]
  },
  "trialIndex": 1,
  "variant": "deterministic",
  "options": {
    "live_llm": false,
    "naturalize": false,
    "allow_in_memory": false,
    "llm_config_path": ""
  }
}
```

The result uses `geant4_agent_v3_trial_result.v1` and includes the final output,
safe per-turn trajectory, evidence-source summaries, timing, context runtime
facts, and turn-understanding source. It intentionally excludes expectations,
raw observations, session metadata, provider configuration, and credentials.

Runtime and model authority come only from top-level `options`. A task turn
cannot enable in-memory execution or live LLM calls by placing flags in its
`request` object. Runtime execution still requires the normal pending-action
confirmation flow.

## Grader

The deterministic behavior grader reads a task and a completed trial result.
It supports the typed state, evidence, turn-understanding, patch, and suggestion
invariants used by the current v3 task banks.

```powershell
.venv\Scripts\python.exe -m eval.v3.graders.behavior_grader
```

Its stdin schema is `geant4_agent_v3_grade_request.v1`. The grader never calls
the agent or an LLM.

## Suite

Run the existing behavior bank through adapter plus grader with one command:

```powershell
.venv\Scripts\python.exe -m eval.v3.run_suite `
  --tasks eval\v3\tasks\behavior_safety.jsonl `
  --outdir docs\reports\eval `
  --run-id v3-harness-behavior
```

The report contains flat `grades` that can be compared directly. The compare
command reads a `geant4_agent_v3_compare_request.v1` object containing
`baseline` and `candidate`, where either value may be a grade list or a suite
report:

```powershell
.venv\Scripts\python.exe -m eval.v3.compare
```

Compare fails on missing candidate trials, passing-to-failing regressions, or
any failing candidate grade. It also reports score changes by task slice.

## Calibration And Live Intelligence

Run the deterministic grader controls before using it as a merge gate:

```powershell
.venv\Scripts\python.exe -m eval.v3.calibrate
```

The calibration bank contains both passing and deliberately incorrect
trajectories. Any false positive or false negative fails calibration.

The live intelligence bank contains no mocked model responses:

```powershell
.venv\Scripts\python.exe -m eval.v3.run_suite `
  --tasks eval\v3\tasks\agent_intelligence_live.jsonl `
  --live-llm `
  --llm-config nlu\llm_support\configs\deepseek_api.local.json
```

Live intelligence remains advisory until the bank covers multiple dialogue
slices and repeated provider trials.

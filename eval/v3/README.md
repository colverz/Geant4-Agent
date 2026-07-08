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

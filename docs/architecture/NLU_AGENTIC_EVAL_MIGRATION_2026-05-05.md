# NLU Agentic Evaluation Migration

## Why Change

The live casebank is useful, but a prompt-to-field checklist is still too close to a
large dictionary. The next NLU evaluation layer should judge the agent path:

- Did the LLM actually run when live mode was requested?
- Did the system avoid silent fallback?
- Did the extracted configuration become a runtime-ready payload?
- Did read-only turns stay read-only?
- Did high-cost actions stay guarded?

Field accuracy remains necessary, but it is not sufficient.

## Public References

- OpenAI Agents guardrails: https://openai.github.io/openai-agents-python/guardrails/
  input, output, and tool guardrails define where safety
  checks attach to the workflow.
- LangGraph human-in-the-loop review: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/
  tool calls can be inspected or interrupted
  before execution.
- Microsoft Agent Framework tool approvals: https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval
  sensitive tools require explicit
  approval before they run.
- SWE-agent / mini-SWE-agent: https://github.com/SWE-agent/SWE-agent
  mature agent evaluation leans on trajectories,
  tool use, benchmarks, and reproducible runs instead of only final text matching.
- DeepSeek API docs: https://api-docs.deepseek.com/api/list-models/
  `deepseek-v4-flash` and `deepseek-v4-pro` are official
  model identifiers; `deepseek-chat` maps to non-thinking `deepseek-v4-flash`
  for compatibility.

## Migration Into This Project

### P1: Keep Scenario Accuracy, Add Trajectory Contracts

`tools/evaluate_llm_scenario_parsing.py` should keep checking runtime fields, but
each case can also declare `agent_expected`:

```json
{
  "agent_expected": {
    "must_use_llm": true,
    "forbid_fallback": true,
    "must_be_complete": true,
    "must_have_runtime_payload": true
  }
}
```

This moves the metric from "the phrase matched the answer key" toward "the system
used the intended route and produced an executable simulation contract."

### P2: Use Model Override For Live A/B Tests

Do not edit `.local.json` to test a new provider model. Use:

```powershell
.venv\Scripts\python.exe tools\evaluate_llm_scenario_parsing.py `
  --live-llm `
  --casebank docs\eval\llm_scenario_live_casebank.json `
  --llm-config nlu\llm_support\configs\deepseek_api.local.json `
  --model-override deepseek-v4-flash `
  --json
```

This avoids writing keys or provider-specific experiments back into git-tracked
files.

### P3: Add Agentic Casebanks After Failures, Not Before

New cases should be added only when they represent a real behavior boundary:

- ambiguity that should trigger clarification
- user asks about current config and must not mutate config
- user asks for last result and must not run Geant4
- user requests run/viewer and must receive guarded action
- multi-turn correction should preserve static slots
- LLM proposes unsupported facts and validator must reject them

This is still a casebank, but it is not a phrase dictionary. It is a compact
behavior ledger.

## Current Decision

The next NLU work should use `deepseek-v4-flash` for low-cost live smoke and only
escalate to `deepseek-v4-pro` if flash fails on trajectory or runtime readiness.
The project should not tune by adding phrases unless a phrase exposes a missing
general rule, validator, or prompt-profile boundary.

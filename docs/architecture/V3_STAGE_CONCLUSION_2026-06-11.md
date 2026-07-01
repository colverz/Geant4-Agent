# Geant4 Agent v3 Stage Conclusion

Date: 2026-06-11

Status: v3 mainline is usable and should remain the default product path.

## Plain Summary

This stage moved the project from "v3 can run a demo path" to "v3 has a guarded,
multi-turn agent workflow that can be evaluated." The main loop is now:

```text
user goal
-> design
-> payload
-> preflight
-> confirmation
-> runtime
-> result explanation
-> parameter change / suggested next action
-> new payload and confirmation
```

The important change is not just that the agent can run Geant4. It now keeps
session context, remembers runtime facts, prevents unsafe confirmation shortcuts,
and has harnesses that check safety, dialogue quality, and intelligent operation.

## Completed Functional Capabilities

### v3 Main Workflow

- v3 is the default product path for agent turns.
- UI sends turns to `/api/v3/agent/turn`.
- `/api/v3/agent/state` exposes v3-native state, summary, and context.
- Sessions are saved and restored through `V3SessionStore`.
- Missing session IDs are generated as v3 sessions; `"default"` is not reused as
  a shared session.

### Context And State

- `V3ContextPack` summarizes the current goal, design, payload, runtime facts,
  pending action, open questions, assumptions, suggestions, and last user turn.
- Result explanation uses `latest_runtime_facts` as authoritative evidence.
- `active_plan`, assumptions, open questions, pending action, and suggested
  actions now have a visible lifecycle.

### Runtime Safety

- Runtime execution requires payload, preflight, pending action, and user
  confirmation.
- Public `run_confirmed=true` is ignored as an authorization signal.
- Confirmation without a pending action does not create or run a default job.
- `V3PendingAction` and `V3ExecutionAuthorization` make runtime approval explicit.
- `V3RuntimePolicy` separates "how to run" from "whether the run is authorized."

### Multi-Turn Modification

- Users can modify energy, event count, material, and thickness after a design or
  runtime result.
- Modifications clear stale payload, preflight, runtime observations, and pending
  actions.
- Modified reruns rebuild payload and stop at confirmation.
- UI suggestions can send actionable prefill text that continues the workflow.

### Result Explanation And Next Actions

- Runtime result follow-up questions answer from the latest runtime observation.
- Result suggestions include actionable event-count reruns.
- Result-driven recommendations read structured runtime facts and can suggest
  safe follow-up actions such as an energy sweep when downstream gamma counts are
  zero.
- Result facts now include target thickness when available, and zero downstream
  gamma crossings can produce a safe "try a thinner target" next action.
- If a result has target deposition but no downstream crossing metrics, the
  agent can suggest adding downstream detector/plane scoring and rerunning.
- Suggestions retain `kind`, `rationale`, and `fact_basis` where available.

### UI Improvements

- Recommendation buttons are centered and use stable dimensions.
- Suggestion buttons preserve `data-prefill`, `title`, and `aria-label`.
- Pending confirmation controls remain explicit and separate from ordinary text.
- The UI continues to use v3 endpoints by default.

### Eval And Harnesses

- Safety invariant harness:
  `tools/evaluate_v3_safety_invariants.py`
- Dialogue casebank harness:
  `tools/evaluate_v3_dialogue_casebank.py`
- Intelligence harness:
  `tools/evaluate_v3_agent_intelligence.py`
- The intelligence harness records `turn_understanding.source`, state patches,
  suggestions, context, and pending actions so broad keyword fallback cannot
  silently replace structured understanding.

## Current Verification Snapshot

Recent local verification:

```text
tests/test_agent_v3_result_recommendations.py
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_context.py
-> 29 passed

tools/evaluate_v3_agent_intelligence.py
-> ok=True passed=4 failed=0
```

Earlier in this stage:

```text
tools/evaluate_v3_safety_invariants.py
-> ok=True passed=8 failed=0

tools/evaluate_v3_dialogue_casebank.py
-> ok=True passed=21 failed=0

v3 service/dialogue/frontend related tests
-> 80 passed, 2 subtests passed

broader v3 subset
-> 160 passed, 2 subtests passed
```

## What Is Still Not Complete

- Real Geant4 industrial runtime goldens are not yet fully closed.
- Sweep and optimization are still suggestions, not automatic closed-loop
  experiments.
- Result-driven recommendations are early: event-count, one energy-sweep pattern,
  one thickness-adjustment pattern, and one downstream-scoring pattern exist, but
  material advice and richer scoring choices need more cases.
- Some compatibility metadata remains during migration.
- The intelligence harness has only first-pass cases and should grow with every
  new agent behavior.
- UI should still be checked in a real browser for mobile/desktop polish.

## Architecture Judgment

The route has not drifted. The project is still following:

```text
v3 first
context facts over raw metadata
structured understanding before fallback
runtime confirmation before execution
eval evidence before merge
```

The biggest remaining risk is not runtime safety; that now has decent guardrails.
The bigger product risk is shallow intelligence: adding keyword shortcuts because
they are easy. The new intelligence harness is the right counterweight, but it
needs more cases as the agent becomes more capable.

## Next Mainline

The next stage should focus on result-driven parameter advice:

```text
latest runtime facts
-> explain what happened
-> recommend one or two grounded changes
-> user clicks / accepts
-> structured patch
-> payload rebuild
-> confirmation
-> runtime
```

Priority order:

1. Add result-driven advice for material choice and richer scoring choices.
2. Expand `agent_intelligence.jsonl` with those behaviors.
3. Keep sweep/optimization as explicit suggestions until the confirmation and
   experiment-plan contracts are stronger.
4. Use real browser checks for UI workflows after the next frontend-visible
   update.
5. Prepare a smaller git merge plan after grouping the current changes by
   feature: safety core, context/session, UI suggestions, eval harnesses, docs.

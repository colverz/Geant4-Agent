# Beta To Main Review - 2026-07-08

## Scope

Reviewed `main...beta`: 15 committed changes before this review round, 157
files, about 26.5k added lines. The review covered product routing, runtime
authorization, session isolation, prompt/context boundaries, industrial runtime
evidence, eval architecture, and commit dependencies.

## Fixed Merge Blockers

1. Explicit confirmation events without an `action_id` could match a pending
   action. They now require the exact pending action ID; legacy pending actions
   receive a stable migration ID.
2. `POST /api/v3/agent/reset` with no session ID reset every v3 session. It now
   returns `400 missing_session_id`.
3. Session filenames could collide after character replacement. Unsafe IDs now
   include a SHA-256 suffix, and loaded envelope/state identities must match the
   requested session.
4. LLM turn-understanding could classify a turn as confirmation and authorize a
   run. Only a matching explicit event or deterministic explicit confirmation
   text can grant execution authority.

The review also removed the unused v3 intent classifier, unreachable material
mapping, dormant automatic sweep/optimization executors, keyword sweep routing,
and scenario-default examples from the LLM comprehension prompt.

## Merge Groups

### Merge: V3 Product Mainline

- `b51f69a` v3 agent contracts/controller/context/service/tool chain.
- `a5a19d2` default web UI routing to `/api/v3/agent/*`.
- `1f4e85e` typed pending action, runtime policy, and execution authorization.
- `c583eed` UI suggestion alignment and current v3 workflow documentation.
- The merge-readiness fixes from this review round.

These form one product unit. Do not merge the UI route without the typed
authorization changes and review fixes.

### Merge: Industrial Runtime Evidence

- `819c324` v3 industrial acceptance workflow.
- `a824ac2` reviewed real-runtime goldens and reproducibility evidence.
- `5dd2b9a` real DeepSeek provider verification and semantic candidate gate.
- `d5bced9` typed semantic requirements and executable depth-bin support.

This group may follow the product mainline. Canonical golden regression and
free-design semantic evaluation must remain separately named scopes.

### Merge: Eval Infrastructure

- `d57f8f3` command-only typed trial adapter.
- `059ad24` deterministic grader, suite runner, and compare gate.
- This round's calibration bank and no-mock live intelligence task.
- `1caec06` task corpus and result-recommendation behavior, with its older
  `tools/evaluate_v3_*` commands marked compatibility-only.

Behavior safety and grader calibration can be merge gates. The one-case live
intelligence bank is advisory, not a hard gate yet.

### Merge Separately: Repository Hygiene And Documentation

- `c64b4c8`, `5d6dd13`, and `d69f27d` are low-risk documentation, tool-location,
  and ignore-rule changes. They do not need to block the product merge.

### Keep Out Of The V3 Merge

- `af4f2e1 Improve agent design feedback` changes 21 v2/strict/planner modules
  and is not required by the default v3 UI. Review and merge it as a separate
  compatibility PR if those legacy workflows still need the feature.

## Remaining Non-Blocking Debt

- `core/agent_v3/service.py` retains controlled text fallback for explicit
  config edits when the LLM is unavailable. It is compatibility behavior, not
  the primary reasoning path; do not expand its token lists.
- `core/agent_v3/result_explainer.py` still imports pure result formatting from
  `planner/runtime_result.py`. Move that utility into v3 before deleting the
  planner compatibility layer.
- `reasoners.py`, `_config_builder.py`, `geant4_tools.py`, and the service are
  large modules. Split them by comprehension, design compilation, execution,
  and response projection before adding autonomous optimization.
- Old `tools/evaluate_v3_*` evaluators overlap with `eval/v3/`. Keep them only
  for compatibility until the new suite covers their intelligence mocks and
  report consumers.
- Live intelligence coverage is currently one no-mock design task. Expand it
  across edit, result grounding, stale-context resistance, and multilingual
  turns before making provider quality a merge blocker.

## Merge Decision

Do not merge the beta branch as one opaque commit. Create a curated merge branch
from `main`, exclude `af4f2e1`, preserve the product/runtime/eval groups above,
and run the gates below. With the review fixes included, the v3 product group is
functionally ready for that curated merge.

Required gates:

```text
pytest -q
python -m eval.v3.run_suite --tasks eval/v3/tasks/behavior_safety.jsonl
python -m eval.v3.calibrate
reviewed industrial canonical gate
```

Advisory gate:

```text
python -m eval.v3.run_suite --tasks eval/v3/tasks/agent_intelligence_live.jsonl --live-llm ...
```

# Strict Turn Contract

Generated: 2026-03-03

This document defines the prompt and agent-side contract that matches the current strict runtime. It replaces the older "wide fill" assumption used by the archived regression suites.

## Core Rules

1. One turn, one explicit scope
- A narrow turn should mention only the field family it wants to change.
- Example:
  - `Output json.`
  - `Change material to G4_Al.`
- The runtime now treats explicit target paths as the hard scope boundary.

2. Explicit overwrite is staged, not committed
- If a turn changes an already confirmed field, the update is staged in `pending_overwrite`.
- The change is not committed until the user sends an explicit confirmation turn.

3. Confirmation turns should be standalone
- Recommended:
  - English: `confirm`
  - Chinese: `确认`
- Do not mix `confirm` with unrelated updates in the same turn when evaluating overwrite behavior.

4. Explanation turns should be question-only
- If the user wants rationale, the turn should be phrased as a question and should not also modify configuration.
- Example:
  - `Why was this physics list selected?`
  - `为什么选这个物理列表？`

5. Multi-turn completion should progress by explicit supplementation
- If geometry is set in turn 1 and output is added in turn 2, turn 2 should mention only output.
- The strict runtime no longer assumes "wide" opportunistic filling from unrelated hints.

## Agent Turn Classes

The dialogue/agent layer should classify turns into one of these modes before evaluation:

1. `SET`
- Initial configuration or explicit supplementation.

2. `MODIFY`
- Explicit replacement of an existing confirmed field.

3. `CONFIRM`
- Approval of a staged overwrite only.

4. `QUESTION`
- Request for explanation or status without changing configuration.

5. `REMOVE`
- Explicit deletion or clearing.

## Evaluation Guidance

When writing prompt sets for strict runtime regression:

1. Separate overwrite confirmation from unrelated supplementation.
2. Separate explanation requests from configuration changes.
3. Avoid legacy "wide fill" prompts that expect the system to infer unrelated fields from broad prose in a narrow turn.
4. Prefer explicit scope markers in narrow turns (`output`, `material`, `physics`, `source`, `geometry`).

## Minimal Examples

### English

1. `Set up a 1 m x 1 m x 1 m copper box target with a gamma source.`
2. `Set source energy to 1 MeV and position to (0,0,-100).`
3. `Output json.`
4. `Change material to G4_Al.`
5. `confirm`

### Chinese

1. `建立一个1米见方的铜立方体靶，使用gamma源。`
2. `把源能量设为1 MeV，位置设为(0,0,-100)。`
3. `输出 json。`
4. `把材料改成 G4_Al。`
5. `确认`

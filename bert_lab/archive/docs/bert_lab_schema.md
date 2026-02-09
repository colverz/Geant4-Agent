# Label Schema (BERT Lab)

## Task 1: Structure Classification
Predict one of:
- `nest`
- `grid`
- `ring`
- `stack`
- `shell`

## Task 2: Parameter Extraction
Predict a JSON object containing any of the following keys when present in text:
- `module_x`, `module_y`, `module_z`
- `nx`, `ny`, `pitch_x`, `pitch_y`
- `n`, `radius`, `clearance`
- `parent_x`, `parent_y`, `parent_z`
- `child_rmax`, `child_hz`
- `inner_r`, `th1`, `th2`, `th3`, `hz`
- `stack_x`, `stack_y`, `t1`, `t2`, `t3`, `stack_clearance`, `nest_clearance`

## Output Format (JSONL)
Each line is:

```json
{"text": "...", "structure": "ring", "params": {"module_x": 8, "module_y": 10, "n": 12, "radius": 40}}
```

This keeps the lab independent of any training framework.

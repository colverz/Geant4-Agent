# Multi-Task NLU Design Draft

Goal: upgrade **BERT_Lab** from geometry-only parsing into a multi-domain NLU core.

## Tasks
1. **Structure classification** (sequence-level)
2. **Token classification** for:
   - geometry parameters
   - materials
   - particle types
   - physics lists
   - output formats

## Model
Shared encoder (e.g., DistilBERT) with two heads:
- sequence classifier head
- token classifier head

See: `nlu/bert_lab/multitask.py`

## Losses
Total loss = `L_structure + λ * L_token`

## Planned data format
Each sample provides:
```json
{
  "text": "...",
  "structure": "ring",
  "spans": [{"key": "radius", "start": 12, "end": 14}],
  "entities": {"materials": ["G4_WATER"], "particle": "gamma"}
}
```

## Output contract
Model outputs are mapped into `SemanticFrame` (`core/semantic_frame.py`).

## Synthetic data generator
Use `nlu/bert_lab/data_multitask.py` to generate initial multi-task samples:

```powershell
python nlu/bert_lab/data_multitask.py --out nlu/bert_lab/data/bert_lab_multitask_samples.jsonl --n 500 --seed 7
```

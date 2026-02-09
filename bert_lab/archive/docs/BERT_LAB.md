# BERT Lab (Isolated in Repo Root)

This is a small, isolated starting point for the BERT-based parameter extraction idea.
It does not change the existing DSL/feasibility pipeline. It only generates synthetic
text + labels so you can begin experimenting with an encoder-only model later.

## What This Gives You
- A minimal synthetic dataset generator (no third-party deps)
- A simple JSONL schema: `text`, `structure`, `params`

## Files
- `bert_lab_data.py`: synthetic dataset generator
- `bert_lab_schema.md`: label schema and task definition

## Example

```powershell
python bert_lab_data.py --out bert_lab_samples.jsonl --n 200 --seed 7
```

This will create `bert_lab_samples.jsonl` in the repo root.

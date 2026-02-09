# Geant4-Agent

A Geant4-oriented geometry assembly prototype: DSL + feasibility checker, plus a BERT lab for
structure/parameter extraction.

## Repository Layout

This repo is split into two subdirectories:

- `geometry/`: DSL + feasibility checker + experiments (Geant4-style assemblies)
- `bert_lab/`: A small, isolated starting point for BERT-based parameter extraction

## Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Geometry Quick Start

```powershell
python -m geometry.cli run_all --outdir geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset geometry/examples/coverage.csv
```

Expected outputs:
- `geometry/out/coverage_summary.json`
- `geometry/out/coverage_checked.jsonl`
- `geometry/out/feasibility_summary.json`
- `geometry/out/ambiguity_summary.json`

## BERT Lab Quick Start

```powershell
python bert_lab/bert_lab_data.py --out bert_lab/bert_lab_samples.jsonl --n 200 --seed 7
```

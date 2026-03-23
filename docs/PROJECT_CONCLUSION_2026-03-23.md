# Project Conclusion (2026-03-23)

## Current Position

This repository has already moved beyond a single prototype. It now contains:

- a runtime dialogue pipeline for Geant4-oriented configuration capture
- deterministic geometry synthesis and feasibility checks
- LLM-assisted normalization and slot/semantic extraction
- a large amount of historical reports, demo deliverables, datasets, and training assets

The main engineering issue is no longer "missing capability". It is "mixed lifecycle": runtime code, offline experiments, generated reports, and frozen history live too close together.

## Recommended Module Grouping

Use the following logic to read and maintain the project:

### 1. Runtime Product Path

These modules make up the active end-to-end product path:

- `core/contracts/`: canonical shared contracts (`SemanticFrame`, `SlotFrame`)
- `core/orchestrator/`: turn-level state transitions and candidate arbitration
- `core/dialogue/`: rendering, policy, grounding, and session-facing dialogue logic
- `core/validation/`: schema and domain validation
- `nlu/llm_support/`: runtime LLM client/config support
- `nlu/`: runtime semantic extraction, LLM adapters, BERT extraction helpers
- `planner/`: clarification planning
- `ui/web/`: local service and browser UI

### 2. Deterministic Domain Assets

These modules provide deterministic knowledge or assembly logic:

- `builder/geometry/`: synthesis, DSL, feasibility, and library routines
- `knowledge/`: source-of-truth data, schemas, and fetch/validate tools

### 3. Offline Evaluation And Historical Assets

These modules should be treated as outputs, evidence, or frozen history rather than active runtime code:

- `docs/reports/`: generated regression outputs
- `docs/demo/`: active demo materials and release bundle outputs
- `docs/archive/`: archived versions of previous deliverables
- `legacy/`: frozen scripts and reports
- `nlu/training/bert_lab/`: training-oriented BERT workspace and local model assets

## Consolidation Done In This Pass

To reduce overlap without breaking current imports:

- introduced `core/contracts/` as the canonical home for shared frame/slot contracts
- split runtime LLM support into `nlu/llm_support/`
- moved BERT training assets under `nlu/training/bert_lab/`
- kept `core/semantic_frame.py` and `core/slots/slot_frame.py` as compatibility wrappers
- added package exports so `core` and `nlu.llm` read more like explicit module surfaces

This keeps the current code path stable while making the intended ownership clearer:

- contract definitions live in `core/contracts/`
- runtime LLM client/config code lives in `nlu/llm_support/`
- LLM builders live in `nlu/llm/`
- runtime inference stays in `nlu/runtime_*`

## Archive Strategy

Recommended retention logic:

- keep only release-grade demo files in `docs/demo/release/`
- keep historical snapshots in `docs/archive/`
- treat `docs/reports/regression/` as generated output, not hand-maintained documentation
- keep `legacy/` frozen; do not route new features into it
- treat `nlu/bert_lab/models/`, checkpoints, and local configs as local artifacts

## Git Ignore Strategy

The repository benefits from ignoring four classes of generated files:

- LaTeX auxiliaries: `*.aux`, `*.out`, `*.toc`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`
- generated regression trees under `docs/reports/regression/`
- local scratch files such as `tmp_*.txt`
- local release bundles and model/config artifacts

This keeps the tracked history focused on source code, curated docs, and intentional releases.

## Suggested Next Cleanup

If you want one more round after this pass, the next highest-value cleanup would be:

1. split `nlu/bert_lab/` into `nlu/training/` and `nlu/archive/bert_lab/`
2. move active report templates into a small `docs/templates/` area
3. rewrite `docs/architecture/ARCHITECTURE.md` from scratch to remove mojibake and reflect the current runtime path

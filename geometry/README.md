# Assembly DSL Feasibility Prototype

This project is a theory-level feasibility checker for cross-scenario Geant4-like assemblies. It is **not** a Geant4 geometry builder, and it does not run boolean overlap checks. Instead, it provides conservative analytical checks (AABB containment and spacing inequalities) so that an upstream LLM can avoid hallucinating impossible nested structures.

## Design Principles
- No per-scenario template classes. Scenarios are expressed as data using the assembly DSL.
- A finite operator set + constraints paradigm. New scenarios should be expressed by composing existing operators first.
- Add new operators only when you encounter a truly new geometric pattern that cannot be expressed by composition.
- This is a **conservative** validator. A future step can integrate Geant4 `checkOverlaps` as a final judge.

## Structure
- `dsl.py`: DSL nodes and JSON parsing/serialization.
- `geom.py`: AABB helpers and derived spans.
- `feasibility.py`: Analytical feasibility checks and error codes.
- `library.py`: Operator-graph skeletons and sampling spaces.
- `experiments.py`: Coverage, feasibility rate, ambiguity experiments.
- `cli.py`: Command-line interface.
- `examples/`: Small CSV coverage dataset and JSON DSL examples.
- `tests/`: Smoke tests.
- `synthesize.py`: Build DSL from structure + params and run feasibility.

## Structure Diagram
```
          +------------------+
          |  DSL JSON Input  |
          +---------+--------+
                    |
                    v
            +-------+-------+
            |   dsl.py      |
            | parse/validate|
            +-------+-------+
                    |
         +----------+----------+
         |                     |
         v                     v
    +----+----+           +----+----+
    | geom.py |           | library |
    | AABB    |           | skeleton|
    +----+----+           +----+----+
         |                     |
         +----------+----------+
                    v
            +-------+-------+
            | feasibility   |
            | checks + codes|
            +-------+-------+
                    |
                    v
            +-------+-------+
            | experiments   |
            | + cli.py      |
            +---------------+
```

## Minimal Example
Run all experiments and produce outputs in `out/`:

```powershell
python -m geometry.cli run_all --outdir geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset geometry/examples/coverage.csv
```

Expected output files:
- `geometry/out/coverage_summary.json`
- `geometry/out/coverage_checked.jsonl`
- `geometry/out/feasibility_summary.json`
- `geometry/out/ambiguity_summary.json`

Synthesize DSL from a structure + params JSON:

```powershell
python -m geometry.cli synthesize --input geometry/examples/synth_input.json --outdir geometry/out --seed 7
```

Expected output file:
- `geometry/out/synthesis_result.json`

## Extending
- Prefer adding new DSL data instances and operator compositions.
- Only add a new operator when existing ones cannot express the structural pattern.
- Keep analytical checks conservative; do not attempt exact boolean intersections here.

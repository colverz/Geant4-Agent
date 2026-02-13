# Prompt vs BERT Strategy Alignment

## Current Strategy

- BERT is trained on **LLM-normalized controlled English**, not raw user language.
- LLM normalization must output a compact field-like sentence starting with:
  - `geometry_intent: ...`
- If ambiguous, normalization should output:
  - `geometry_intent: unresolved; candidate_pattern: ...`

## Prompt Contract

Implemented in `nlu/bert_lab/llm_bridge.py::build_normalization_prompt`:

- Output JSON only.
- Required keys:
  - `normalized_text`
  - `language_detected`
  - `structure_hint`
- Numeric values and units must be copied exactly.
- No free-form explanation text.

## Runtime Fallback

- In `nlu/bert_lab/semantic.py`:
  - If BERT predicts `unknown` and `structure_hint` is valid, the hint is applied.
  - A note `llm_structure_hint_applied` is recorded.

## Data Generation Consistency

- `nlu/bert_lab/build_controlled_corpus.py` now keeps mandatory core fields per structure
  to avoid over-short normalized samples.

# BERT Lab Active Paths (2026-02)

Use these paths as the current baseline.

- Corpus build:
  - `python -m nlu.bert_lab.build_controlled_corpus --n_structure 12000 --n_ner 16000 --n_multitask 24000 --seed 37`
- Structure data: `nlu/bert_lab/data/controlled_structure.jsonl`
- NER data: `nlu/bert_lab/data/controlled_ner.jsonl`
- Multitask data: `nlu/bert_lab/data/controlled_multitask.jsonl`
- Eval sets:
  - `nlu/bert_lab/data/eval/structure_eval_in_dist.jsonl`
  - `nlu/bert_lab/data/eval/structure_eval_hard.jsonl`
  - `nlu/bert_lab/data/eval/structure_eval_realnorm.jsonl`
- Default structure model priority:
  - `nlu/bert_lab/models/structure_controlled_smoke`
  - (fallbacks only if present)

Legacy assets are archived under `nlu/bert_lab/archive/legacy_2026-02-13/`.

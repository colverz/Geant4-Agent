# Training Preflight (Structure + Multitask)

## 1) What was checked

- Corpus integrity
  - label balance
  - duplicate rate
  - token length distribution
  - span validity (offset range checks)
- Leakage
  - exact normalized-text overlap between train and eval suites
- Evaluation representativeness
  - in-distribution vs hard vs real-normalized suites

## 2) Current generator status

- `nlu/bert_lab/build_controlled_corpus.py`
  - structure corpus is class-balanced and includes `unknown` + hard negatives
  - multitask corpus now comes from `nlu/bert_lab/data_multitask.py`
  - multitask includes entity spans (`material`, `particle`, `physics_list`, `source_type`, `output_format`)
- `nlu/bert_lab/build_eval_suites.py`
  - generates:
    - `structure_eval_in_dist.jsonl`
    - `structure_eval_hard.jsonl`
    - `structure_eval_realnorm.jsonl`
  - removes leakage against the chosen train structure corpus

## 3) Recommended corpus scale for formal use (v1)

- Structure classifier:
  - target: `12k - 20k`
  - include `unknown` at `15% - 20%`
- Multitask (structure + token labels):
  - target: `20k - 40k`
  - keep entity/span coverage balanced by structure class
- Optional NER-only corpus:
  - target: `10k - 20k`

These ranges are enough to enter stable iterative use (not final upper bound).

## 4) Evaluation suite recommendation

Use 3 suites together; do not rely on one single set.

- In-distribution (`structure_eval_in_dist.jsonl`)
  - checks fit to the controlled grammar
- Hard (`structure_eval_hard.jsonl`)
  - checks distractors and ambiguous wording robustness
- Real-normalized (`structure_eval_realnorm.jsonl`)
  - checks behavior on normalized/structured instruction style

## 5) Acceptance gates (suggested)

- Structure:
  - macro-F1 >= 0.85 on in-distribution
  - macro-F1 >= 0.75 on hard
  - macro-F1 >= 0.78 on real-normalized
  - unknown recall >= 0.80
- Multitask:
  - structure accuracy >= 0.88
  - token accuracy (or micro-F1) >= 0.90
  - no critical slot collapse (`material`, `particle`, `physics_list`, `source_type`, `output_format`)

## 6) Commands

Build controlled training corpora:

```powershell
.\.venv\Scripts\python -m nlu.bert_lab.build_controlled_corpus --n_structure 6000 --n_ner 8000 --n_multitask 10000 --seed 37
```

Build 3 evaluation suites:

```powershell
.\.venv\Scripts\python -m nlu.bert_lab.build_eval_suites --outdir nlu/bert_lab/data/eval --train_structure nlu/bert_lab/data/controlled_structure.jsonl --n_in_dist 2400 --n_hard 2400 --n_realnorm 1800 --seed 37
```

Run preflight audit:

```powershell
.\.venv\Scripts\python -m nlu.bert_lab.audit_corpus --train_structure nlu/bert_lab/data/controlled_structure.jsonl --train_multitask nlu/bert_lab/data/controlled_multitask.jsonl --eval_structure nlu/bert_lab/data/eval/structure_eval_in_dist.jsonl --out nlu/bert_lab/data/eval/audit_in_dist.json
```


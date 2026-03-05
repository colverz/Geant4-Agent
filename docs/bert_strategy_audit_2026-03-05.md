# BERT Training Strategy Audit

- Date: `2026-03-05`
- Models dir: `nlu\bert_lab\models`
- Audited models: `6`

## Conclusion
- Status: `not_optimal`
- Best model: `structure_controlled_v4c_e1`
- Weighted score: `1.0000`
- Robustness(min metric): `1.0000`
- Generalization gap: `0.0000`
- Probe accuracy (noisy normalized text): `0.9200`

## Probe Summary
- Probe total: `300`
- Probe hits: `276`
- Probe accuracy: `0.9200`
- Per-label:
  - `grid`: `1.0000`
  - `nest`: `0.8507`
  - `ring`: `1.0000`
  - `shell`: `0.7021`
  - `stack`: `1.0000`
  - `unknown`: `1.0000`

## Ranked Models

| model | in_dist | hard | realnorm | weighted | robustness | gap |
|---|---:|---:|---:|---:|---:|---:|
| structure_controlled_v4c_e1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| structure_controlled_v3_e1 | 1.0000 | 0.8337 | 0.9111 | 0.8852 | 0.8337 | 0.1276 |
| structure_controlled_v4_e1 | 1.0000 | 0.6600 | 1.0000 | 0.8470 | 0.6600 | 0.1700 |
| structure_controlled_v2_e1 | 0.8404 | 0.8500 | 0.6800 | 0.7725 | 0.6800 | 0.0754 |
| structure_controlled_v4b_e1 | 1.0000 | 0.6600 | - | 0.7218 | 0.6600 | 0.3400 |
| structure_controlled_smoke | 0.5162 | 0.6000 | 0.5100 | 0.5511 | 0.5100 | -0.0387 |

## Recommendations
- Probe accuracy on noisy normalized text is below 0.95; add slot-style perturbation augmentation for nest/shell.
- Keep LLM-first normalization; train BERT on normalized-text styles instead of free-form text.

## Suggested Next Run
- Build normalized-style v2 dataset:
  - `python scripts/build_structure_v2_dataset.py --keep_original`
- Train structure model with v2 profile:
  - `python scripts/train_structure_v2.py --data nlu/bert_lab/data/controlled_structure_v2.jsonl --outdir nlu/bert_lab/models/structure_controlled_v5_e2 --config nlu/bert_lab/configs/structure_train_v2.json`
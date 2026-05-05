# Archived BERT-Lab Tools

This folder contains older training, evaluation, and dataset utility entrypoints that are not part of the active strict runtime path.

Typical examples moved here:

- training scripts (`train_structure.py`, `train_ner.py`, `train_multitask.py`)
- evaluation runners (`run_workflow_e2e.py`, `run_workflow_e2e_lite.py`, `run_regression_3class.py`)
- corpus and normalization utilities (`build_controlled_corpus.py`, `build_eval_suites.py`, `normalize_dataset_with_llm.py`)

These scripts are kept for reference. They may still import compatibility shims
under `nlu/bert_lab/`, but the active web/runtime path should not depend on
that directory directly. Runtime code should use `nlu/runtime_components`,
`nlu/llm_support`, and `nlu/training/bert_lab` as the source of truth.

Compatibility shims retained for these archived scripts include:

- `semantic.py`
- `ollama_client.py`
- `llm_bridge.py`
- `graph_search.py`
- `infer.py`
- `postprocess.py`

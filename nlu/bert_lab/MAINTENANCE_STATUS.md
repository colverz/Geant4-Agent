# BERT Lab Maintenance Status

This directory is a compatibility and lab-facing path.

Active paths:

- Runtime extraction: `nlu/bert/`
- Training utilities: `nlu/training/bert_lab/`
- Legacy runtime reference: `legacy/runtime/bert_lab/`

Several files in this directory intentionally re-export from active or legacy
locations so older imports keep working.

Do not add new runtime behavior here. New runtime extraction work should target
`nlu/bert/`; new training utilities should target `nlu/training/bert_lab/`.

Before removing or moving files from this directory, run an import audit and the
full regression suite.

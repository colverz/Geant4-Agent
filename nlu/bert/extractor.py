from __future__ import annotations

"""Compatibility wrapper for the old NLP-BERT extractor import path.

The active implementation lives in :mod:`nlu.runtime_extractor`. Keep this
module thin so new production code does not grow a dependency on the old
``nlu.bert`` namespace.
"""

from nlu.runtime_extractor import (
    extract_candidates_from_normalized_text,
    _infer_material,
    _infer_particle,
    _parse_energy_mev,
    _parse_module_triplet_mm,
)

__all__ = [
    "extract_candidates_from_normalized_text",
    "_infer_material",
    "_infer_particle",
    "_parse_energy_mev",
    "_parse_module_triplet_mm",
]

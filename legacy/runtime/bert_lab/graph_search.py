from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from builder.geometry.library import SKELETONS
from builder.geometry.synthesize import synthesize_from_params


SKELETON_SUMMARY: Dict[str, str] = {
    "nest_box_tubs": "nest",
    "grid_modules": "grid",
    "ring_modules": "ring",
    "stack_in_box": "stack",
    "shell_nested": "shell",
    "single_box": "single_box",
    "single_tubs": "single_tubs",
    "single_sphere": "single_sphere",
    "single_orb": "single_orb",
    "single_cons": "single_cons",
    "single_trd": "single_trd",
    "single_polycone": "single_polycone",
    "single_cuttubs": "single_cuttubs",
    "boolean_union_boxes": "boolean",
    "boolean_subtraction_boxes": "boolean",
    "boolean_intersection_boxes": "boolean",
    "tilted_box_in_parent": "nest",
}

KEYWORD_CUES: Dict[str, Tuple[str, ...]] = {
    "ring": ("ring", "circular", "annulus", "circle", "radius"),
    "grid": ("grid", "array", "matrix", "pitch", "nx", "ny"),
    "nest": ("nest", "inside", "contains", "contain", "inner", "outer"),
    "stack": ("stack", "layer", "layers", "along z"),
    "shell": ("shell", "concentric", "thickness", "coaxial"),
    "single_box": ("single_box", "box", "cube", "cuboid"),
    "single_tubs": ("single_tubs", "tubs", "cylinder", "tube"),
    "single_sphere": ("single_sphere", "sphere", "ball"),
    "single_orb": ("single_orb", "orb"),
    "single_cons": ("single_cons", "cons", "cone", "frustum"),
    "single_trd": ("single_trd", "trd", "trapezoid", "trapezoidal"),
    "single_polycone": ("single_polycone", "polycone", "z planes"),
    "single_cuttubs": ("single_cuttubs", "cuttubs", "cut tubs"),
    "boolean": ("boolean", "union", "subtraction", "intersection"),
}

AMBIGUITY_CUES = ("ambiguous", "undecided", "unresolved", "not fixed")


@dataclass(frozen=True)
class CandidateGraph:
    structure: str
    summary: str
    score: float
    score_breakdown: Dict[str, float]
    feasible: bool
    missing_params: List[str]
    errors: List[str]
    warnings: List[str]
    dsl: Dict[str, Any]
    params_filled: Dict[str, float]


@dataclass(frozen=True)
class GraphSearchResult:
    structure: str
    chosen_skeleton: str
    graph_program: Dict[str, Any] | None
    scores: Dict[str, float]
    ranked: List[Tuple[str, float]]
    candidates: List[CandidateGraph]
    notes: List[str]


def _softmax(values: Dict[str, float]) -> Dict[str, float]:
    m = max(values.values()) if values else 0.0
    ex = {k: math.exp(v - m) for k, v in values.items()}
    s = sum(ex.values()) or 1.0
    return {k: ex[k] / s for k in values}


def _cue_score(text: str, structure: str) -> float:
    t = text.lower()
    score = 0.0
    for kw in KEYWORD_CUES.get(structure, ()):
        needle = kw.lower()
        if " " in needle:
            if needle in t:
                score += 0.32
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", t):
            score += 0.32
    return score


def _explicit_structure_hint(text: str) -> str:
    m = re.search(
        r"(?:^|[;\s])structure\s*[:=]\s*(ring|grid|nest|stack|shell|single_box|single_tubs|single_sphere|single_orb|single_cons|single_trd|single_polycone|single_cuttubs|boolean)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    return ""


def _coverage_score(params: Dict[str, float], structure: str) -> float:
    required = tuple(k for sk in SKELETONS if sk.name == structure for k in sk.param_keys)
    if not required:
        return 0.0
    hit = sum(1 for k in required if k in params)
    return hit / len(required)


def _build_candidate(structure: str, params: Dict[str, float], seed: int, apply_autofix: bool) -> CandidateGraph:
    synth = synthesize_from_params(structure, params, seed=seed, apply_autofix=apply_autofix)
    return CandidateGraph(
        structure=structure,
        summary=SKELETON_SUMMARY.get(structure, structure),
        score=0.0,
        score_breakdown={},
        feasible=bool(synth.get("feasible", False)),
        missing_params=list(synth.get("missing_params", [])),
        errors=list(synth.get("errors", [])),
        warnings=list(synth.get("warnings", [])),
        dsl=dict(synth.get("dsl", {})),
        params_filled=dict(synth.get("params_filled", {})),
    )


def _score_candidate(
    text: str,
    params: Dict[str, float],
    candidate: CandidateGraph,
    explicit_hint: str,
) -> Tuple[float, Dict[str, float]]:
    coverage = _coverage_score(params, candidate.structure)
    cue = _cue_score(text, candidate.summary)
    feasible_term = 0.85 if candidate.feasible else -0.65
    missing_term = -0.12 * len(candidate.missing_params)
    error_term = -0.08 * len(candidate.errors)
    hint_term = 0.0
    if explicit_hint:
        hint_term = 1.15 if candidate.summary == explicit_hint else -0.18

    total = 2.1 * coverage + cue + feasible_term + missing_term + error_term + hint_term
    return total, {
        "coverage": 2.1 * coverage,
        "cue": cue,
        "feasible": feasible_term,
        "missing": missing_term,
        "errors": error_term,
        "hint": hint_term,
        "total": total,
    }


def search_candidate_graphs(
    text: str,
    params: Dict[str, float],
    *,
    min_confidence: float = 0.6,
    seed: int = 7,
    top_k: int = 3,
    apply_autofix: bool = False,
) -> GraphSearchResult:
    notes: List[str] = []
    explicit_hint = _explicit_structure_hint(text)
    if explicit_hint:
        notes.append(f"explicit_structure_hint:{explicit_hint}")

    structures = [s.name for s in SKELETONS]
    core_skeletons = {
        "nest_box_tubs",
        "grid_modules",
        "ring_modules",
        "stack_in_box",
        "shell_nested",
        "single_box",
        "single_tubs",
        "single_sphere",
        "single_orb",
        "single_cons",
        "single_trd",
        "single_polycone",
        "single_cuttubs",
        "boolean_union_boxes",
        "boolean_subtraction_boxes",
        "boolean_intersection_boxes",
    }
    scored_candidates: List[CandidateGraph] = []
    raw_scores_by_summary: Dict[str, float] = {}
    for structure in structures:
        summary = SKELETON_SUMMARY.get(structure, structure)
        cov_probe = _coverage_score(params, structure)
        cue_probe = _cue_score(text, summary)
        if (
            structure not in core_skeletons
            and cov_probe < 0.34
            and cue_probe < 0.30
            and explicit_hint != summary
        ):
            continue
        c = _build_candidate(structure, params, seed=seed, apply_autofix=apply_autofix)
        score, breakdown = _score_candidate(text, params, c, explicit_hint)
        c2 = CandidateGraph(
            structure=c.structure,
            summary=c.summary,
            score=score,
            score_breakdown=breakdown,
            feasible=c.feasible,
            missing_params=c.missing_params,
            errors=c.errors,
            warnings=c.warnings,
            dsl=c.dsl,
            params_filled=c.params_filled,
        )
        scored_candidates.append(c2)
        raw_scores_by_summary[c2.summary] = max(raw_scores_by_summary.get(c2.summary, -1e9), score)

    unknown_raw = 0.18
    tl = text.lower()
    if any(cue in tl for cue in AMBIGUITY_CUES):
        unknown_raw += 1.0
    raw_scores_by_summary["unknown"] = unknown_raw

    probs = _softmax(raw_scores_by_summary)
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    best_label, best_prob = ranked[0]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

    chosen = best_label
    low_conf = best_prob < 0.30
    low_margin = (best_prob - second_prob) < 0.02 and best_prob < 0.45
    if best_label != "unknown" and (low_conf or low_margin):
        chosen = "unknown"
    if best_label == "unknown":
        chosen = "unknown"

    scores = dict(probs)
    scores["best_prob"] = float(best_prob)
    scores["second_prob"] = float(second_prob)
    scores["margin"] = float(best_prob - second_prob)

    candidates_sorted = sorted(scored_candidates, key=lambda c: c.score, reverse=True)
    chosen_candidate = next((c for c in candidates_sorted if c.summary == chosen), None) if chosen != "unknown" else None
    candidates_topk = candidates_sorted[: max(1, top_k)]
    return GraphSearchResult(
        structure=chosen,
        chosen_skeleton=chosen_candidate.structure if chosen_candidate else "",
        graph_program=chosen_candidate.dsl if chosen_candidate else None,
        scores=scores,
        ranked=ranked,
        candidates=candidates_topk,
        notes=notes,
    )

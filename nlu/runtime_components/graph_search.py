from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from builder.geometry.library import SKELETONS
from builder.geometry.synthesize import synthesize_from_params


SKELETON_SUMMARY: Dict[str, str] = {
    "nest_box_box": "nest",
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
    "single_trap": "single_trap",
    "single_para": "single_para",
    "single_torus": "single_torus",
    "single_ellipsoid": "single_ellipsoid",
    "single_elltube": "single_elltube",
    "single_polyhedra": "single_polyhedra",
    "boolean_union_boxes": "boolean",
    "boolean_subtraction_boxes": "boolean",
    "boolean_intersection_boxes": "boolean",
    "tilted_box_in_parent": "nest",
}

KEYWORD_CUES: Dict[str, Tuple[str, ...]] = {
    "ring": ("ring", "circular", "annulus", "circle", "环", "环形", "圆环", "围成一圈", "一圈"),
    "grid": ("grid", "array", "matrix", "pitch", "nx", "ny", "阵列", "二维阵列", "探测板", "网格", "排布"),
    "nest": ("nest", "inside", "contains", "contain", "inner", "outer", "嵌套", "内嵌", "盒子里", "外盒", "外盒体", "内盒体", "包住", "包裹"),
    "stack": ("stack", "layer", "layers", "along z", "stacked", "sandwich", "堆叠", "层", "夹层", "沿 z 方向", "沿z方向", "层厚"),
    "shell": ("shell", "concentric", "thickness", "coaxial", "壳", "屏蔽壳", "同心", "屏蔽层", "多层壳", "外壳"),
    "single_box": ("single_box", "box", "cube", "cuboid"),
    "single_tubs": ("single_tubs", "tubs", "cylinder", "tube"),
    "single_sphere": ("single_sphere", "sphere", "ball"),
    "single_orb": ("single_orb", "orb"),
    "single_cons": ("single_cons", "cons", "cone", "frustum"),
    "single_trd": ("single_trd", "trd", "trapezoid", "trapezoidal"),
    "single_polycone": ("single_polycone", "polycone", "z planes"),
    "single_cuttubs": ("single_cuttubs", "cuttubs", "cut tubs"),
    "single_trap": ("single_trap", "trap", "trapezoid prism"),
    "single_para": ("single_para", "para", "parallelepiped", "skewed box"),
    "single_torus": ("single_torus", "torus", "donut", "ring tube"),
    "single_ellipsoid": ("single_ellipsoid", "ellipsoid", "elliptic"),
    "single_elltube": ("single_elltube", "elliptical tube", "ellipse tube", "elliptic tube"),
    "single_polyhedra": ("single_polyhedra", "polyhedra", "polyhedron"),
    "boolean": (
        "boolean",
        "union",
        "subtraction",
        "intersection",
        "subtract",
        "minus",
        "difference",
        "hole",
        "cutout",
        "cut out",
        "减去",
        "差集",
        "并",
        "并集",
        "合并",
        "挖空",
        "开孔",
    ),
}

BOOLEAN_OP_CUES: Dict[str, Tuple[str, ...]] = {
    "boolean_union_boxes": ("union", "combine", "merge", "并", "并集", "合并"),
    "boolean_subtraction_boxes": (
        "subtraction",
        "subtract",
        "minus",
        "difference",
        "hole",
        "cutout",
        "cut out",
        "减去",
        "差集",
        "挖空",
        "开孔",
        "打孔",
    ),
    "boolean_intersection_boxes": ("intersection", "intersect", "overlap", "交", "相交"),
}

AMBIGUITY_CUES = ("ambiguous", "undecided", "unresolved", "not fixed")
GRAPH_SUMMARIES = {"ring", "grid", "nest", "stack", "shell", "boolean"}


def _minus_cue_is_boolean_operator(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search(r"\b(?:box|cube|cuboid|cylinder|sphere|solid|target)\b.{0,80}\bminus\b.{0,80}\b(?:box|cube|cuboid|cylinder|sphere|solid|target)\b", low)
    )


def _contains_keyword_cue(text: str, keyword: str) -> bool:
    t = text.lower()
    needle = keyword.lower()
    if needle == "minus":
        return _minus_cue_is_boolean_operator(t)
    if " " in needle:
        return needle in t
    if any(ord(ch) > 127 for ch in needle):
        return needle in t
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", t))


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
    score = 0.0
    for kw in KEYWORD_CUES.get(structure, ()):
        if _contains_keyword_cue(text, kw):
            score += 0.32
    return score


def _graph_family_cue(text: str) -> str:
    scored = {
        summary: _cue_score(text, summary)
        for summary in ("boolean", "ring", "grid", "stack", "shell", "nest")
    }
    best_summary = ""
    best_score = 0.0
    for summary, score in scored.items():
        if score > best_score:
            best_summary = summary
            best_score = score
    return best_summary


def _param_signature_hint(params: Dict[str, float]) -> str:
    has_module_triplet = all(key in params for key in ("module_x", "module_y", "module_z"))
    has_parent_triplet = all(key in params for key in ("parent_x", "parent_y", "parent_z"))
    has_child_triplet = all(key in params for key in ("child_x", "child_y", "child_z"))
    has_bool_triplet = all(key in params for key in ("bool_a_x", "bool_a_y", "bool_a_z", "bool_b_x", "bool_b_y", "bool_b_z"))
    has_shell = all(key in params for key in ("inner_r", "th1", "th2"))
    has_stack = all(key in params for key in ("stack_x", "stack_y", "t1", "t2"))
    has_ring = has_module_triplet and all(key in params for key in ("n", "radius"))
    maybe_ring = has_module_triplet and "radius" in params
    has_grid = has_module_triplet and all(key in params for key in ("nx", "ny"))
    has_nest = has_parent_triplet and (has_child_triplet or all(key in params for key in ("child_rmax", "child_hz")))

    if has_bool_triplet:
        return "boolean"
    if has_grid:
        return "grid"
    if has_stack:
        return "stack"
    if has_ring:
        return "ring"
    if has_nest:
        return "nest"
    if has_shell:
        return "shell"
    if maybe_ring:
        return "ring"
    return ""


def _graph_lock_hint(text: str, params: Dict[str, float]) -> str:
    low = text.lower()
    has_module_triplet = all(key in params for key in ("module_x", "module_y", "module_z"))
    has_parent_triplet = all(key in params for key in ("parent_x", "parent_y", "parent_z"))
    has_child_triplet = all(key in params for key in ("child_x", "child_y", "child_z"))
    has_tubs_child = all(key in params for key in ("child_rmax", "child_hz"))
    has_bool_triplet = all(key in params for key in ("bool_a_x", "bool_a_y", "bool_a_z", "bool_b_x", "bool_b_y", "bool_b_z"))
    has_shell = all(key in params for key in ("inner_r", "th1", "th2", "hz"))
    has_stack = all(key in params for key in ("stack_x", "stack_y", "t1", "t2"))
    has_grid = has_module_triplet and all(key in params for key in ("nx", "ny", "pitch_x", "pitch_y"))
    has_ring = has_module_triplet and "radius" in params and ("n" in params or any(token in low for token in ("ring", "annulus", "circular", "pet", "环", "圆环")))
    has_nest = has_parent_triplet and (has_child_triplet or has_tubs_child)
    has_boolean = has_bool_triplet and any(_contains_keyword_cue(low, token) for token in KEYWORD_CUES.get("boolean", ()))

    if has_boolean:
        return "boolean"
    if has_stack:
        return "stack"
    if has_grid:
        return "grid"
    if has_nest:
        return "nest"
    if has_shell:
        return "shell"
    if has_ring:
        return "ring"
    return ""


def _explicit_structure_hint(text: str) -> str:
    m = re.search(
        r"(?:^|[;\s])structure\s*[:=]\s*(ring|grid|nest|stack|shell|single_box|single_tubs|single_sphere|single_orb|single_cons|single_trd|single_polycone|single_cuttubs|single_trap|single_para|single_torus|single_ellipsoid|single_elltube|single_polyhedra|boolean)\b",
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
    prior_summary: str = "",
    prior_confidence: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    coverage = _coverage_score(params, candidate.structure)
    cue = _cue_score(text, candidate.summary)
    feasible_term = 0.85 if candidate.feasible else -0.65
    missing_term = -0.12 * len(candidate.missing_params)
    error_term = -0.08 * len(candidate.errors)
    hint_term = 0.0
    if explicit_hint:
        hint_term = 1.15 if candidate.summary == explicit_hint else -0.18
    prior_term = 0.0
    if prior_summary and prior_summary != "unknown" and prior_confidence > 0.0:
        capped_conf = min(max(float(prior_confidence), 0.0), 1.0)
        if candidate.summary == prior_summary:
            prior_term = (1.05 if prior_summary in GRAPH_SUMMARIES else 0.55) * capped_conf
        elif prior_summary in GRAPH_SUMMARIES and candidate.summary.startswith("single_"):
            prior_term = -0.15 * capped_conf

    boolean_op_term = 0.0
    if candidate.structure in BOOLEAN_OP_CUES:
        boolean_low = text.lower()
        matched = any(_contains_keyword_cue(boolean_low, cue) for cue in BOOLEAN_OP_CUES[candidate.structure])
        any_boolean_op = any(
            _contains_keyword_cue(boolean_low, cue)
            for cues in BOOLEAN_OP_CUES.values()
            for cue in cues
        )
        if matched:
            boolean_op_term = 0.65
        elif any_boolean_op:
            boolean_op_term = -0.10

    graph_bias = 0.0
    graph_cue = _graph_family_cue(text)
    if graph_cue:
        if candidate.summary == graph_cue:
            graph_bias += 0.95
        elif candidate.summary.startswith("single_"):
            graph_bias -= 0.85
        elif candidate.summary in GRAPH_SUMMARIES and candidate.summary != graph_cue:
            graph_bias -= 0.12

    signature_bias = 0.0
    signature_hint = _param_signature_hint(params)
    if signature_hint:
        if candidate.summary == signature_hint:
            signature_bias += 1.35
        elif candidate.summary.startswith("single_"):
            signature_bias -= 0.95
        elif candidate.summary in GRAPH_SUMMARIES and candidate.summary != signature_hint:
            signature_bias -= 0.18

    graph_lock_bias = 0.0
    graph_lock_hint = _graph_lock_hint(text, params)
    if graph_lock_hint:
        if candidate.summary == graph_lock_hint:
            graph_lock_bias += 2.1
        elif candidate.summary.startswith("single_"):
            graph_lock_bias -= 1.6
        elif candidate.summary in GRAPH_SUMMARIES and candidate.summary != graph_lock_hint:
            graph_lock_bias -= 0.35

    total = (
        2.1 * coverage
        + cue
        + feasible_term
        + missing_term
        + error_term
        + hint_term
        + prior_term
        + boolean_op_term
        + graph_bias
        + signature_bias
        + graph_lock_bias
    )
    return total, {
        "coverage": 2.1 * coverage,
        "cue": cue,
        "feasible": feasible_term,
        "missing": missing_term,
        "errors": error_term,
        "hint": hint_term,
        "structure_prior": prior_term,
        "boolean_op": boolean_op_term,
        "graph_bias": graph_bias,
        "signature_bias": signature_bias,
        "graph_lock_bias": graph_lock_bias,
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
    prior_summary: str = "",
    prior_confidence: float = 0.0,
) -> GraphSearchResult:
    notes: List[str] = []
    explicit_hint = _explicit_structure_hint(text)
    if explicit_hint:
        notes.append(f"explicit_structure_hint:{explicit_hint}")
    graph_lock_hint = _graph_lock_hint(text, params)
    if graph_lock_hint:
        notes.append(f"graph_lock_hint:{graph_lock_hint}")

    structures = [s.name for s in SKELETONS]
    core_skeletons = {
        "nest_box_box",
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
        "single_trap",
        "single_para",
        "single_torus",
        "single_ellipsoid",
        "single_elltube",
        "single_polyhedra",
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
        score, breakdown = _score_candidate(
            text,
            params,
            c,
            explicit_hint,
            prior_summary=prior_summary,
            prior_confidence=prior_confidence,
        )
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
    if graph_lock_hint and chosen.startswith("single_"):
        locked_prob = probs.get(graph_lock_hint, 0.0)
        if locked_prob >= 0.12:
            chosen = graph_lock_hint

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

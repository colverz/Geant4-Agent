from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session


INTERNAL_FIELD_RE = re.compile(r"\b[a-z]+(?:\.[a-z_]+)+\b")
TEMPLATE_RE = re.compile(r"(please provide|please confirm|still needs|ask for|missing field)", re.IGNORECASE)
PHYSICS_ITEMS = ["FTFP_BERT", "QBBC", "QGSP_BERT", "QGSP_BERT_HP", "Shielding"]
OUTPUT_ITEMS = ["root", "csv", "hdf5", "xml", "json"]
CLARIFICATION_ACTIONS = {"ask_clarification", "summarize_progress", "confirm_overwrite"}
CONTRADICTION_ACTIONS = {"confirm_overwrite", "ask_clarification"}


@dataclass
class Scenario:
    sid: str
    category: str
    domain: str
    title: str
    turns: list[str]
    expected: dict[str, Any]
    tags: set[str] = field(default_factory=set)


def _deep_get(cfg: dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
    return cur


def _find_mentions(text: str, items: list[str]) -> list[str]:
    low = (text or "").lower()
    found: list[str] = []
    for item in items:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(item.lower())}(?![A-Za-z0-9_])")
        if pat.search(low):
            found.append(item)
    return found


def _grounding_mismatch(msg: str, cfg: dict[str, Any]) -> bool:
    physics_actual = str(_deep_get(cfg, "physics.physics_list") or "")
    output_actual = str(_deep_get(cfg, "output.format") or "")
    p_mentions = _find_mentions(msg, PHYSICS_ITEMS)
    o_mentions = _find_mentions(msg, OUTPUT_ITEMS)
    if physics_actual and p_mentions and all(m.lower() != physics_actual.lower() for m in p_mentions):
        return True
    if output_actual and o_mentions and all(m.lower() != output_actual.lower() for m in o_mentions):
        return True
    return False


def _score_case(scenario: Scenario, final: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    hit = 0
    total = 0
    final_missing = list(final.get("missing_fields", []))
    final_cfg = dict(final.get("config", {}))
    for key, want in scenario.expected.items():
        total += 1
        if key == "complete":
            got = bool(final.get("is_complete", False))
            ok = got == bool(want)
        elif key == "has_action":
            got = bool(str(want) in actions)
            ok = got
        elif key == "has_any_action":
            target = [str(x) for x in list(want)]
            got = [x for x in actions if x in target]
            ok = bool(got)
        elif key == "missing_at_least":
            got = len(final_missing)
            ok = got >= int(want)
        else:
            got = _deep_get(final_cfg, key)
            ok = got == want
        if ok:
            hit += 1
        checks.append({"field": key, "expected": want, "actual": got, "ok": ok})
    return {"hits": hit, "total": total, "accuracy": (hit / total) if total else 0.0, "pass": hit == total, "checks": checks}


def _base_scenarios() -> list[Scenario]:
    return [
        Scenario(
            sid="S01",
            category="full_closure",
            domain="Geometry+Source+Physics+Output",
            title="Single-turn full closure",
            turns=[
                "Need a 1m x 1m x 1m copper box, gamma point source 1 MeV at center along +z, physics FTFP_BERT, output json."
            ],
            expected={
                "complete": True,
                "geometry.structure": "single_box",
                "source.type": "point",
                "source.particle": "gamma",
                "physics.physics_list": "FTFP_BERT",
                "output.format": "json",
            },
        ),
        Scenario(
            sid="S02",
            category="geometry_family",
            domain="Geometry family",
            title="Two-turn tubs setup",
            turns=[
                "Need a copper cylinder, radius 30 mm, half-length 50 mm, gamma point source 2 MeV at (0,0,-100) toward (0,0,1).",
                "Use QBBC, output json.",
            ],
            expected={"complete": True, "geometry.structure": "single_tubs", "physics.physics_list": "QBBC", "output.format": "json"},
        ),
        Scenario(
            sid="S03",
            category="overwrite_control",
            domain="Overwrite safety",
            title="Confirm overwrite flow",
            turns=[
                "Copper cube 1m x 1m x 1m, gamma point 1 MeV at center +z, physics FTFP_BERT, output json.",
                "Change material to G4_Al.",
                "Confirm.",
            ],
            expected={"complete": True, "materials.selected_materials": ["G4_Al"], "has_action": "confirm_overwrite"},
            tags={"overwrite_guard"},
        ),
        Scenario(
            sid="S04",
            category="question_safety",
            domain="Question turn behavior",
            title="Question should not overwrite",
            turns=[
                "1m copper box, gamma point source 1 MeV at origin along +z, physics FTFP_BERT, output json.",
                "what if proton?",
            ],
            expected={"source.particle": "gamma", "complete": True},
            tags={"overwrite_guard"},
        ),
        Scenario(
            sid="S05",
            category="source_mode",
            domain="Source mode",
            title="Beam source setup",
            turns=[
                "Silicon cube 50 cm each side, beam gamma source 3 MeV at (0,0,-200) direction (0,0,1), physics QBBC, output root."
            ],
            expected={"complete": True, "source.type": "beam", "output.format": "root"},
        ),
        Scenario(
            sid="S06",
            category="source_mode",
            domain="Source mode",
            title="Isotropic source setup",
            turns=["Water cube 40 cm each side, isotropic gamma source 1.2 MeV at center, physics QBBC, output hdf5."],
            expected={"complete": True, "source.type": "isotropic", "output.format": "hdf5"},
        ),
        Scenario(
            sid="S07",
            category="consultative",
            domain="Consultative flow",
            title="Consult then commit",
            turns=[
                "This is pure gamma attenuation in copper, no hadrons. Recommend a Geant4 physics list and explain briefly.",
                "Adopt your recommendation. Geometry is 1m copper cube, gamma point 1MeV at center along +z, output json.",
            ],
            expected={"complete": True, "physics.physics_list": "FTFP_BERT", "output.format": "json"},
        ),
        Scenario(
            sid="S08",
            category="output_update",
            domain="Output update",
            title="Output-only change",
            turns=[
                "Copper cube 1m x 1m x 1m, gamma point 1 MeV at center +z, physics FTFP_BERT, output json.",
                "Change output to root.",
            ],
            expected={"complete": True, "output.format": "root", "source.particle": "gamma"},
        ),
    ]


def _fuzzy_scenarios() -> list[Scenario]:
    return [
        Scenario(
            sid="F01",
            category="fuzzy_language",
            domain="Fuzzy language",
            title="Colloquial two-turn completion",
            turns=[
                "I want gamma attenuation in copper, roughly one-meter cube, source in the center forward.",
                "Energy 1MeV, direction +z, physics FTFP_BERT, output json.",
            ],
            expected={"geometry.structure": "single_box", "source.particle": "gamma", "physics.physics_list": "FTFP_BERT", "output.format": "json"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F02",
            category="fuzzy_language",
            domain="Fuzzy language",
            title="Code-switch shorthand",
            turns=["target: Cu cube 1000mm^3 each side; src gamma point E=1MeV pos(0,0,0) dir(0,0,1); physics=FTFP_BERT; output=root"],
            expected={"complete": True, "geometry.structure": "single_box", "source.type": "point", "output.format": "root"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F03",
            category="fuzzy_contradiction",
            domain="Contradiction handling",
            title="Contradictory rewrite in one turn",
            turns=[
                "Copper cube 1m x 1m x 1m, gamma point 1 MeV at center +z, physics FTFP_BERT, output json.",
                "Change geometry to cylinder radius 30mm, but keep the 1m cube unchanged.",
            ],
            expected={"geometry.structure": "single_box", "has_any_action": ["confirm_overwrite", "ask_clarification"]},
            tags={"fuzzy", "contradiction", "overwrite_guard"},
        ),
        Scenario(
            sid="F04",
            category="fuzzy_question",
            domain="Question safety",
            title="Fuzzy question should not overwrite",
            turns=[
                "Setup 1m copper cube, gamma 1MeV point source at center +z, FTFP_BERT, json output.",
                "Would proton be better?",
            ],
            expected={"source.particle": "gamma", "has_any_action": ["explain_choice", "answer_status", "summarize_progress"]},
            tags={"fuzzy", "overwrite_guard"},
        ),
        Scenario(
            sid="F05",
            category="fuzzy_incomplete",
            domain="Novice request",
            title="Incomplete novice input",
            turns=["Can you set up a copper gamma test and give me a runnable baseline?"],
            expected={"complete": False, "missing_at_least": 1, "has_any_action": ["ask_clarification", "summarize_progress", "answer_status"]},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F06",
            category="fuzzy_units",
            domain="Unit ambiguity",
            title="Mixed units in one sentence",
            turns=["Copper target 100cm x 1000mm x 1m, gamma point source 1000keV at center along +z, physics FTFP_BERT, output json."],
            expected={"source.particle": "gamma", "output.format": "json"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F07",
            category="fuzzy_output",
            domain="Output update",
            title="Colloquial output rewrite",
            turns=[
                "Copper cube 1m x 1m x 1m, gamma point 1 MeV at center +z, physics FTFP_BERT, output json.",
                "I want the ROOT tree style output file.",
            ],
            expected={"output.format": "root"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F08",
            category="fuzzy_consultative",
            domain="Consultative flow",
            title="Consultative with fuzzy wording",
            turns=[
                "I only care about gamma attenuation in copper and no hadrons. Recommend a safe Geant4 physics list.",
                "Use your recommendation. Geometry 1m copper cube, gamma point 1MeV at center +z, output json.",
            ],
            expected={"complete": True, "output.format": "json"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F09",
            category="fuzzy_multimodal_stub",
            domain="Pseudo multimodal text",
            title="Multimodal placeholder text",
            turns=["Follow this sketch: [image: copper cube, center source, +z direction], build config first."],
            expected={"complete": False, "missing_at_least": 1, "has_any_action": ["ask_clarification", "answer_status", "summarize_progress"]},
            tags={"fuzzy", "multimodal_stub"},
        ),
    ]


def _run_scenario(s: Scenario, config_path: str, lang: str, min_confidence: float) -> dict[str, Any]:
    session_id = f"latex-{s.sid}"
    reset_session(session_id)
    turn_logs: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    actions: list[str] = []

    for idx, text in enumerate(s.turns, start=1):
        out = process_turn(
            {
                "session_id": session_id,
                "text": text,
                "llm_router": True,
                "llm_question": True,
                "normalize_input": True,
                "min_confidence": min_confidence,
            },
            ollama_config_path=config_path,
            min_confidence=min_confidence,
            lang=lang,
        )
        final = out
        action = str(out.get("dialogue_action", ""))
        actions.append(action)
        turn_logs.append(
            {
                "turn": idx,
                "user": text,
                "assistant": out.get("assistant_message", ""),
                "dialogue_action": action,
                "asked_fields_friendly": list(out.get("asked_fields_friendly", [])),
                "missing_fields": list(out.get("missing_fields", [])),
                "normalization": dict(out.get("normalization", {})),
                "llm_used": bool(out.get("llm_used", False)),
                "fallback_reason": out.get("fallback_reason"),
                "llm_stage_failures": list(out.get("llm_stage_failures", [])),
                "rejected_updates": list(out.get("rejected_updates", []))[:8],
                "applied_rules": list(out.get("applied_rules", []))[:8],
                "grounding_mismatch": _grounding_mismatch(str(out.get("assistant_message", "")), dict(out.get("config", {}))),
            }
        )

    assert final is not None
    score = _score_case(s, final, actions)
    raw_dialogue = list(final.get("raw_dialogue", []))
    msgs = [str(x.get("assistant", "")) for x in turn_logs]
    leaks = sum(1 for m in msgs if INTERNAL_FIELD_RE.search(m))
    templates = sum(1 for m in msgs if TEMPLATE_RE.search(m))
    mismatch_count = sum(1 for x in turn_logs if x["grounding_mismatch"])
    n = len(msgs)
    naturalness = {
        "assistant_turns": n,
        "internal_field_leak_rate": leaks / n if n else 0.0,
        "template_phrase_rate": templates / n if n else 0.0,
        "grounding_mismatch_rate": mismatch_count / n if n else 0.0,
        "naturalness_score": max(0.0, 1.0 - (0.4 * (leaks / n if n else 0.0) + 0.2 * (templates / n if n else 0.0) + 0.4 * (mismatch_count / n if n else 0.0))),
    }
    clarification_turns = sum(1 for t in turn_logs if t.get("dialogue_action") in CLARIFICATION_ACTIONS)
    contradiction_detected = bool(CONTRADICTION_ACTIONS.intersection(set(actions)))

    reset_session(session_id)
    return {
        "id": s.sid,
        "category": s.category,
        "domain": s.domain,
        "title": s.title,
        "expected": s.expected,
        "tags": sorted(list(s.tags)),
        "score": score,
        "naturalness": naturalness,
        "turns": turn_logs,
        "raw_dialogue": raw_dialogue,
        "final_config": final.get("config", {}),
        "final_missing": list(final.get("missing_fields", [])),
        "is_complete": bool(final.get("is_complete", False)),
        "clarification_turns": clarification_turns,
        "actions": actions,
        "contradiction_detected": contradiction_detected,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    pass_cnt = sum(1 for r in results if r["score"]["pass"])
    acc = sum(float(r["score"]["accuracy"]) for r in results) / total if total else 0.0
    nat = sum(float(r["naturalness"]["naturalness_score"]) for r in results) / total if total else 0.0
    mismatch = sum(float(r["naturalness"]["grounding_mismatch_rate"]) for r in results) / total if total else 0.0
    closure_cnt = sum(1 for r in results if r["is_complete"])
    total_turns = sum(len(r["turns"]) for r in results)
    total_clarification = sum(int(r["clarification_turns"]) for r in results)

    overwrite_cases = [r for r in results if "overwrite_guard" in set(r.get("tags", []))]
    contradiction_cases = [r for r in results if "contradiction" in set(r.get("tags", []))]
    fuzzy_cases = [r for r in results if "fuzzy" in set(r.get("tags", []))]

    return {
        "total_scenarios": total,
        "strict_pass_rate": pass_cnt / total if total else 0.0,
        "avg_accuracy": acc,
        "avg_naturalness": nat,
        "avg_grounding_mismatch_rate": mismatch,
        "closure_rate": closure_cnt / total if total else 0.0,
        "avg_turns_per_scenario": (total_turns / total) if total else 0.0,
        "clarification_turn_ratio": (total_clarification / total_turns) if total_turns else 0.0,
        "overwrite_guard_pass_rate": (
            sum(1 for r in overwrite_cases if r["score"]["pass"]) / len(overwrite_cases) if overwrite_cases else 0.0
        ),
        "contradiction_detect_rate": (
            sum(1 for r in contradiction_cases if bool(r.get("contradiction_detected", False))) / len(contradiction_cases)
            if contradiction_cases
            else 0.0
        ),
        "fuzzy_pass_rate": (sum(1 for r in fuzzy_cases if r["score"]["pass"]) / len(fuzzy_cases) if fuzzy_cases else 0.0),
        "fuzzy_closure_rate": (sum(1 for r in fuzzy_cases if r["is_complete"]) / len(fuzzy_cases) if fuzzy_cases else 0.0),
        "failed_ids": [r["id"] for r in results if not r["score"]["pass"]],
    }


def _by_category(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        bucket.setdefault(r["category"], []).append(r)
    rows: list[dict[str, Any]] = []
    for category, items in sorted(bucket.items(), key=lambda x: x[0]):
        n = len(items)
        rows.append(
            {
                "category": category,
                "count": n,
                "pass_rate": sum(1 for x in items if x["score"]["pass"]) / n if n else 0.0,
                "closure_rate": sum(1 for x in items if x["is_complete"]) / n if n else 0.0,
                "avg_accuracy": sum(float(x["score"]["accuracy"]) for x in items) / n if n else 0.0,
                "avg_naturalness": sum(float(x["naturalness"]["naturalness_score"]) for x in items) / n if n else 0.0,
            }
        )
    return rows


def _pick_representatives(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        bucket.setdefault(r["category"], []).append(r)
    reps: list[dict[str, Any]] = []
    for category, items in sorted(bucket.items(), key=lambda x: x[0]):
        _ = category
        items_sorted = sorted(
            items,
            key=lambda x: (
                0 if x["score"]["pass"] else 1,
                -float(x["score"]["accuracy"]),
                -float(x["naturalness"]["naturalness_score"]),
                len(x.get("turns", [])),
                x["id"],
            ),
        )
        reps.append(items_sorted[0])
    return reps


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _scenario_coords(results: list[dict[str, Any]], metric: str) -> str:
    coords: list[str] = []
    for r in results:
        if metric == "accuracy":
            value = float(r["score"]["accuracy"])
        elif metric == "naturalness":
            value = float(r["naturalness"]["naturalness_score"])
        else:
            value = 1.0 if r["is_complete"] else 0.0
        coords.append(f"({r['id']},{value:.4f})")
    return " ".join(coords)


def _category_coords(rows: list[dict[str, Any]], metric: str) -> str:
    coords: list[str] = []
    for idx, row in enumerate(rows, start=1):
        label = f"C{idx:02d}"
        coords.append(f"({label},{float(row.get(metric, 0.0)):.4f})")
    return " ".join(coords)


def _to_tex(run_cfg: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]], cat_rows: list[dict[str, Any]], reps: list[dict[str, Any]]) -> str:
    scenario_ids = ",".join(r["id"] for r in results)
    category_ids = ",".join(f"C{idx:02d}" for idx, _ in enumerate(cat_rows, start=1))
    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{ctexart}")
    lines.append(r"\usepackage[a4paper,margin=1in]{geometry}")
    lines.append(r"\usepackage{longtable,booktabs,array}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{listings}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{pgfplots}")
    lines.append(r"\pgfplotsset{compat=1.18}")
    lines.append(r"\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible,showstringspaces=false}")
    lines.append(r"\title{Geant4-Agent Full Test Report with Fuzzy-Language Robustness}")
    lines.append(r"\author{Automated Evaluation Pipeline}")
    lines.append(rf"\date{{{_latex_escape(run_cfg['date'])}}}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\section{Run Config}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item config: \texttt{{{_latex_escape(run_cfg['config_path'])}}}")
    lines.append(rf"\item provider: \texttt{{{_latex_escape(run_cfg['provider'])}}}")
    lines.append(rf"\item base\_url: \texttt{{{_latex_escape(run_cfg['base_url'])}}}")
    lines.append(rf"\item model: \texttt{{{_latex_escape(run_cfg['model'])}}}")
    lines.append(rf"\item lang: \texttt{{{_latex_escape(run_cfg['lang'])}}}, min\_confidence={run_cfg['min_confidence']}")
    lines.append(r"\end{itemize}")
    lines.append(r"\section{Summary Metrics}")
    lines.append(r"\begin{longtable}{ll}")
    lines.append(r"\toprule")
    lines.append(r"metric & value\\")
    lines.append(r"\midrule")
    for key in [
        "total_scenarios",
        "strict_pass_rate",
        "avg_accuracy",
        "avg_naturalness",
        "avg_grounding_mismatch_rate",
        "closure_rate",
        "avg_turns_per_scenario",
        "clarification_turn_ratio",
        "overwrite_guard_pass_rate",
        "contradiction_detect_rate",
        "fuzzy_pass_rate",
        "fuzzy_closure_rate",
    ]:
        lines.append(rf"{_latex_escape(key)} & {summary[key]:.4f} \\")
    lines.append(rf"failed\_ids & {_latex_escape(', '.join(summary['failed_ids']) if summary['failed_ids'] else 'none')} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Scenario Table}")
    lines.append(r"\begin{longtable}{p{0.08\linewidth}p{0.15\linewidth}p{0.22\linewidth}p{0.09\linewidth}p{0.10\linewidth}p{0.10\linewidth}p{0.12\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"id & category & domain & turns & accuracy & closure & pass \\")
    lines.append(r"\midrule")
    for r in results:
        lines.append(
            rf"{_latex_escape(r['id'])} & {_latex_escape(r['category'])} & {_latex_escape(r['domain'])} & {len(r['turns'])} & {r['score']['accuracy']:.3f} & {_latex_escape(str(r['is_complete']))} & {_latex_escape(str(r['score']['pass']))} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Category Table}")
    lines.append(r"\begin{longtable}{p{0.08\linewidth}p{0.22\linewidth}p{0.10\linewidth}p{0.13\linewidth}p{0.13\linewidth}p{0.13\linewidth}p{0.13\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"legend & category & n & pass\_rate & closure\_rate & avg\_acc & avg\_nat \\")
    lines.append(r"\midrule")
    for idx, row in enumerate(cat_rows, start=1):
        lines.append(
            rf"C{idx:02d} & {_latex_escape(row['category'])} & {row['count']} & {row['pass_rate']:.3f} & {row['closure_rate']:.3f} & {row['avg_accuracy']:.3f} & {row['avg_naturalness']:.3f} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Visual Analysis}")
    lines.append(r"\subsection{Scenario Accuracy}")
    lines.append(r"\begin{center}\begin{tikzpicture}")
    lines.append(rf"\begin{{axis}}[ybar,width=0.94\linewidth,height=6.0cm,ymin=0,ymax=1,symbolic x coords={{{scenario_ids}}},xtick=data,x tick label style={{rotate=45,anchor=east}},bar width=8pt]")
    lines.append(rf"\addplot coordinates {{{_scenario_coords(results, 'accuracy')}}};")
    lines.append(r"\end{axis}\end{tikzpicture}\end{center}")
    lines.append(r"\subsection{Scenario Naturalness}")
    lines.append(r"\begin{center}\begin{tikzpicture}")
    lines.append(rf"\begin{{axis}}[ybar,width=0.94\linewidth,height=6.0cm,ymin=0,ymax=1,symbolic x coords={{{scenario_ids}}},xtick=data,x tick label style={{rotate=45,anchor=east}},bar width=8pt]")
    lines.append(rf"\addplot coordinates {{{_scenario_coords(results, 'naturalness')}}};")
    lines.append(r"\end{axis}\end{tikzpicture}\end{center}")
    lines.append(r"\subsection{Category Pass vs Closure}")
    lines.append(r"\begin{center}\begin{tikzpicture}")
    lines.append(rf"\begin{{axis}}[ybar,width=0.94\linewidth,height=6.0cm,ymin=0,ymax=1,symbolic x coords={{{category_ids}}},xtick=data,x tick label style={{rotate=45,anchor=east}},bar width=7pt,legend style={{at={{(0.5,1.06)}},anchor=south,legend columns=2}}]")
    lines.append(rf"\addplot coordinates {{{_category_coords(cat_rows, 'pass_rate')}}};")
    lines.append(rf"\addplot coordinates {{{_category_coords(cat_rows, 'closure_rate')}}};")
    lines.append(r"\legend{pass\_rate,closure\_rate}")
    lines.append(r"\end{axis}\end{tikzpicture}\end{center}")
    lines.append(r"\section{Representative Raw Dialogues (1 per category)}")
    for r in reps:
        lines.append(rf"\subsection{{{_latex_escape(r['category'])} / {_latex_escape(r['id'])}: {_latex_escape(r['title'])}}}")
        lines.append(r"\paragraph{User-facing raw dialogue}")
        lines.append(r"\begin{lstlisting}")
        lines.append(json.dumps(r["raw_dialogue"], ensure_ascii=False, indent=2))
        lines.append(r"\end{lstlisting}")
        lines.append(r"\paragraph{Internal iterative trace}")
        lines.append(r"\begin{lstlisting}")
        lines.append(json.dumps(r["turns"], ensure_ascii=False, indent=2))
        lines.append(r"\end{lstlisting}")
        lines.append(r"\paragraph{Checks}")
        lines.append(r"\begin{lstlisting}")
        lines.append(json.dumps(r["score"]["checks"], ensure_ascii=False, indent=2))
        lines.append(r"\end{lstlisting}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX report with fuzzy-language robustness tests.")
    parser.add_argument("--config", required=True, help="LLM config path")
    parser.add_argument("--outdir", default="docs")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    config_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = _base_scenarios() + _fuzzy_scenarios()
    results = [_run_scenario(s, args.config, args.lang, args.min_confidence) for s in scenarios]
    summary = _aggregate(results)
    cat_rows = _by_category(results)
    reps = _pick_representatives(results)

    stamp = date.today().isoformat()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / f"full_test_report_data_{stamp}.json"
    tex_path = outdir / f"full_test_report_{stamp}.tex"
    run_cfg = {
        "date": stamp,
        "config_path": args.config,
        "provider": str(config_payload.get("provider", "ollama")),
        "base_url": str(config_payload.get("base_url", "")),
        "model": str(config_payload.get("model", "")),
        "lang": args.lang,
        "min_confidence": args.min_confidence,
    }
    payload = {
        "run_config": run_cfg,
        "summary": summary,
        "category_summary": cat_rows,
        "representatives": [{"category": r["category"], "id": r["id"]} for r in reps],
        "results": results,
    }
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tex_path.write_text(_to_tex(run_cfg, summary, results, cat_rows, reps), encoding="utf-8")
    print(f"[ok] wrote {data_path}")
    print(f"[ok] wrote {tex_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

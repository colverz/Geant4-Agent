from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session


@dataclass
class Scenario:
    sid: str
    category: str  # accurate | fuzzy
    background: str  # physics | medical | aerospace
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


def _build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            sid="A01",
            category="accurate",
            background="physics",
            title="Detector calibration baseline (box)",
            turns=[
                "For detector calibration: copper box 1m x 1m x 1m, gamma point source 1 MeV at center along +z, physics FTFP_BERT, output json."
            ],
            expected={"complete": True, "geometry.structure": "single_box", "source.type": "point", "output.format": "json"},
        ),
        Scenario(
            sid="A02",
            category="accurate",
            background="physics",
            title="Attenuation study (tubs)",
            turns=[
                "Copper cylinder radius 30 mm half-length 50 mm, gamma point source 2 MeV at (0,0,-100) toward (0,0,1), physics QBBC, output root."
            ],
            expected={"complete": True, "geometry.structure": "single_tubs", "source.type": "point", "output.format": "root"},
        ),
        Scenario(
            sid="A03",
            category="accurate",
            background="medical",
            title="Water phantom beam setup",
            turns=[
                "Medical phantom: water box 300 mm x 300 mm x 300 mm, gamma beam source 6 MeV at (0,0,-500) direction (0,0,1), physics QBBC, output hdf5."
            ],
            expected={"complete": True, "geometry.structure": "single_box", "source.type": "beam", "output.format": "hdf5"},
        ),
        Scenario(
            sid="A04",
            category="accurate",
            background="medical",
            title="Brachy-style isotropic setup",
            turns=[
                "Water cylinder phantom, radius 30 mm, half-length 100 mm, isotropic gamma source 0.662 MeV at center, physics FTFP_BERT, output json."
            ],
            expected={"complete": True, "geometry.structure": "single_tubs", "source.type": "isotropic", "output.format": "json"},
        ),
        Scenario(
            sid="A05",
            category="accurate",
            background="aerospace",
            title="Shielding benchmark",
            turns=[
                "Aerospace shielding: aluminum cube 800 mm x 800 mm x 800 mm, gamma beam 2 MeV at (0,0,-400) direction (0,0,1), physics FTFP_BERT, output root."
            ],
            expected={"complete": True, "geometry.structure": "single_box", "source.type": "beam", "output.format": "root"},
        ),
        Scenario(
            sid="A06",
            category="accurate",
            background="aerospace",
            title="Silicon sensor body",
            turns=[
                "Silicon cylinder radius 50 mm half-length 100 mm, proton point source 150 MeV at origin toward +z, physics QGSP_BERT, output json."
            ],
            expected={"complete": True, "geometry.structure": "single_tubs", "source.particle": "proton", "output.format": "json"},
        ),
        Scenario(
            sid="F01",
            category="fuzzy",
            background="physics",
            title="Colloquial attenuation request",
            turns=[
                "I need a copper meter-ish block and a gamma shot from center forward, around 1 MeV, give me json and a standard Geant4 physics list."
            ],
            expected={"geometry.structure": "single_box", "source.particle": "gamma", "output.format": "json"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F02",
            category="fuzzy",
            background="physics",
            title="Consult then adopt recommendation",
            turns=[
                "Pure gamma attenuation in copper, no hadrons. Recommend a physics list.",
                "Use your recommended list. Keep 1m copper cube, gamma point source 1 MeV at center +z, output json.",
            ],
            expected={"complete": True, "geometry.structure": "single_box", "output.format": "json"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F03",
            category="fuzzy",
            background="medical",
            title="Colloquial medical setup",
            turns=[
                "Need a water phantom around 30 cm cube, beam-like gamma about 6 MeV from front to center, output root, use a common list."
            ],
            expected={"geometry.structure": "single_box", "source.type": "beam", "output.format": "root"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F04",
            category="fuzzy",
            background="medical",
            title="Mixed units + staged completion",
            turns=[
                "Water cyl phantom r=3cm and h=20cm, isotropic gamma 662keV at center, output hdf5.",
                "Use QBBC physics.",
            ],
            expected={"complete": True, "geometry.structure": "single_tubs", "source.type": "isotropic", "output.format": "hdf5"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F05",
            category="fuzzy",
            background="aerospace",
            title="Aerospace shorthand",
            turns=[
                "Al shield cube 0.8m each side; src gamma beam E=2MeV from z- to z+; physics FTFP_BERT; root output."
            ],
            expected={"complete": True, "geometry.structure": "single_box", "source.type": "beam", "output.format": "root"},
            tags={"fuzzy"},
        ),
        Scenario(
            sid="F06",
            category="fuzzy",
            background="aerospace",
            title="High-level incomplete baseline request",
            turns=[
                "Set up a satellite shielding gamma test with aluminum wall first, then we refine details."
            ],
            expected={"complete": False, "missing_at_least": 1, "has_any_action": ["ask_clarification", "summarize_progress", "answer_status"]},
            tags={"fuzzy"},
        ),
    ]


def _run_scenario(s: Scenario, config_path: str, lang: str, min_confidence: float) -> dict[str, Any]:
    session_id = f"dual-{s.sid}"
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
                "missing_fields": list(out.get("missing_fields", [])),
                "normalization": dict(out.get("normalization", {})),
                "llm_used": bool(out.get("llm_used", False)),
                "fallback_reason": out.get("fallback_reason"),
                "internal_trace": dict(out.get("internal_trace", {})),
            }
        )
    assert final is not None
    score = _score_case(s, final, actions)
    reset_session(session_id)
    return {
        "id": s.sid,
        "category": s.category,
        "background": s.background,
        "title": s.title,
        "expected": s.expected,
        "tags": sorted(list(s.tags)),
        "score": score,
        "actions": actions,
        "is_complete": bool(final.get("is_complete", False)),
        "turns": turn_logs,
        "raw_dialogue": list(final.get("raw_dialogue", [])),
        "final_config": dict(final.get("config", {})),
        "final_missing": list(final.get("missing_fields", [])),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    pass_cnt = sum(1 for r in results if r["score"]["pass"])
    close_cnt = sum(1 for r in results if r["is_complete"])
    return {
        "total_scenarios": total,
        "strict_pass_rate": pass_cnt / total if total else 0.0,
        "closure_rate": close_cnt / total if total else 0.0,
        "avg_accuracy": (sum(float(r["score"]["accuracy"]) for r in results) / total) if total else 0.0,
        "failed_ids": [r["id"] for r in results if not r["score"]["pass"]],
    }


def _group_metrics(results: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    bucket: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        bucket.setdefault(str(r[field_name]), []).append(r)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(bucket.items(), key=lambda x: x[0]):
        n = len(items)
        rows.append(
            {
                field_name: key,
                "count": n,
                "pass_rate": sum(1 for x in items if x["score"]["pass"]) / n if n else 0.0,
                "closure_rate": sum(1 for x in items if x["is_complete"]) / n if n else 0.0,
                "avg_accuracy": sum(float(x["score"]["accuracy"]) for x in items) / n if n else 0.0,
            }
        )
    return rows


def _pick_representatives(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        key = (r["category"], r["background"])
        bucket.setdefault(key, []).append(r)
    reps: list[dict[str, Any]] = []
    for key, items in sorted(bucket.items(), key=lambda x: x[0]):
        _ = key
        items_sorted = sorted(
            items,
            key=lambda x: (
                0 if x["score"]["pass"] else 1,
                -float(x["score"]["accuracy"]),
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
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _to_tex(
    run_cfg: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    by_class: list[dict[str, Any]],
    by_background: list[dict[str, Any]],
    reps: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=1in]{geometry}")
    lines.append(r"\usepackage{longtable,booktabs}")
    lines.append(r"\usepackage{listings}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible,showstringspaces=false}")
    lines.append(r"\title{Geant4-Agent Dual-Class Domain Test Report}")
    lines.append(r"\author{Automated Evaluation Pipeline}")
    lines.append(rf"\date{{{_latex_escape(run_cfg['date'])}}}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\section{Run Config}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item config: \texttt{{{_latex_escape(run_cfg['config_path'])}}}")
    lines.append(rf"\item provider: \texttt{{{_latex_escape(run_cfg['provider'])}}}")
    lines.append(rf"\item model: \texttt{{{_latex_escape(run_cfg['model'])}}}")
    lines.append(rf"\item lang: \texttt{{{_latex_escape(run_cfg['lang'])}}}, min\_confidence={run_cfg['min_confidence']}")
    lines.append(r"\end{itemize}")
    lines.append(r"\section{Overall Metrics}")
    lines.append(r"\begin{longtable}{ll}")
    lines.append(r"\toprule")
    lines.append(r"metric & value\\")
    lines.append(r"\midrule")
    lines.append(rf"total\_scenarios & {summary['total_scenarios']} \\")
    lines.append(rf"strict\_pass\_rate & {summary['strict_pass_rate']:.4f} \\")
    lines.append(rf"closure\_rate & {summary['closure_rate']:.4f} \\")
    lines.append(rf"avg\_accuracy & {summary['avg_accuracy']:.4f} \\")
    lines.append(rf"failed\_ids & {_latex_escape(', '.join(summary['failed_ids']) if summary['failed_ids'] else 'none')} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Class Metrics (accurate vs fuzzy)}")
    lines.append(r"\begin{longtable}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"class & n & pass\_rate & closure\_rate & avg\_accuracy\\")
    lines.append(r"\midrule")
    for row in by_class:
        lines.append(rf"{_latex_escape(row['category'])} & {row['count']} & {row['pass_rate']:.3f} & {row['closure_rate']:.3f} & {row['avg_accuracy']:.3f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Background Metrics}")
    lines.append(r"\begin{longtable}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"background & n & pass\_rate & closure\_rate & avg\_accuracy\\")
    lines.append(r"\midrule")
    for row in by_background:
        lines.append(rf"{_latex_escape(row['background'])} & {row['count']} & {row['pass_rate']:.3f} & {row['closure_rate']:.3f} & {row['avg_accuracy']:.3f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Scenario Results}")
    lines.append(r"\begin{longtable}{p{0.08\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.32\linewidth}p{0.1\linewidth}p{0.1\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"id & class & background & title & accuracy & pass\\")
    lines.append(r"\midrule")
    for r in results:
        lines.append(
            rf"{_latex_escape(r['id'])} & {_latex_escape(r['category'])} & {_latex_escape(r['background'])} & {_latex_escape(r['title'])} & {r['score']['accuracy']:.3f} & {_latex_escape(str(r['score']['pass']))} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\section{Representative Dialogues (one per class-background)}")
    for r in reps:
        lines.append(rf"\subsection{{{_latex_escape(r['id'])}: {_latex_escape(r['category'])} / {_latex_escape(r['background'])} / {_latex_escape(r['title'])}}}")
        lines.append(r"\paragraph{Raw dialogue}")
        lines.append(r"\begin{lstlisting}")
        lines.append(json.dumps(r["raw_dialogue"], ensure_ascii=False, indent=2))
        lines.append(r"\end{lstlisting}")
        lines.append(r"\paragraph{Internal iterative turns}")
        lines.append(r"\begin{lstlisting}")
        lines.append(json.dumps(r["turns"], ensure_ascii=False, indent=2))
        lines.append(r"\end{lstlisting}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-class (accurate/fuzzy) domain test report generator.")
    parser.add_argument("--config", required=True, help="LLM config path")
    parser.add_argument("--outdir", default="docs")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    config_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = _build_scenarios()
    results = [_run_scenario(s, args.config, args.lang, args.min_confidence) for s in scenarios]
    summary = _aggregate(results)
    by_class = _group_metrics(results, "category")
    by_background = _group_metrics(results, "background")
    reps = _pick_representatives(results)

    stamp = date.today().isoformat()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / f"dual_class_domain_test_data_{stamp}.json"
    tex_path = outdir / f"dual_class_domain_test_report_{stamp}.tex"
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
        "by_class": by_class,
        "by_background": by_background,
        "representatives": [{"id": r["id"], "category": r["category"], "background": r["background"]} for r in reps],
        "results": results,
    }
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tex_path.write_text(_to_tex(run_cfg, summary, results, by_class, by_background, reps), encoding="utf-8")
    print(f"[ok] wrote {data_path}")
    print(f"[ok] wrote {tex_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


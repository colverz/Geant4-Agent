from __future__ import annotations

import argparse
from datetime import date
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.orchestrator.path_ops import get_path
from core.orchestrator.session_manager import process_turn, reset_session


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"


@dataclass(frozen=True)
class EvalTurn:
    text: str
    lang: str


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    suite: str
    title: str
    language: str
    turns: list[EvalTurn]
    evaluator: Callable[[list[dict[str, Any]]], tuple[bool, list[str]]]


def _write_ollama_config(base_url: str, model: str, timeout_s: int) -> str:
    payload = {
        "base_url": base_url,
        "model": model,
        "timeout_s": timeout_s,
        "headers": {"Content-Type": "application/json"},
    }
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
    with handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return handle.name


def _field(config: dict[str, Any], path: str, default: Any = None) -> Any:
    return get_path(config, path, default)


def _is_point_source(config: dict[str, Any]) -> bool:
    return _field(config, "source.type") == "point"


def _last(turns: list[dict[str, Any]]) -> dict[str, Any]:
    return turns[-1]


def _expect_box_complete(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    out = _last(turns)
    cfg = out["config"]
    reasons: list[str] = []
    if not out.get("is_complete"):
        reasons.append("final turn not complete")
    if _field(cfg, "geometry.structure") != "single_box":
        reasons.append("geometry.structure != single_box")
    if _field(cfg, "geometry.params.module_x") != 1000.0:
        reasons.append("module_x != 1000")
    if _field(cfg, "geometry.params.module_y") != 1000.0:
        reasons.append("module_y != 1000")
    if _field(cfg, "geometry.params.module_z") != 1000.0:
        reasons.append("module_z != 1000")
    if "G4_Cu" not in (_field(cfg, "materials.selected_materials", []) or []):
        reasons.append("material is not G4_Cu")
    if not _is_point_source(cfg):
        reasons.append("source.type != point")
    if _field(cfg, "source.particle") != "gamma":
        reasons.append("source.particle != gamma")
    if _field(cfg, "physics.physics_list") != "FTFP_BERT":
        reasons.append("physics.physics_list != FTFP_BERT")
    if _field(cfg, "output.format") != "root":
        reasons.append("output.format != root")
    return (not reasons, reasons)


def _expect_tubs_complete(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    out = _last(turns)
    cfg = out["config"]
    reasons: list[str] = []
    if not out.get("is_complete"):
        reasons.append("final turn not complete")
    if _field(cfg, "geometry.structure") != "single_tubs":
        reasons.append("geometry.structure != single_tubs")
    if _field(cfg, "geometry.params.child_rmax") != 30.0:
        reasons.append("child_rmax != 30")
    if _field(cfg, "geometry.params.child_hz") != 50.0:
        reasons.append("child_hz != 50")
    if "G4_Cu" not in (_field(cfg, "materials.selected_materials", []) or []):
        reasons.append("material is not G4_Cu")
    if _field(cfg, "output.format") != "root":
        reasons.append("output.format != root")
    return (not reasons, reasons)


def _expect_recommend_then_explain(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    first = turns[0]
    second = turns[1]
    reasons: list[str] = []
    cfg = second["config"]
    if not first.get("is_complete"):
        reasons.append("setup turn not complete")
    if second.get("dialogue_action") != "explain_choice":
        reasons.append("final action != explain_choice")
    if not _field(cfg, "physics.physics_list"):
        reasons.append("physics.physics_list missing")
    if not _field(cfg, "physics.selection_source"):
        reasons.append("physics.selection_source missing")
    if not (_field(cfg, "physics.selection_reasons", []) or []):
        reasons.append("physics.selection_reasons missing")
    return (not reasons, reasons)


def _expect_modify_then_confirm(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    second = turns[1]
    third = turns[2]
    reasons: list[str] = []
    if second.get("dialogue_action") != "confirm_overwrite":
        reasons.append("modify turn did not stage overwrite")
    if third.get("dialogue_action") == "confirm_overwrite":
        reasons.append("confirm turn still blocked")
    mats = _field(third["config"], "materials.selected_materials", []) or []
    if mats != ["G4_Al"]:
        reasons.append("material not committed to G4_Al")
    if _field(third["config"], "materials.volume_material_map") != {"box": "G4_Al"}:
        reasons.append("volume_material_map not synchronized")
    return (not reasons, reasons)


def _expect_output_supplement(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    second = turns[1]
    reasons: list[str] = []
    if second.get("dialogue_action") == "confirm_overwrite":
        reasons.append("output supplement incorrectly triggered confirm_overwrite")
    if _field(second["config"], "output.format") != "json":
        reasons.append("output.format != json")
    if not _field(second["config"], "output.path"):
        reasons.append("output.path missing")
    return (not reasons, reasons)


def _expect_output_format_complete(expected_format: str) -> Callable[[list[dict[str, Any]]], tuple[bool, list[str]]]:
    def _evaluator(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        out = _last(turns)
        cfg = out["config"]
        reasons: list[str] = []
        if not out.get("is_complete"):
            reasons.append("final turn not complete")
        if _field(cfg, "output.format") != expected_format:
            reasons.append(f"output.format != {expected_format}")
        path = str(_field(cfg, "output.path", "") or "")
        if not path:
            reasons.append("output.path missing")
        else:
            suffix_map = {"root": ".root", "json": ".json", "csv": ".csv", "xml": ".xml", "hdf5": ".hdf5"}
            expected_suffix = suffix_map.get(expected_format)
            if expected_suffix and not path.lower().endswith(expected_suffix):
                reasons.append(f"output.path does not end with {expected_suffix}")
        return (not reasons, reasons)

    return _evaluator


def _expect_progressive_completion(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    out = _last(turns)
    reasons: list[str] = []
    if not out.get("is_complete"):
        reasons.append("final turn not complete")
    if _field(out["config"], "output.format") != "json":
        reasons.append("output.format != json")
    if not any(turn.get("dialogue_action") == "summarize_progress" for turn in turns[:-1]):
        reasons.append("summarize_progress never triggered")
    return (not reasons, reasons)


def _expect_field_scoped_pending(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    second = turns[1]
    third = turns[2]
    fourth = turns[3]
    reasons: list[str] = []
    if second.get("dialogue_action") != "confirm_overwrite":
        reasons.append("modify turn did not enter confirm_overwrite")
    if third.get("dialogue_action") != "confirm_overwrite":
        reasons.append("non-conflict supplement should still show pending confirm")
    if _field(third["config"], "output.format") != "json":
        reasons.append("non-conflict output update did not commit")
    if (_field(third["dialogue_trace"], "overwrite_preview", []) or [{}])[0].get("path") != "materials.selected_materials":
        reasons.append("overwrite preview did not remain scoped to materials.selected_materials")
    if _field(fourth["config"], "materials.selected_materials", []) != ["G4_Al"]:
        reasons.append("confirmation did not commit staged material change")
    return (not reasons, reasons)


def _expect_double_overwrite_cycles(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    second = turns[1]
    third = turns[2]
    fourth = turns[3]
    fifth = turns[4]
    reasons: list[str] = []
    if second.get("dialogue_action") != "confirm_overwrite":
        reasons.append("first overwrite not staged")
    if _field(third["config"], "materials.selected_materials", []) != ["G4_Al"]:
        reasons.append("first confirmation did not commit material")
    if fourth.get("dialogue_action") != "confirm_overwrite":
        reasons.append("second overwrite not staged")
    if _field(fifth["config"], "output.format") != "json":
        reasons.append("second confirmation did not commit output json")
    return (not reasons, reasons)


def build_bilingual_cases() -> list[EvalCase]:
    en_box = "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT, output ROOT."
    zh_box = "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT，输出ROOT。"
    en_tubs = "Please set up a cylindrical copper target with radius 30 mm and half-length 50 mm, with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT, output ROOT."
    zh_tubs = "请建立一个铜圆柱靶，半径30毫米，半长50毫米，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT，输出ROOT。"
    en_reco = [
        "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, output ROOT. Choose the most suitable Geant4 reference physics list for pure gamma attenuation in copper.",
        "Why was this physics list selected?",
    ]
    zh_reco = [
        "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，输出ROOT。请为纯gamma在铜中的衰减选择最合适的Geant4参考物理列表。",
        "为什么选择这个物理列表？",
    ]
    en_modify = [
        en_box,
        "Change material to G4_Al.",
        "confirm",
    ]
    zh_modify = [
        zh_box,
        "把材料改成 G4_Al。",
        "确认",
    ]
    en_output = [
        "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT.",
        "Output json.",
    ]
    zh_output = [
        "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT。",
        "输出 json。",
    ]
    cases: list[EvalCase] = []
    def add(case_id: str, title: str, turns: list[str], lang: str, evaluator: Callable[[list[dict[str, Any]]], tuple[bool, list[str]]]) -> None:
        cases.append(
            EvalCase(
                case_id=case_id,
                suite="bilingual_strict_contract",
                title=title,
                language=lang,
                turns=[EvalTurn(text=text, lang=lang) for text in turns],
                evaluator=evaluator,
            )
        )
    add("S1_EN", "Explicit single_box complete", [en_box], "en", _expect_box_complete)
    add("S1_ZH", "显式 single_box 完整配置", [zh_box], "zh", _expect_box_complete)
    add("S2_EN", "Explicit single_tubs complete", [en_tubs], "en", _expect_tubs_complete)
    add("S2_ZH", "显式 single_tubs 完整配置", [zh_tubs], "zh", _expect_tubs_complete)
    add("S3_EN", "Recommendation then explain", en_reco, "en", _expect_recommend_then_explain)
    add("S3_ZH", "推荐后解释", zh_reco, "zh", _expect_recommend_then_explain)
    add("S4_EN", "Modify then confirm", en_modify, "en", _expect_modify_then_confirm)
    add("S4_ZH", "修改后确认", zh_modify, "zh", _expect_modify_then_confirm)
    add("S5_EN", "Narrow output supplement", en_output, "en", _expect_output_supplement)
    add("S5_ZH", "窄轮次输出补全", zh_output, "zh", _expect_output_supplement)
    return cases


def build_multiturn_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    def add(case_id: str, title: str, turns: list[str], lang: str, evaluator: Callable[[list[dict[str, Any]]], tuple[bool, list[str]]]) -> None:
        cases.append(
            EvalCase(
                case_id=case_id,
                suite="multiturn_strict_contract",
                title=title,
                language=lang,
                turns=[EvalTurn(text=text, lang=lang) for text in turns],
                evaluator=evaluator,
            )
        )
    add(
        "MT1_EN",
        "Progressive explicit completion",
        [
            "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source. Use physics FTFP_BERT.",
            "Set source energy to 1 MeV.",
            "Set source position to (0,0,-100).",
            "Set source direction to (0,0,1).",
            "Output json.",
        ],
        "en",
        _expect_progressive_completion,
    )
    add(
        "MT1_ZH",
        "逐步显式补全",
        [
            "请建立一个1米见方的铜立方体靶，使用gamma点源，物理列表用FTFP_BERT。",
            "把源能量设为1 MeV。",
            "把源位置设为(0,0,-100)。",
            "把源方向设为(0,0,1)。",
            "输出 json。",
        ],
        "zh",
        _expect_progressive_completion,
    )
    add(
        "MT2_EN",
        "Pending overwrite with non-conflict supplement",
        [
            "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT.",
            "Change material to G4_Al.",
            "Output json.",
            "confirm",
        ],
        "en",
        _expect_field_scoped_pending,
    )
    add(
        "MT2_ZH",
        "待确认覆盖期间补充非冲突字段",
        [
            "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT。",
            "把材料改成 G4_Al。",
            "输出 json。",
            "确认",
        ],
        "zh",
        _expect_field_scoped_pending,
    )
    add(
        "MT3_EN",
        "Recommendation then explanation",
        [
            "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, output ROOT. Choose the most suitable Geant4 reference physics list for pure gamma attenuation in copper.",
            "Why was this physics list selected?",
        ],
        "en",
        _expect_recommend_then_explain,
    )
    add(
        "MT3_ZH",
        "推荐后解释",
        [
            "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，输出ROOT。请为纯gamma在铜中的衰减选择最合适的Geant4参考物理列表。",
            "为什么选择这个物理列表？",
        ],
        "zh",
        _expect_recommend_then_explain,
    )
    add(
        "MT4_EN",
        "Two overwrite cycles",
        [
            "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT, output ROOT.",
            "Change material to G4_Al.",
            "confirm",
            "Change output to json.",
            "confirm",
        ],
        "en",
        _expect_double_overwrite_cycles,
    )
    add(
        "MT4_ZH",
        "两次覆盖确认循环",
        [
            "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT，输出ROOT。",
            "把材料改成 G4_Al。",
            "确认",
            "把输出改成 json。",
            "确认",
        ],
        "zh",
        _expect_double_overwrite_cycles,
    )
    return cases


def build_output_format_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []

    def add(case_id: str, title: str, turns: list[str], lang: str, expected_format: str) -> None:
        cases.append(
            EvalCase(
                case_id=case_id,
                suite="output_formats_strict_contract",
                title=title,
                language=lang,
                turns=[EvalTurn(text=text, lang=lang) for text in turns],
                evaluator=_expect_output_format_complete(expected_format),
            )
        )

    add(
        "OF1_EN",
        "Explicit HDF5 output",
        [
            "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT, output ROOT.",
            "Output hdf5.",
            "confirm",
        ],
        "en",
        "hdf5",
    )
    add(
        "OF1_ZH",
        "显式 HDF5 输出",
        [
            "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT，输出ROOT。",
            "输出 hdf5。",
            "确认",
        ],
        "zh",
        "hdf5",
    )
    add(
        "OF2_EN",
        "Explicit XML output",
        [
            "Please set up a 1 m x 1 m x 1 m copper box target with a gamma point source at (0,0,-100) mm pointing (0,0,1), energy 1 MeV, physics FTFP_BERT, output ROOT.",
            "Output xml.",
            "confirm",
        ],
        "en",
        "xml",
    )
    add(
        "OF2_ZH",
        "显式 XML 输出",
        [
            "请建立一个1米见方的铜立方体靶，使用gamma点源，位置为(0,0,-100)毫米，方向为(0,0,1)，能量1 MeV，物理列表用FTFP_BERT，输出ROOT。",
            "输出 xml。",
            "确认",
        ],
        "zh",
        "xml",
    )
    return cases


def _run_case(case: EvalCase, *, ollama_config_path: str, min_confidence: float) -> dict[str, Any]:
    session_id = f"eval-{case.case_id}"
    reset_session(session_id)
    turns_out: list[dict[str, Any]] = []
    for turn in case.turns:
        out = process_turn(
            {
                "session_id": session_id,
                "text": turn.text,
                "lang": turn.lang,
                "llm_router": True,
                "llm_question": False,
                "normalize_input": True,
                "autofix": True,
            },
            ollama_config_path=ollama_config_path,
            min_confidence=min_confidence,
            lang=turn.lang,
        )
        turns_out.append(
            {
                "user": turn.text,
                "lang": turn.lang,
                "dialogue_action": out.get("dialogue_action"),
                "assistant_message": out.get("assistant_message"),
                "dialogue_trace": out.get("dialogue_trace"),
                "raw_dialogue": out.get("raw_dialogue"),
                "config": out.get("config"),
                "missing_fields": out.get("missing_fields"),
                "is_complete": out.get("is_complete"),
                "inference_backend": out.get("inference_backend"),
            }
        )
    passed, reasons = case.evaluator(turns_out)
    return {
        "case_id": case.case_id,
        "suite": case.suite,
        "title": case.title,
        "language": case.language,
        "passed": passed,
        "failure_reasons": reasons,
        "turns": turns_out,
    }


def _suite_report(cases: list[EvalCase], *, ollama_config_path: str, min_confidence: float) -> dict[str, Any]:
    results = [_run_case(case, ollama_config_path=ollama_config_path, min_confidence=min_confidence) for case in cases]
    passed = sum(1 for item in results if item["passed"])
    return {
        "suite": cases[0].suite if cases else "unknown",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def _filter_cases(cases: list[EvalCase], selected_ids: set[str]) -> list[EvalCase]:
    if not selected_ids:
        return cases
    return [case for case in cases if case.case_id in selected_ids]


def _write_suite_outputs(
    report: dict[str, Any],
    *,
    stem: str,
    base_url: str,
    model: str,
    report_date: str,
) -> None:
    payload = {
        "date": report_date,
        "base_url": base_url,
        "model": model,
        **report,
    }
    json_path = DOCS_DIR / f"{stem}_{report_date}.json"
    md_path = DOCS_DIR / f"{stem}_{report_date}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {report['suite']}",
        "",
        f"- Date: {report_date}",
        f"- Base URL: `{base_url}`",
        f"- Model: `{model}`",
        f"- Passed: `{report['passed']}/{report['total']}`",
        "",
    ]
    for result in report["results"]:
        lines.append(f"## {result['case_id']} - {result['title']}")
        lines.append(f"- Language: `{result['language']}`")
        lines.append(f"- Passed: `{'yes' if result['passed'] else 'no'}`")
        if result["failure_reasons"]:
            lines.append(f"- Failure reasons: `{result['failure_reasons']}`")
        for idx, turn in enumerate(result["turns"], 1):
            lines.append(f"### Turn {idx}")
            lines.append(f"- User: `{turn['user']}`")
            lines.append(f"- Dialogue action: `{turn['dialogue_action']}`")
            lines.append(f"- Backend: `{turn['inference_backend']}`")
            lines.append(f"- Missing fields: `{turn['missing_fields']}`")
            lines.append(f"- Assistant: `{turn['assistant_message']}`")
            lines.append("- Raw dialogue:")
            for item in (turn.get("raw_dialogue") or [])[-4:]:
                lines.append(f"  - {item.get('role')}: {item.get('content')}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(
    *,
    bilingual: dict[str, Any],
    multiturn: dict[str, Any],
    base_url: str,
    model: str,
    report_date: str,
) -> None:
    lines = [
        "# Strict Contract Dialogue Evaluation Summary",
        "",
        f"- Date: {report_date}",
        f"- Base URL: `{base_url}`",
        f"- Model: `{model}`",
        "",
        "## Results",
        f"- Bilingual strict-contract suite: `{bilingual['passed']}/{bilingual['total']}`",
        f"- Multiturn strict-contract suite: `{multiturn['passed']}/{multiturn['total']}`",
        "",
        "## Interpretation",
        "- These suites use the new strict turn contract rather than the archived wide-fill prompts.",
        "- Failures here are more representative of current strict runtime behavior.",
        "- If these suites pass while the archived suites fail, the issue is prompt-contract mismatch rather than immediate runtime instability.",
        "",
    ]
    for title, report in [("Bilingual", bilingual), ("Multiturn", multiturn)]:
        failed = [item for item in report["results"] if not item["passed"]]
        lines.append(f"## {title} Failures")
        if not failed:
            lines.append("- None")
        else:
            for item in failed:
                lines.append(f"- `{item['case_id']}`: {item['failure_reasons']}")
        lines.append("")
    (DOCS_DIR / f"strict_contract_dialogue_summary_{report_date}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--suite", choices=["all", "bilingual", "multiturn", "output_formats"], default="all")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--report-date", default=date.today().isoformat())
    args = parser.parse_args()

    ollama_config_path = _write_ollama_config(args.base_url, args.model, args.timeout_s)
    summary: dict[str, Any] = {}
    bilingual: dict[str, Any] | None = None
    multiturn: dict[str, Any] | None = None
    selected_ids = {str(item) for item in args.case_id if str(item)}
    if args.suite in {"all", "bilingual"}:
        bilingual_cases = _filter_cases(build_bilingual_cases(), selected_ids)
        bilingual = _suite_report(bilingual_cases, ollama_config_path=ollama_config_path, min_confidence=args.min_confidence)
        _write_suite_outputs(
            bilingual,
            stem="bilingual_dialogue_eval_strict_contract",
            base_url=args.base_url,
            model=args.model,
            report_date=args.report_date,
        )
        summary["bilingual"] = {"passed": bilingual["passed"], "total": bilingual["total"]}
    if args.suite in {"all", "multiturn"}:
        multiturn_cases = _filter_cases(build_multiturn_cases(), selected_ids)
        multiturn = _suite_report(multiturn_cases, ollama_config_path=ollama_config_path, min_confidence=args.min_confidence)
        _write_suite_outputs(
            multiturn,
            stem="multiturn_dialogue_eval_strict_contract",
            base_url=args.base_url,
            model=args.model,
            report_date=args.report_date,
        )
        summary["multiturn"] = {"passed": multiturn["passed"], "total": multiturn["total"]}
    if args.suite in {"all", "output_formats"}:
        output_cases = _filter_cases(build_output_format_cases(), selected_ids)
        output_formats = _suite_report(output_cases, ollama_config_path=ollama_config_path, min_confidence=args.min_confidence)
        _write_suite_outputs(
            output_formats,
            stem="output_formats_eval_strict_contract",
            base_url=args.base_url,
            model=args.model,
            report_date=args.report_date,
        )
        summary["output_formats"] = {"passed": output_formats["passed"], "total": output_formats["total"]}
    if bilingual and multiturn:
        _write_summary(
            bilingual=bilingual,
            multiturn=multiturn,
            base_url=args.base_url,
            model=args.model,
            report_date=args.report_date,
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

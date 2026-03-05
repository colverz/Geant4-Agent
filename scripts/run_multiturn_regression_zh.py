from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session


INTERNAL_FIELD_RE = re.compile(r"\b[a-z]+(?:\.[a-z_]+)+\b")


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "case01_box_full_in_one_turn",
            "turns": [
                "我要一个铜立方体，尺寸 1m x 1m x 1m。Gamma 点源，能量 1 MeV，位置在中心，方向 +z，物理列表 FTFP_BERT，输出 json。"
            ],
            "expect_complete": True,
        },
        {
            "id": "case02_box_two_turns_fill_output",
            "turns": [
                "铜立方体 1m x 1m x 1m，gamma 点源 1 MeV，位置原点，方向 +z，物理列表 FTFP_BERT。",
                "输出 root。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case03_tubs_two_turns",
            "turns": [
                "我要圆柱铜靶，半径 30 mm，半长 50 mm，gamma 点源 2 MeV，位置(0,0,-100)，方向(0,0,1)。",
                "物理列表 QBBC，输出 json。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case04_overwrite_requires_confirm",
            "turns": [
                "铜立方体 1m x 1m x 1m，gamma 点源 1 MeV，位置中心，方向 +z，物理 FTFP_BERT，输出 json。",
                "把材料改成 G4_Al。",
                "确认。",
            ],
            "expect_complete": True,
            "expect_overwrite_flow": True,
        },
        {
            "id": "case05_source_split_turns",
            "turns": [
                "铜立方体 1m x 1m x 1m，gamma 点源。",
                "能量 1 MeV，位置 (0,0,0)，方向 +z。",
                "物理 FTFP_BERT，输出 json。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case06_material_then_geometry",
            "turns": [
                "材料用 G4_Si，粒子 gamma。",
                "几何是立方体，尺寸 20 cm x 20 cm x 20 cm，点源 0.5 MeV，位置中心，方向 +z。",
                "物理 FTFP_BERT，输出 csv。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case07_question_then_set",
            "turns": [
                "为什么选这个物理列表？",
                "那就用 FTFP_BERT。几何立方体 1m x 1m x 1m，材料 G4_Cu，gamma 点源 1 MeV，位置中心，方向 +z，输出 json。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case08_beam_source",
            "turns": [
                "硅立方体 50 cm x 50 cm x 50 cm，束流源 gamma，能量 3 MeV，位置(0,0,-200)，方向(0,0,1)，物理 QBBC。",
                "输出 root。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case09_plane_source",
            "turns": [
                "铝立方体 30 cm x 30 cm x 30 cm，面源 gamma，能量 0.8 MeV，位置(0,0,-50)，方向 +z。",
                "物理 FTFP_BERT，输出 json。",
            ],
            "expect_complete": True,
        },
        {
            "id": "case10_isotropic_source",
            "turns": [
                "水立方体 40 cm x 40 cm x 40 cm，各向同性 gamma 点源，能量 1.2 MeV，位置中心。",
                "方向 (0,0,1)，物理 QBBC，输出 hdf5。",
            ],
            "expect_complete": True,
        },
    ]


def _run_case(case: dict[str, Any], *, ollama_config: str, min_confidence: float, lang: str) -> dict[str, Any]:
    session_id = f"reg-{case['id']}"
    reset_session(session_id)
    turn_records: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for idx, text in enumerate(case["turns"], start=1):
        out = process_turn(
            {
                "session_id": session_id,
                "text": text,
                "llm_router": False,
                "llm_question": False,
                "normalize_input": False,
                "min_confidence": min_confidence,
            },
            ollama_config_path=ollama_config,
            min_confidence=min_confidence,
            lang=lang,
        )
        final = out
        turn_records.append(
            {
                "turn": idx,
                "user": text,
                "assistant": out.get("assistant_message", ""),
                "action": out.get("dialogue_action"),
                "missing_fields": list(out.get("missing_fields", [])),
                "is_complete": bool(out.get("is_complete", False)),
            }
        )

    assert final is not None
    reset_session(session_id)

    assistant_texts = [str(item["assistant"]) for item in turn_records]
    has_internal_field_text = any(INTERNAL_FIELD_RE.search(msg) for msg in assistant_texts)
    actions = [str(item.get("action", "")) for item in turn_records]
    overwrite_ok = True
    if case.get("expect_overwrite_flow"):
        overwrite_ok = "confirm_overwrite" in actions

    return {
        "id": case["id"],
        "expect_complete": bool(case.get("expect_complete", False)),
        "actual_complete": bool(final.get("is_complete", False)),
        "complete_match": bool(final.get("is_complete", False)) == bool(case.get("expect_complete", False)),
        "missing_fields_final": list(final.get("missing_fields", [])),
        "source_type": final.get("config", {}).get("source", {}).get("type"),
        "overwrite_flow_ok": overwrite_ok,
        "has_internal_field_text": has_internal_field_text,
        "turns": turn_records,
        "raw_dialogue": list(final.get("raw_dialogue", [])),
        "final_action": final.get("dialogue_action"),
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    complete_ok = sum(1 for item in results if item["complete_match"])
    overwrite_ok = sum(1 for item in results if item["overwrite_flow_ok"])
    no_internal = sum(1 for item in results if not item["has_internal_field_text"])
    return {
        "total_cases": total,
        "complete_match_rate": complete_ok / total if total else 0.0,
        "overwrite_flow_pass_rate": overwrite_ok / total if total else 0.0,
        "natural_reply_no_internal_field_rate": no_internal / total if total else 0.0,
        "failed_cases": [item["id"] for item in results if not item["complete_match"] or not item["overwrite_flow_ok"]],
    }


def _to_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# 多轮回归测试报告（中文，Strict/No-Ollama）")
    lines.append("")
    lines.append("## 汇总")
    lines.append(f"- 测试用例数: {summary['total_cases']}")
    lines.append(f"- 完整闭环匹配率: {summary['complete_match_rate']:.2%}")
    lines.append(f"- 覆盖确认流通过率: {summary['overwrite_flow_pass_rate']:.2%}")
    lines.append(f"- 回复无内部字段泄漏率: {summary['natural_reply_no_internal_field_rate']:.2%}")
    lines.append(f"- 失败用例: {', '.join(summary['failed_cases']) if summary['failed_cases'] else '无'}")
    lines.append("")
    lines.append("## 逐例明细（含 raw dialogue）")
    for case in results:
        lines.append(f"### {case['id']}")
        lines.append(f"- 期望完成: {case['expect_complete']}")
        lines.append(f"- 实际完成: {case['actual_complete']}")
        lines.append(f"- 最终缺失: {case['missing_fields_final']}")
        lines.append(f"- 最终源类型: {case['source_type']}")
        lines.append(f"- 覆盖确认流: {case['overwrite_flow_ok']}")
        lines.append(f"- 回复泄漏内部字段: {case['has_internal_field_text']}")
        lines.append("- turns:")
        for t in case["turns"]:
            lines.append(f"  - turn {t['turn']} user: {t['user']}")
            lines.append(f"    assistant: {t['assistant']}")
            lines.append(f"    action: {t['action']}, complete: {t['is_complete']}, missing: {t['missing_fields']}")
        lines.append("- raw_dialogue:")
        lines.append("```json")
        lines.append(json.dumps(case["raw_dialogue"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict no-ollama multi-turn regression (zh).")
    parser.add_argument("--outdir", default="docs")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--ollama_config", default="nlu/bert_lab/configs/ollama_config.json")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    results = [_run_case(case, ollama_config=args.ollama_config, min_confidence=args.min_confidence, lang=args.lang) for case in _cases()]
    summary = _summary(results)

    json_path = outdir / f"multiturn_regression_zh_{stamp}.json"
    md_path = outdir / f"multiturn_regression_zh_{stamp}.md"
    json_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(summary, results), encoding="utf-8")

    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


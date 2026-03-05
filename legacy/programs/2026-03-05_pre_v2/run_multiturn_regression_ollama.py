from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session
from nlu.bert_lab.ollama_client import OPENAI_COMPAT_PROVIDERS
from nlu.bert_lab.ollama_client import load_config


INTERNAL_FIELD_RE = re.compile(r"\b[a-z]+(?:\.[a-z_]+)+\b")
TEMPLATE_RE = re.compile(r"(please provide:|请补充字段[:：])", re.IGNORECASE)


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "ollama_case01_box_full",
            "turns": [
                "我要一个铜立方体，尺寸 1m x 1m x 1m。Gamma 点源，能量 1 MeV，位置在中心，方向 +z，物理列表 FTFP_BERT，输出 json。"
            ],
            "expected": {
                "complete": True,
                "geometry.structure": "single_box",
                "source.type": "point",
                "source.particle": "gamma",
                "physics.physics_list": "FTFP_BERT",
                "output.format": "json",
            },
        },
        {
            "id": "ollama_case02_box_two_turns",
            "turns": [
                "铜立方体 1m x 1m x 1m，gamma 点源，1 MeV，位置原点，方向 +z。",
                "物理列表 QBBC，输出 root。",
            ],
            "expected": {
                "complete": True,
                "geometry.structure": "single_box",
                "source.type": "point",
                "physics.physics_list": "QBBC",
                "output.format": "root",
            },
        },
        {
            "id": "ollama_case03_tubs_two_turns",
            "turns": [
                "我要圆柱铜靶，半径 30 mm，半长 50 mm，gamma 点源 2 MeV，位置(0,0,-100)，方向(0,0,1)。",
                "物理列表 QBBC，输出 json。",
            ],
            "expected": {
                "complete": True,
                "geometry.structure": "single_tubs",
                "source.type": "point",
                "physics.physics_list": "QBBC",
                "output.format": "json",
            },
        },
        {
            "id": "ollama_case04_overwrite_confirm",
            "turns": [
                "铜立方体 1m x 1m x 1m，gamma 点源 1 MeV，位置中心，方向 +z，物理 FTFP_BERT，输出 json。",
                "把材料改成 G4_Al。",
                "确认。",
            ],
            "expected": {
                "complete": True,
                "materials.selected_materials": ["G4_Al"],
                "source.type": "point",
            },
        },
        {
            "id": "ollama_case05_source_split",
            "turns": [
                "铜立方体 1m x 1m x 1m，gamma 点源。",
                "能量 1 MeV，位置 (0,0,0)，方向 +z。",
                "物理 FTFP_BERT，输出 json。",
            ],
            "expected": {
                "complete": True,
                "source.type": "point",
                "source.particle": "gamma",
                "source.energy": 1.0,
                "output.format": "json",
            },
        },
    ]


def _deep_get(cfg: dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
    return cur


def _resolve_auth_token(cfg: Any) -> str:
    if getattr(cfg, "api_key", None):
        return str(cfg.api_key)
    if getattr(cfg, "api_key_env", None):
        return os.getenv(str(cfg.api_key_env), "").strip()
    for env_name in ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token
    return ""


def _request_json(url: str, timeout_s: int, token: str = "") -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _probe_provider(cfg: Any) -> dict[str, Any]:
    provider = str(getattr(cfg, "provider", "ollama") or "ollama").lower()
    base_url = str(getattr(cfg, "base_url", "")).rstrip("/")
    timeout_s = int(getattr(cfg, "timeout_s", 8) or 8)
    token = _resolve_auth_token(cfg)

    candidate_urls: list[str]
    if provider in OPENAI_COMPAT_PROVIDERS:
        # Try common OpenAI-compatible model listing endpoints.
        candidate_urls = [f"{base_url}/v1/models", f"{base_url}/models"]
    else:
        candidate_urls = [f"{base_url}/api/tags"]

    errors: list[str] = []
    for url in candidate_urls:
        try:
            payload = _request_json(url, timeout_s=timeout_s, token=token)
            if provider in OPENAI_COMPAT_PROVIDERS:
                models = [str(x.get("id", "")) for x in payload.get("data", []) if str(x.get("id", ""))]
            else:
                models = [str(x.get("name", "")) for x in payload.get("models", []) if str(x.get("name", ""))]
            return {
                "ok": True,
                "base_url": base_url,
                "provider": provider,
                "model_count": len(models),
                "models": models[:10],
                "probe_url": url,
                "error": "",
            }
        except Exception as ex:
            errors.append(f"{url} -> {ex}")

    return {
        "ok": False,
        "base_url": base_url,
        "provider": provider,
        "model_count": 0,
        "models": [],
        "probe_url": candidate_urls[0] if candidate_urls else "",
        "error": " | ".join(errors),
    }


def _naturalness_stats(messages: list[str]) -> dict[str, Any]:
    if not messages:
        return {
            "internal_field_leak_rate": 1.0,
            "template_rate": 1.0,
            "short_reply_rate": 1.0,
            "naturalness_score": 0.0,
        }
    leaks = sum(1 for m in messages if INTERNAL_FIELD_RE.search(m))
    templates = sum(1 for m in messages if TEMPLATE_RE.search(m))
    shorts = sum(1 for m in messages if len(m.strip()) < 12)
    n = len(messages)
    leak_rate = leaks / n
    template_rate = templates / n
    short_rate = shorts / n
    score = max(0.0, 1.0 - (0.5 * leak_rate + 0.3 * template_rate + 0.2 * short_rate))
    return {
        "internal_field_leak_rate": leak_rate,
        "template_rate": template_rate,
        "short_reply_rate": short_rate,
        "naturalness_score": score,
    }


def _accuracy(case: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    expected = dict(case.get("expected", {}))
    cfg = dict(final.get("config", {}))
    checks: list[dict[str, Any]] = []
    hit = 0
    total = 0
    for key, want in expected.items():
        total += 1
        if key == "complete":
            got = bool(final.get("is_complete", False))
        else:
            got = _deep_get(cfg, key)
        ok = got == want
        if ok:
            hit += 1
        checks.append({"field": key, "expected": want, "actual": got, "ok": ok})
    return {
        "checks": checks,
        "hits": hit,
        "total": total,
        "accuracy": (hit / total) if total else 0.0,
        "pass": hit == total,
    }


def _run_case(case: dict[str, Any], *, ollama_config: str, min_confidence: float, lang: str) -> dict[str, Any]:
    sid = f"ollama-{case['id']}"
    reset_session(sid)
    turns: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for i, text in enumerate(case["turns"], start=1):
        out = process_turn(
            {
                "session_id": sid,
                "text": text,
                "llm_router": True,
                "llm_question": True,
                "normalize_input": True,
                "min_confidence": min_confidence,
            },
            ollama_config_path=ollama_config,
            min_confidence=min_confidence,
            lang=lang,
        )
        final = out
        turns.append(
            {
                "turn": i,
                "user": text,
                "assistant": out.get("assistant_message", ""),
                "dialogue_action": out.get("dialogue_action", ""),
                "llm_used": bool(out.get("llm_used", False)),
                "fallback_reason": out.get("fallback_reason"),
                "llm_stage_failures": list(out.get("llm_stage_failures", [])),
                "missing_fields": list(out.get("missing_fields", [])),
            }
        )

    assert final is not None
    natur = _naturalness_stats([str(t["assistant"]) for t in turns])
    acc = _accuracy(case, final)
    llm_turns = sum(1 for t in turns if t["llm_used"])
    reset_session(sid)

    return {
        "id": case["id"],
        "naturalness": natur,
        "accuracy": acc,
        "llm_used_turn_ratio": llm_turns / len(turns),
        "final": {
            "is_complete": bool(final.get("is_complete", False)),
            "missing_fields": list(final.get("missing_fields", [])),
            "fallback_reason": final.get("fallback_reason"),
        },
        "turns": turns,
        "raw_dialogue": list(final.get("raw_dialogue", [])),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    nat = sum(float(r["naturalness"]["naturalness_score"]) for r in results) / total if total else 0.0
    acc = sum(float(r["accuracy"]["accuracy"]) for r in results) / total if total else 0.0
    strict_pass = sum(1 for r in results if bool(r["accuracy"]["pass"]))
    llm_ratio = sum(float(r["llm_used_turn_ratio"]) for r in results) / total if total else 0.0
    return {
        "total_cases": total,
        "avg_naturalness_score": nat,
        "avg_accuracy_score": acc,
        "strict_pass_rate": strict_pass / total if total else 0.0,
        "avg_llm_used_turn_ratio": llm_ratio,
        "failed_cases": [r["id"] for r in results if not r["accuracy"]["pass"]],
    }


def _to_md(config: dict[str, Any], conn: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Ollama 开启回归报告（多轮）")
    lines.append("")
    lines.append("## 运行配置")
    lines.append(f"- config_path: {config.get('config_path')}")
    lines.append(f"- provider: {config.get('provider')}")
    lines.append(f"- base_url: {config.get('base_url')}")
    lines.append(f"- model: {config.get('model')}")
    lines.append(f"- min_confidence: {config.get('min_confidence')}")
    lines.append("")
    lines.append("## 连接探针")
    lines.append(f"- connectivity_ok: {conn.get('ok')}")
    lines.append(f"- model_count: {conn.get('model_count')}")
    lines.append(f"- models(sample): {conn.get('models')}")
    lines.append(f"- error: {conn.get('error') or 'none'}")
    lines.append("")
    lines.append("## 总体指标")
    lines.append(f"- 用例数: {summary.get('total_cases')}")
    lines.append(f"- 语言自然化平均分: {summary.get('avg_naturalness_score'):.3f}")
    lines.append(f"- 准确性平均分: {summary.get('avg_accuracy_score'):.3f}")
    lines.append(f"- 严格通过率: {summary.get('strict_pass_rate'):.2%}")
    lines.append(f"- LLM 使用回合占比: {summary.get('avg_llm_used_turn_ratio'):.2%}")
    lines.append(f"- 未通过用例: {summary.get('failed_cases')}")
    lines.append("")
    lines.append("## 逐例（含 raw dialogue）")
    for item in results:
        lines.append(f"### {item['id']}")
        lines.append(f"- naturalness: {item['naturalness']}")
        lines.append(f"- accuracy: {item['accuracy']['accuracy']:.3f} ({item['accuracy']['hits']}/{item['accuracy']['total']})")
        lines.append(f"- strict_pass: {item['accuracy']['pass']}")
        lines.append(f"- llm_used_turn_ratio: {item['llm_used_turn_ratio']:.2f}")
        lines.append("- turns:")
        for t in item["turns"]:
            lines.append(f"  - turn {t['turn']} user: {t['user']}")
            lines.append(f"    assistant: {t['assistant']}")
            lines.append(
                f"    action={t['dialogue_action']}, llm_used={t['llm_used']}, fallback={t['fallback_reason']}, missing={t['missing_fields']}"
            )
        lines.append("- raw_dialogue:")
        lines.append("```json")
        lines.append(json.dumps(item["raw_dialogue"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiturn regression with Ollama enabled.")
    parser.add_argument("--outdir", default="docs")
    parser.add_argument("--config", default="nlu/bert_lab/configs/ollama_config.json")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = _probe_provider(cfg)
    results = [_run_case(c, ollama_config=args.config, min_confidence=args.min_confidence, lang=args.lang) for c in _cases()]
    summary = _aggregate(results)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    payload = {
        "run_config": {
            "config_path": args.config,
            "provider": cfg.provider,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "min_confidence": args.min_confidence,
            "lang": args.lang,
        },
        "connectivity": conn,
        "summary": summary,
        "results": results,
    }
    json_path = outdir / f"multiturn_regression_ollama_{stamp}.json"
    md_path = outdir / f"multiturn_regression_ollama_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_md(payload["run_config"], conn, summary, results), encoding="utf-8")
    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

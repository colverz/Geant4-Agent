from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from nlu.llm_support.ollama_client import chat, extract_json
from tools.evaluate_geant4_agent_benchmark import DEFAULT_BENCHMARK_PATH, validate_benchmark_shape


REVIEW_SCHEMA_VERSION = "geant4_agent_benchmark_review.v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_benchmark_payload(path: Path) -> dict[str, Any]:
    cases = _load_json(path)
    compact_cases: list[dict[str, Any]] = []
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict):
            continue
        compact_cases.append(
            {
                "id": case.get("id"),
                "suite": case.get("suite"),
                "difficulty": case.get("difficulty"),
                "lang": case.get("lang"),
                "capabilities": case.get("capabilities", []),
                "turns": [
                    {
                        "text": turn.get("text"),
                        "expected_trace": turn.get("expected_trace", {}),
                    }
                    for turn in case.get("turns", [])
                    if isinstance(turn, dict)
                ],
                "expected_runtime": case.get("expected_runtime", {}),
                "forbidden": case.get("forbidden", {}),
            }
        )
    return {"case_count": len(compact_cases), "cases": compact_cases}


def build_review_prompt(benchmark_path: Path = DEFAULT_BENCHMARK_PATH) -> str:
    shape_report = validate_benchmark_shape(benchmark_path)
    benchmark_payload = _compact_benchmark_payload(benchmark_path)
    prompt_payload = {
        "benchmark_shape_report": {
            "total": shape_report["total"],
            "failed": shape_report["failed"],
            "suite_counts": shape_report["suite_counts"],
            "difficulty_counts": shape_report["difficulty_counts"],
            "capability_counts": shape_report["capability_counts"],
        },
        "benchmark": benchmark_payload,
    }
    return (
        "You are reviewing a benchmark for a Geant4 simulation agent.\n"
        "The benchmark is hand-designed. Do not generate new cases in bulk.\n"
        "Evaluate whether the cases are necessary, comprehensive, non-dictionary, and measurable.\n"
        "Focus on benchmark quality, not model performance.\n"
        "Review stage: V1 shape gate and dry-run trajectory benchmark.\n"
        "Important distinction: runtime payload readiness can be evaluated without executing Geant4.\n"
        "A case may require must_have_runtime_payload=true and must_not_call_runtime=true at the same time; "
        "that means the agent should prepare a valid payload but must not launch an expensive runtime action.\n"
        "Real runtime execution is intentionally opt-in and may be listed as a future gap, "
        "but should not be treated as a contradiction in V1.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        f'  "schema_version": "{REVIEW_SCHEMA_VERSION}",\n'
        '  "overall_score": 0.0,\n'
        '  "passes_review": false,\n'
        '  "dimension_scores": {\n'
        '    "necessary": 0.0,\n'
        '    "comprehensive": 0.0,\n'
        '    "non_dictionary": 0.0,\n'
        '    "measurable": 0.0,\n'
        '    "p7_readiness": 0.0\n'
        "  },\n"
        '  "major_issues": [\n'
        '    {"severity": "high|medium|low", "issue": "...", "affected_cases": ["case_id"]}\n'
        "  ],\n"
        '  "missing_capabilities": ["..."],\n'
        '  "recommended_changes": [\n'
        '    {"priority": "p0|p1|p2", "change": "...", "reason": "..."}\n'
        "  ],\n"
        '  "case_feedback": [\n'
        '    {"id": "case_id", "verdict": "keep|revise|remove", "reason": "..."}\n'
        "  ]\n"
        "}\n"
        "Scoring guidance: 1.0 means strong, 0.7 means acceptable with minor gaps, "
        "below 0.6 means the benchmark should be revised before live LLM comparison.\n"
        "Benchmark payload:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def _parse_review_response(raw_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    parsed = extract_json(raw_text)
    if parsed is None:
        return None, ["invalid_json_response"]
    if parsed.get("schema_version") != REVIEW_SCHEMA_VERSION:
        errors.append(f"invalid_schema_version:{parsed.get('schema_version')}")
    if not isinstance(parsed.get("dimension_scores"), dict):
        errors.append("dimension_scores_not_object")
    if not isinstance(parsed.get("major_issues"), list):
        errors.append("major_issues_not_list")
    if not isinstance(parsed.get("recommended_changes"), list):
        errors.append("recommended_changes_not_list")
    if not isinstance(parsed.get("case_feedback"), list):
        errors.append("case_feedback_not_list")
    return parsed, errors


def _with_model_override(model: str):
    class _ModelOverride:
        def __enter__(self):
            self.previous = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
            if model:
                os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = model
            return self

        def __exit__(self, exc_type, exc, tb):
            if model:
                if self.previous is None:
                    os.environ.pop("GEANT4_LLM_MODEL_OVERRIDE", None)
                else:
                    os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = self.previous

    return _ModelOverride()


def _with_timeout_override(timeout_s: int | None):
    class _TimeoutOverride:
        def __enter__(self):
            self.previous = os.environ.get("GEANT4_LLM_TIMEOUT_S")
            if timeout_s is not None:
                os.environ["GEANT4_LLM_TIMEOUT_S"] = str(int(timeout_s))
            return self

        def __exit__(self, exc_type, exc, tb):
            if timeout_s is not None:
                if self.previous is None:
                    os.environ.pop("GEANT4_LLM_TIMEOUT_S", None)
                else:
                    os.environ["GEANT4_LLM_TIMEOUT_S"] = self.previous

    return _TimeoutOverride()


def review_benchmark_with_llm(
    *,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    llm_config_path: str,
    models: list[str],
    live_llm: bool = False,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    shape_report = validate_benchmark_shape(benchmark_path)
    if shape_report["failed"]:
        return {
            "name": "geant4_agent_benchmark_llm_review",
            "schema_version": REVIEW_SCHEMA_VERSION,
            "ok": False,
            "benchmark": str(benchmark_path),
            "shape_report": shape_report,
            "reviews": [],
            "errors": ["shape_validation_failed"],
        }
    if not live_llm:
        return {
            "name": "geant4_agent_benchmark_llm_review",
            "schema_version": REVIEW_SCHEMA_VERSION,
            "ok": True,
            "mode": "dry_run",
            "benchmark": str(benchmark_path),
            "shape_report": shape_report,
            "reviews": [],
            "errors": [],
        }
    if not llm_config_path:
        return {
            "name": "geant4_agent_benchmark_llm_review",
            "schema_version": REVIEW_SCHEMA_VERSION,
            "ok": False,
            "mode": "live_llm",
            "benchmark": str(benchmark_path),
            "shape_report": shape_report,
            "reviews": [],
            "errors": ["missing_llm_config_path"],
        }

    prompt = build_review_prompt(benchmark_path)
    selected_models = models or [""]
    reviews: list[dict[str, Any]] = []
    errors: list[str] = []
    for model in selected_models:
        with _with_model_override(model):
            with _with_timeout_override(timeout_s):
                try:
                    response = chat(prompt, config_path=llm_config_path, temperature=0.0)
                    raw_text = str(response.get("response", ""))
                except Exception as exc:  # pragma: no cover - live network path
                    reviews.append({"model": model, "ok": False, "errors": [f"llm_call_failed:{type(exc).__name__}"]})
                    errors.append(f"{model or '<config_model>'}:llm_call_failed")
                    continue
        parsed, parse_errors = _parse_review_response(raw_text)
        reviews.append(
            {
                "model": model,
                "ok": not parse_errors,
                "errors": parse_errors,
                "review": parsed,
                "raw_preview": raw_text[:1000] if parsed is None else "",
            }
        )
        if parse_errors:
            errors.append(f"{model or '<config_model>'}:invalid_review")

    return {
        "name": "geant4_agent_benchmark_llm_review",
        "schema_version": REVIEW_SCHEMA_VERSION,
        "ok": not errors,
        "mode": "live_llm",
        "benchmark": str(benchmark_path),
        "shape_report": shape_report,
        "reviews": reviews,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask one or more live LLMs to review Geant4Agent benchmark quality.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--llm-config", default=os.environ.get("GEANT4_LLM_CONFIG", ""))
    parser.add_argument("--model", action="append", default=[], help="Model override. Can be provided multiple times.")
    parser.add_argument("--timeout-s", type=int, default=None, help="Optional live LLM request timeout override in seconds.")
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = review_benchmark_with_llm(
        benchmark_path=args.benchmark,
        llm_config_path=args.llm_config,
        models=list(args.model or []),
        live_llm=bool(args.live_llm),
        timeout_s=args.timeout_s,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{report['name']}: ok={report['ok']} mode={report.get('mode', 'unknown')}")
        for review in report.get("reviews", []):
            print(f"  model={review.get('model') or '<config_model>'} ok={review.get('ok')}")
            if review.get("errors"):
                print(f"    errors={review['errors']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

import sys
import time

from tools.check_deepseek_connection import _run_worker_process, validate_deepseek_response


def test_real_response_requires_nonce_request_id_model_and_usage() -> None:
    nonce = "geant4-agent-test"
    report = validate_deepseek_response(
        {
            "response": nonce,
            "provider_raw": {
                "id": "req-123",
                "model": "deepseek-chat",
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        },
        nonce,
    )

    assert report["ok"] is True
    assert report["nonce_verified"] is True


def test_fallback_text_cannot_pass_as_real_deepseek_response() -> None:
    report = validate_deepseek_response({"response": "local fallback"}, "geant4-agent-test")

    assert report["ok"] is False
    assert "nonce_not_returned" in report["errors"]
    assert "provider_request_id_missing" in report["errors"]
    assert "provider_usage_missing" in report["errors"]


def test_worker_has_a_hard_total_deadline() -> None:
    started = time.monotonic()
    _, timed_out = _run_worker_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        total_timeout_s=0.2,
    )

    assert timed_out is True
    assert time.monotonic() - started < 3

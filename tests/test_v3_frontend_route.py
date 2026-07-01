from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_default_send_path_uses_v3_agent_turn() -> None:
    app_js = (ROOT / "ui" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'agentMode' not in app_js or 'agentMode: "v3"' in app_js
    assert 'postJson("/api/v3/agent/turn"' in app_js
    assert 'postJson("/api/v3/agent/state"' in app_js
    assert "sendTurn" in app_js
    assert "llm_design_enabled: true" in app_js
    assert "deepseek_api.local.json" not in app_js
    assert "loadRuntimeConfig()" in app_js


def test_frontend_v3_path_does_not_call_legacy_intent_before_turn() -> None:
    app_js = (ROOT / "ui" / "web" / "app.js").read_text(encoding="utf-8")
    assert "classifyIntent" not in app_js
    assert "requestDesign" not in app_js
    assert "runGeant4(" not in app_js


def test_frontend_keeps_multiturn_controls_wired() -> None:
    app_js = (ROOT / "ui" / "web" / "app.js").read_text(encoding="utf-8")
    assert "ensureSessionId()" in app_js
    assert "readEvents()" in app_js
    assert "querySelectorAll(\".quick-btn\")" in app_js
    assert "showSuggestions(data)" in app_js
    assert "normalizeSuggestion" in app_js
    assert "data?.suggestions" in app_js
    assert "refreshV3AgentState()" in app_js
    assert "auto_discover_runtime: true" in app_js
    assert "allow_in_memory: false" in app_js
    assert "document.createElement(\"button\")" in app_js
    assert "suggestion-btn btn-ghost sm" in app_js
    assert "button.dataset.prefill = item.prefill" in app_js
    assert "button.setAttribute(\"aria-label\", item.prefill)" in app_js
    assert "confirmation_event" in app_js
    assert "decision: \"confirm\"" in app_js
    assert "confirmationEvent" in app_js
    assert "extraPayload.confirmation_event" in app_js


def test_frontend_suggestion_buttons_are_centered_and_stable() -> None:
    css = (ROOT / "ui" / "web" / "style.css").read_text(encoding="utf-8")
    assert ".suggestion-bar" in css
    assert "justify-content:center" in css
    assert ".suggestion-bar .suggestion-btn" in css
    assert "display:inline-flex" in css
    assert "text-align:center" in css
    assert "white-space:normal" in css

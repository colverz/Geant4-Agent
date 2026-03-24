from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

from ui.web.geant4_api import geant4_state_payload
from ui.web.request_router import handle_post_request, is_supported_post_path
from ui.web.runtime_state import runtime_config_payload as _runtime_config_payload
from ui.web.strict_api import handle_strict_step


ROOT = Path(__file__).parent


def _load_legacy_api():
    from ui.web.legacy_api import SESSIONS, legacy_solve, legacy_step

    return SESSIONS, legacy_solve, legacy_step


def _respond(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _load_file(path: Path) -> bytes:
    return path.read_bytes()


def step(payload: Dict[str, Any], progress_cb=None) -> Dict[str, Any]:
    text = str(payload.get('text', '')).strip()
    session_id = payload.get('session_id')
    llm_router = bool(payload.get('llm_router', True))
    llm_question = bool(payload.get('llm_question', True))
    normalize_input = bool(payload.get('normalize_input', True))
    min_conf = float(payload.get('min_confidence', 0.6))
    autofix = bool(payload.get('autofix', False))
    lang = str(payload.get('lang', 'zh')).lower()
    strict_mode = bool(payload.get('strict_mode', True))

    if strict_mode:
        return handle_strict_step(
            {
                'text': text,
                'session_id': session_id,
                'llm_router': llm_router,
                'llm_question': llm_question,
                'normalize_input': normalize_input,
                'autofix': autofix,
                'lang': lang,
                'min_confidence': min_conf,
            },
            progress_cb=progress_cb,
        )

    _, _, legacy_step = _load_legacy_api()
    return legacy_step(payload)


def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
    _, legacy_solve, _ = _load_legacy_api()
    return legacy_solve(payload)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?', 1)[0].strip('/')
        if path == '':
            data = _load_file(ROOT / 'index.html')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == 'api/runtime':
            _respond(self, 200, _runtime_config_payload())
            return
        if path == 'api/geant4/state':
            _respond(self, 200, geant4_state_payload())
            return
        if path in {'style.css', 'app.js'}:
            data = _load_file(ROOT / path)
            mime = 'text/css' if path.endswith('.css') else 'application/javascript'
            self.send_response(200)
            self.send_header('Content-Type', f'{mime}; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if not is_supported_post_path(path):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode('utf-8')) if raw else {}
        except json.JSONDecodeError:
            _respond(self, 400, {'error': 'invalid json'})
            return

        status, out = handle_post_request(
            path,
            payload,
            legacy_sessions=None if path != "/api/reset" else _load_legacy_api()[0],
            solve_fn=solve,
            step_fn=step,
        )
        _respond(self, status, out)


def main() -> None:
    host = '127.0.0.1'
    port = 8088
    runtime = _runtime_config_payload()
    print(f'Serving on http://{host}:{port}')
    print(
        'Ollama runtime:',
        f"provider={runtime.get('current_provider', '')};",
        f"path={runtime.get('current_path', '')};",
        f"model={runtime.get('current_model', '')};",
        f"base_url={runtime.get('current_base_url', '')}",
    )
    preflight = runtime.get("model_preflight", {})
    print(
        "Model preflight:",
        f"ready={preflight.get('ready', False)};",
        f"structure_ok={preflight.get('structure', {}).get('ok', False)};",
        f"ner_ok={preflight.get('ner', {}).get('ok', False)}",
    )
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    main()

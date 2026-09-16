#!/usr/bin/env python3
"""Local Ra-Thor wrap shim.

Injects wrappers/system-prompt.txt into POST /v1/chat/completions and
forwards to the operator's upstream. No keys in this repo.

  export RATHOR_UPSTREAM=http://localhost:11434/v1
  export RATHOR_UPSTREAM_KEY=   # optional Bearer; operator-held
  python3 wrappers/local-shim/rathor_wrap.py
  # POST http://127.0.0.1:8787/v1/chat/completions  (stream true or false)
  # GET  http://127.0.0.1:8787/v1/models

stream=false: JSON forward. stream=true: byte-forward SSE (text/event-stream).
GET /v1/models: proxy upstream /models when present, else a one-item stub.

Not a public rathor.ai product. Drafts only. Workspace 14.15.6.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = os.environ.get("RATHOR_WRAP_HOST", "127.0.0.1")
PORT = int(os.environ.get("RATHOR_WRAP_PORT", "8787"))
UPSTREAM = os.environ.get("RATHOR_UPSTREAM", "http://localhost:11434/v1").rstrip("/")
UPSTREAM_KEY = os.environ.get("RATHOR_UPSTREAM_KEY", "").strip()

HERE = Path(__file__).resolve().parent
PROMPT_CANDIDATES = [
    HERE.parent / "system-prompt.txt",
    HERE.parent.parent / "wrappers" / "system-prompt.txt",
]


def load_constitution() -> str:
    for path in PROMPT_CANDIDATES:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return (
        "You are sitting under the Ra-Thor employ loop (workspace 14.15.6). "
        "Outputs are drafts. inspect \u2260 METR. Independent of xAI. Contact info@Rathor.ai."
    )


CONSTITUTION = load_constitution()
MODELS_STUB = {
    "object": "list",
    "data": [{"id": "rathor-wrap", "owned_by": "operator"}],
}


def inject(messages):
    if not isinstance(messages, list):
        messages = []
    already = any(
        isinstance(m, dict)
        and m.get("role") == "system"
        and "Ra-Thor employ loop" in str(m.get("content", ""))
        for m in messages
    )
    if already:
        return messages
    return [{"role": "system", "content": CONSTITUTION}] + messages


def _upstream_headers(content_type: str | None = None) -> dict:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    if UPSTREAM_KEY:
        headers["Authorization"] = "Bearer " + UPSTREAM_KEY
    return headers


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("rathor-wrap: " + (fmt % args) + "\n")

    def _stamp(self):
        self.send_header("X-Ra-Thor-Draft", "true")
        self.send_header("X-Ra-Thor-Workspace", "14.15.6")

    def _send(self, code: int, body: bytes, content_type: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._stamp()
        self.end_headers()
        self.wfile.write(body)

    def _forward_sse(self, resp):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._stamp()
        self.end_headers()
        try:
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _proxy_models(self):
        url = UPSTREAM + "/models"
        req = Request(url, headers=_upstream_headers(), method="GET")
        try:
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
            self._send(200, data)
        except HTTPError as err:
            try:
                err.read()
            except Exception:
                pass
            self._send(200, json.dumps(MODELS_STUB).encode("utf-8"))
        except (URLError, TimeoutError, OSError):
            self._send(200, json.dumps(MODELS_STUB).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/health", "/v1/health"):
            payload = {
                "ok": True,
                "service": "rathor-wrap",
                "workspace": "14.15.6",
                "upstream": UPSTREAM,
                "drafts": True,
                "claim": "inspect != METR; independent of xAI",
            }
            self._send(200, json.dumps(payload).encode("utf-8"))
            return
        if path in ("/v1/models", "/models"):
            self._proxy_models()
            return
        self._send(404, b'{"error":"not found"}')

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        if not self.path.split("?", 1)[0].rstrip("/").endswith("/chat/completions"):
            self._send(404, b'{"error":"use POST /v1/chat/completions"}')
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}')
            return
        body["messages"] = inject(body.get("messages") or [])
        body.setdefault("stream", False)
        stream = bool(body.get("stream"))
        url = UPSTREAM + "/chat/completions"
        req = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=_upstream_headers("application/json"),
            method="POST",
        )
        try:
            resp = urlopen(req, timeout=120)
        except HTTPError as err:
            self._send(err.code, err.read() or b'{"error":"upstream"}')
            return
        except URLError as err:
            payload = {"error": "upstream unreachable", "detail": str(err.reason), "url": url}
            self._send(502, json.dumps(payload).encode("utf-8"))
            return
        try:
            if stream:
                self._forward_sse(resp)
            else:
                self._send(200, resp.read())
        finally:
            resp.close()


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"rathor-wrap 14.15.6 on http://{HOST}:{PORT}/v1  ->  {UPSTREAM}",
        file=sys.stderr,
    )
    print("drafts only. keys stay with the operator.", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

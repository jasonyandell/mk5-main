"""Dependency-light HTTP server for the Burl microscope.

Run from repo root:

    python -m burl.microscope.server

The server intentionally uses Python's stdlib HTTP stack so the Pi extension can
spawn it from a plain project checkout. Heavy optional dependencies only enter
when a model turn actually runs (MLX-LM) or a harvested case is loaded (Burl's
existing oracle/tool stack).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from burl.microscope.core import (
    DEFAULT_HARVEST,
    MicroscopeSession,
    _jsonable_step,
    list_recipes,
    load_case,
    load_recipe,
    load_registry,
)

_state: dict[str, Any] = {
    "registry": load_registry(),
    "sessions": {},
    "engine": None,
    "engine_key": None,
}


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class MicroscopeHandler(BaseHTTPRequestHandler):
    server_version = "BurlMicroscope/0.1"

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("BURL_MICROSCOPE_QUIET") == "1":
            return
        super().log_message(fmt, *args)

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if method == "GET" and path == "/api/health":
                self._send_json(health())
            elif method == "GET" and path == "/api/recipes":
                self._send_json({"recipes": list_recipes()})
            elif method == "GET" and path == "/api/tools":
                self._send_json(tools_json())
            elif method == "POST" and path == "/api/sessions":
                self._send_json(create_session(self._read_json()))
            elif method == "POST" and path == "/api/shutdown":
                self._send_json({"ok": True, "shutting_down": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            elif method == "GET" and _match(path, "/api/sessions/{sid}"):
                sid = path.split("/")[3]
                self._send_json(session_json(_session(sid)))
            elif method == "GET" and _match(path, "/api/sessions/{sid}/prompt"):
                sid = path.split("/")[3]
                session = _session(sid)
                self._send_json({"session_id": sid, "prompt": session.prompt_summary()})
            elif method == "GET" and _match(path, "/api/sessions/{sid}/events"):
                sid = path.split("/")[3]
                tail = int((query.get("tail") or [80])[0])
                self._send_json(events_json(_session(sid), tail=tail))
            elif method == "POST" and _match(path, "/api/sessions/{sid}/tools"):
                sid = path.split("/")[3]
                body = self._read_json()
                session = _session(sid)
                session.set_tools([str(name) for name in body.get("names", [])])
                self._send_json(session_json(session))
            elif method == "POST" and _match(path, "/api/sessions/{sid}/step"):
                sid = path.split("/")[3]
                body = self._read_json()
                session = _session(sid)
                engine = _engine_for(session)
                result = asyncio.run(
                    session.step(
                        engine,
                        user_text=str(body.get("text", "")),
                        max_tokens=body.get("max_tokens"),
                    )
                )
                self._send_json(_jsonable_step(result))
            elif method == "POST" and _match(path, "/api/sessions/{sid}/auto"):
                sid = path.split("/")[3]
                body = self._read_json()
                session = _session(sid)
                engine = _engine_for(session)
                results = asyncio.run(
                    session.auto(
                        engine,
                        user_text=str(body.get("text", "")),
                        max_steps=int(body.get("max_steps", 8)),
                    )
                )
                self._send_json(
                    {
                        "session_id": sid,
                        "results": [_jsonable_step(result) for result in results],
                        "committed": session.committed,
                        "outcome": session.outcome(),
                    }
                )
            else:
                self._send_json({"detail": f"No route for {method} {path}"}, status=HTTPStatus.NOT_FOUND)
        except HttpError as exc:
            self._send_json({"detail": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"detail": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n <= 0:
            return {}
        raw = self.rfile.read(n).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def _send_json(self, payload: dict[str, Any], *, status: int | HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str, indent=2).encode("utf-8")
        self.send_response(int(status))
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


class HttpError(Exception):
    def __init__(self, status: int | HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def health() -> dict[str, Any]:
    engine = _state.get("engine")
    registry = _state["registry"]
    return {
        "ok": True,
        "engine_loaded": engine is not None,
        "engine_key": _state.get("engine_key"),
        "n_tools": len(registry),
        "sessions": len(_state["sessions"]),
    }


def tools_json() -> dict[str, Any]:
    registry = _state["registry"]
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "protocol_role": spec.protocol_role,
                "protocol_phrase": spec.protocol_phrase,
                "requires_context": spec.requires_context,
            }
            for spec in registry.active()
        ]
    }


def create_session(body: dict[str, Any]) -> dict[str, Any]:
    try:
        recipe_name = str(body.get("recipe") or "baseline")
        harvest = str(body.get("harvest") or DEFAULT_HARVEST)
        seed = body.get("seed")
        idx = body.get("idx", 1) if seed is None else None
        recipe = load_recipe(recipe_name)
        case = load_case(
            harvest,
            idx=int(idx) if idx is not None else None,
            seed=int(seed) if seed is not None else None,
        )
        session = MicroscopeSession(case=case, recipe=recipe, registry=_state["registry"])
    except Exception as exc:  # noqa: BLE001
        raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    _state["sessions"][session.sid] = session
    return session_json(session)


def _session(sid: str) -> MicroscopeSession:
    session = _state["sessions"].get(sid)
    if session is None:
        raise HttpError(HTTPStatus.NOT_FOUND, f"unknown session {sid!r}")
    return session


def session_json(session: MicroscopeSession) -> dict[str, Any]:
    return {
        "session_id": session.sid,
        "dir": str(session.dir),
        "case": session.case_summary(),
        "recipe": session.recipe_summary(),
        "committed": session.committed,
        "outcome": session.outcome() if session.committed else None,
    }


def events_json(session: MicroscopeSession, *, tail: int) -> dict[str, Any]:
    path = session.dir / "events.jsonl"
    if not path.exists():
        return {"session_id": session.sid, "events": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    if tail > 0:
        lines = lines[-tail:]
    return {"session_id": session.sid, "events": [line for line in lines if line]}


def _engine_for(session: MicroscopeSession) -> Any:
    model_repo = str(
        session.recipe.params.get("model_repo")
        or os.environ.get("BURL_MICROSCOPE_MODEL_REPO")
        or "mlx-community/gemma-4-e2b-it-bf16"
    )
    adapter = session.recipe.params.get("adapter_path")
    if adapter is None:
        adapter = os.environ.get("BURL_MICROSCOPE_ADAPTER_PATH") or None
    adapter_path = str(adapter) if adapter else None
    key = (model_repo, adapter_path)
    if _state.get("engine") is not None and _state.get("engine_key") == key:
        return _state["engine"]

    from burl.lab.core.engine import MlxEngine

    _state["engine"] = MlxEngine(model_repo=model_repo, adapter_path=adapter_path)
    _state["engine_key"] = key
    return _state["engine"]


def _match(path: str, pattern: str) -> bool:
    path_parts = path.strip("/").split("/")
    pattern_parts = pattern.strip("/").split("/")
    if len(path_parts) != len(pattern_parts):
        return False
    return all(pp.startswith("{") and pp.endswith("}") or pp == p for p, pp in zip(path_parts, pattern_parts))


def main() -> None:
    host = os.environ.get("BURL_MICROSCOPE_HOST", "127.0.0.1")
    port = int(os.environ.get("BURL_MICROSCOPE_PORT", "8765"))
    httpd = ReusableThreadingHTTPServer((host, port), MicroscopeHandler)
    print(f"Burl microscope listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

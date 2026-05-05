"""Small client for LM Studio's native stateful chat endpoint.

LM Studio owns the chat runtime. burl-lab owns the replayable wrapper:
the server calls this module, then journals the exact request/response ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class LmStudioConfig:
    base_url: str
    api_token: str | None = None
    timeout_s: float = 300.0


class LmStudioClientError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body

    def detail(self) -> dict:
        return {"status": self.status, "body": self.body}


def chat(config: LmStudioConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """POST one request to ``/api/v1/chat`` and return decoded JSON."""

    base = config.base_url.rstrip("/")
    url = f"{base}/api/v1/chat"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if config.api_token:
        headers["Authorization"] = f"Bearer {config.api_token}"

    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=config.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise LmStudioClientError(
            f"LM Studio returned HTTP {exc.code}",
            status=exc.code,
            body=raw,
        ) from exc
    except error.URLError as exc:
        raise LmStudioClientError(f"LM Studio request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LmStudioClientError("LM Studio request timed out") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LmStudioClientError("LM Studio returned non-JSON", body=raw) from exc
    if not isinstance(parsed, dict):
        raise LmStudioClientError("LM Studio returned a non-object JSON payload")
    return parsed


__all__ = ["LmStudioConfig", "LmStudioClientError", "chat"]

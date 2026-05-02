"""SSE emitter with CRLF awareness.

The Server-Sent Events spec uses ``\\n\\n`` (LF LF) as the message terminator,
but in burl/chat we hit a real bug where tool outputs embedded ``\\r``
characters that the terminal silently hid — but the SSE parser saw
``\\r\\n\\r\\n``. Most parsers treat that as the same record terminator,
*except* when a stray ``\\r`` snuck in mid-data and broke streaming.

Two simple invariants:

1. **On send:** strip any ``\\r`` from data lines. We never want a stray CR
   inside a JSON payload.
2. **Frame terminator:** emit a clean ``\\n\\n``; never emit ``\\r\\n\\r\\n``.

If you ever need to *parse* SSE on the client side, split on
``\\r?\\n\\r?\\n`` to be safe — but on the server we just normalise.

The bug was invisible until somebody dumped the raw bytes via ``repr()``.
That's what this module exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator


def encode_event(payload: Any, event: str | None = None) -> str:
    """Encode a single SSE frame. Returns a string ending in ``\\n\\n``."""
    lines: list[str] = []
    if event:
        lines.append(f"event: {_strip_cr(event)}")
    if isinstance(payload, str):
        data = _strip_cr(payload)
    else:
        data = _strip_cr(_to_json(payload))
    for chunk in data.split("\n"):
        lines.append(f"data: {chunk}")
    return "\n".join(lines) + "\n\n"


def _strip_cr(s: str) -> str:
    return s.replace("\r", "")


def _to_json(value: Any) -> str:
    """JSON-serialise a Move (frozen dataclass) or any compatible value."""
    if is_dataclass(value):
        return json.dumps(asdict(value), default=str)
    return json.dumps(value, default=str)


async def sse_from_async_iter(
    moves: AsyncIterator[Any],
    on_each: Any = None,
) -> AsyncIterator[dict]:
    """Bridge an async iterator of Moves into ``sse_starlette``-compatible
    ``{"data": ...}`` dicts.

    ``on_each(move)`` is called for each Move *before* it is yielded — used
    by the server to append to ``events.jsonl``. Failures in ``on_each``
    are swallowed (logged elsewhere) so a disk hiccup doesn't kill the
    SSE stream.
    """
    async for move in moves:
        if on_each is not None:
            try:
                on_each(move)
            except Exception:  # noqa: BLE001
                pass
        yield {"data": _strip_cr(_to_json(move))}


__all__ = ["encode_event", "sse_from_async_iter"]

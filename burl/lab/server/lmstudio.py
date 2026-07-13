"""Small client for LM Studio's Python SDK agent loop.

LM Studio owns inference. burl-lab owns the agent loop, tools, callbacks, and
replayable journal wrapper.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse
from urllib import error, request

from burl.lab.core.tool import ToolSpec


@dataclass(frozen=True)
class LmStudioConfig:
    base_url: str
    api_token: str | None = None
    timeout_s: float = 300.0


class LmStudioClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        extra: dict | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body
        self.extra = extra or {}

    def detail(self) -> dict:
        return {"status": self.status, "body": self.body, **self.extra}


def act(
    config: LmStudioConfig,
    payload: dict[str, Any],
    tools: list[ToolSpec],
    *,
    ctx: Any,
) -> dict[str, Any]:
    """Run LM Studio's SDK ``model.act`` loop with Burl Lab tool functions."""

    try:
        import lmstudio as lms
    except ModuleNotFoundError as exc:
        raise LmStudioClientError(
            "lmstudio-python is not installed in the Burl Lab environment",
            extra={"install": "pip install lmstudio"},
        ) from exc

    events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    fragments: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    sdk_tools = _tool_functions(tools, ctx=ctx, tool_calls=tool_calls)

    def on_message(message: Any) -> None:
        messages.append(_message_to_dict(message))
        try:
            chat_obj.append(message)
        except Exception:  # noqa: BLE001
            pass

    def on_prediction_fragment(fragment: Any, round_index: int = 0) -> None:
        content = str(getattr(fragment, "content", fragment))
        fragments.append(content)
        events.append(
            {
                "type": "prediction_fragment",
                "round_index": round_index,
                "content": content,
            }
        )

    def on_round_start(round_index: int) -> None:
        events.append({"type": "round_start", "round_index": round_index})

    def on_round_end(round_index: int) -> None:
        events.append({"type": "round_end", "round_index": round_index})

    api_host = _api_host(config.base_url)
    try:
        if hasattr(lms, "configure_default_client"):
            lms.configure_default_client(api_host)

        model = lms.llm(str(payload["model"]))
        chat_obj = lms.Chat(str(payload.get("system_prompt") or ""))
        chat_obj.add_user_message(str(payload["input"]))

        result = model.act(
            chat_obj,
            sdk_tools,
            on_message=on_message,
            on_prediction_fragment=on_prediction_fragment,
            on_round_start=on_round_start,
            on_round_end=on_round_end,
            max_parallel_tool_calls=1,
        )
    except Exception as exc:  # noqa: BLE001
        raise LmStudioClientError(
            f"LM Studio SDK act failed: {exc}",
            extra={"api_host": api_host},
        ) from exc

    return {
        "run_id": uuid.uuid4().hex[:12],
        "api": "lmstudio-python.act",
        "api_host": api_host,
        "model": payload["model"],
        "output_text": "".join(fragments),
        "messages": messages,
        "events": events,
        "tool_calls": tool_calls,
        "result": _jsonable(result),
    }


def _tool_functions(
    specs: list[ToolSpec],
    *,
    ctx: Any,
    tool_calls: list[dict[str, Any]],
) -> list[Callable[..., str]]:
    by_name = {spec.name: spec for spec in specs}
    out: list[Callable[..., str]] = []

    def invoke(name: str, args: dict[str, Any]) -> str:
        spec = by_name[name]
        result = spec.impl(ctx, args)
        evidence = dict(result.evidence)
        tool_calls.append(
            {
                "call_id": uuid.uuid4().hex[:12],
                "name": name,
                "args": dict(args),
                "evidence": evidence,
                "next_tools": [tool.name for tool in result.next_tools],
                "next_phase": result.next_phase,
            }
        )
        return str(evidence.get("prose", ""))

    if "belief_trajectory" in by_name:
        def belief_trajectory() -> str:
            """Read the calibrated belief state before considering candidate plays."""
            return invoke("belief_trajectory", {})

        out.append(belief_trajectory)

    if "state_brief" in by_name:
        def state_brief() -> str:
            """Read the current decision in Burl's labeled game-state format."""
            return invoke("state_brief", {})

        out.append(state_brief)

    if "board_snapshot" in by_name:
        def board_snapshot() -> str:
            """Read the full board snapshot before choosing candidate plays."""
            return invoke("board_snapshot", {})

        out.append(board_snapshot)

    if "legal_plays" in by_name:
        def legal_plays() -> str:
            """List legal and illegal dominoes for the current trick."""
            return invoke("legal_plays", {})

        out.append(legal_plays)

    if "explore_game" in by_name:
        def explore_game(play: int) -> str:
            """Sample the outcome distribution for candidate domino_id play."""
            return invoke("explore_game", {"play": play})

        out.append(explore_game)

    if "play_brief" in by_name:
        def play_brief(play: int) -> str:
            """Summarize the outcome distribution for candidate domino_id play."""
            return invoke("play_brief", {"play": play})

        out.append(play_brief)

    if "commit_play" in by_name:
        def commit_play(domino_id: int) -> str:
            """Commit the final domino_id and end the decision."""
            return invoke("commit_play", {"domino_id": domino_id})

        out.append(commit_play)

    return out


def _api_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.netloc:
        return parsed.netloc
    return base_url.replace("http://", "").replace("https://", "").rstrip("/")


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    return {
        "type": type(message).__name__,
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", str(message)),
    }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


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


__all__ = ["LmStudioConfig", "LmStudioClientError", "act", "chat"]

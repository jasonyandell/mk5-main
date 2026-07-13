"""Opus agent for the candlewax spike — plain vs multimodal A/B.

Two run modes share everything except the user-message content:

- ``plain``      → text-only prompt. Opus must play from base-model priors
                   + the game-state description alone.
- ``multimodal`` → text prompt + candlewax PNG attached. Opus sees the same
                   PDF shapes that clicked for the other Claude instance.

Only tool exposed: ``commit_play(domino_id)``. The engine validates; illegal
attempts round-trip back for a retry up to ``max_retries``. No ``is_legal``,
no ``eq_outcome_distribution`` — the whole question is whether the image
substitutes for those.

Trace schema matches ``burl/haiku_spike/agent.py`` so downstream scorers can
read either.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

# In-process MCP races under parallel tool_use blocks — lock mirrored from
# haiku_spike (see burl/haiku_spike/agent.py:44-58).
from claude_agent_sdk._internal.query import Query as _Query

_MCP_DISPATCH_LOCK = asyncio.Lock()
_orig_handle_sdk_mcp = _Query._handle_sdk_mcp_request


async def _locked_handle_sdk_mcp(self, server_name, message):
    async with _MCP_DISPATCH_LOCK:
        return await _orig_handle_sdk_mcp(self, server_name, message)


_Query._handle_sdk_mcp_request = _locked_handle_sdk_mcp

from burl.tools import engine as engine_tools
from burl.candlewax_spike.render import _domino_label


# --------------------------------------------------------------------------- #
# Single-slot per-decision context (same pattern as haiku_spike).              #
# --------------------------------------------------------------------------- #


class _DecisionContext:
    def __init__(self) -> None:
        self.game_state: Any = None
        self.final_play: int | None = None

    def bind(self, game_state: Any) -> None:
        self.game_state = game_state
        self.final_play = None


_CTX = _DecisionContext()


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


# --------------------------------------------------------------------------- #
# Sole tool: commit_play.                                                      #
# --------------------------------------------------------------------------- #


@tool(
    "commit_play",
    "Commit to playing domino_id. This is your final answer. If illegal, "
    "you will be told why and may try a different domino.",
    {"domino_id": int},
)
async def _commit_play(args: dict[str, Any]) -> dict[str, Any]:
    dom = int(args["domino_id"])
    legal, reason = engine_tools.is_legal(_CTX.game_state, dom)
    if not legal:
        return _text({"ok": False, "reason": reason, "committed": False})
    _CTX.final_play = dom
    return _text({"ok": True, "committed": dom, "note": "Decision recorded."})


MCP_SERVER_NAME = "candlewax"
_MCP_TOOL_NAMES = (f"mcp__{MCP_SERVER_NAME}__commit_play",)


def _build_mcp_server() -> Any:
    return create_sdk_mcp_server(
        name=MCP_SERVER_NAME, version="0.1.0", tools=[_commit_play],
    )


# --------------------------------------------------------------------------- #
# Prompt rendering.                                                            #
# --------------------------------------------------------------------------- #


_CANDLEWAX_SYSTEM_PREAMBLE = (
    "You are Burl, a Texas 42 dominoes agent. Pick the next play.\n"
    "\n"
    "You have exactly one tool: `commit_play(domino_id)`. There is no "
    "is_legal tool — the legal plays are spelled out in the user turn. "
    "There is no eq_outcome_distribution tool — reason from what you're "
    "given and commit once you've decided. If your commit is illegal, the "
    "engine will tell you and you can try again.\n"
)


def _render_prompt(game_state: Any, legal_plays: list[int]) -> tuple[str, str]:
    """Return (system, user) text. 42-framing block reused from the native
    path; system preamble is trimmed because only ``commit_play`` exists."""
    from burl.harness.agent_runner import _current_player, _visible_history, _fmt_hand, _fmt_history
    from burl.harness.agent_runner_native import _render_42_framing

    me_abs = _current_player(game_state)
    hand_remaining = [d for d in game_state.hands[me_abs] if d not in game_state.played]
    history = _visible_history(game_state)
    trick_no = len(game_state.play_history) // 4 + 1
    position_in_trick = len(game_state.current_trick) + 1

    from burl.tools.engine import trump_declared
    decl = trump_declared(game_state)

    current_trick_ids: list[int] = []
    if game_state.current_trick:
        first = game_state.current_trick[0]
        if isinstance(first, int):
            current_trick_ids = list(game_state.current_trick)
        else:
            current_trick_ids = [d for _p, d in game_state.current_trick]
    trick_txt = (
        "current trick so far: "
        + ", ".join(f"{d}({_domino_label(d)})" for d in current_trick_ids)
        if current_trick_ids
        else "current trick: you are leading"
    )

    legal_txt = ", ".join(f"{p}({_domino_label(p)})" for p in legal_plays)

    system = (
        _CANDLEWAX_SYSTEM_PREAMBLE
        + "\n# Current decision — 42-aware context\n\n"
        + _render_42_framing(game_state, me_abs, hand_remaining)
    )

    user = (
        f"declaration: {decl}\n"
        f"your seat (absolute): {me_abs}\n"
        f"trick: {trick_no}   position in trick: {position_in_trick}/4\n"
        f"your hand: {_fmt_hand(hand_remaining)}\n"
        f"legal plays: {legal_txt}\n"
        f"{trick_txt}\n"
        f"visible history: {_fmt_history(history)}\n\n"
        "Think about the play, then commit via `commit_play(domino_id)` "
        "where domino_id is one of the legal plays above."
    )
    return system, user


def _user_content_for_mode(
    user_text: str,
    *,
    image_bytes: bytes | None,
) -> list[dict[str, Any]] | str:
    """Build the user-message content for one of the two modes.

    In plain mode we return the string directly — the Agent SDK accepts it.
    In multimodal mode we return a list of content blocks: an image followed
    by the text prompt. Image first is the convention that works best for
    Claude multimodal — the text then explicitly refers to "the image above".
    """
    if image_bytes is None:
        return user_text
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        },
        {
            "type": "text",
            "text": (
                "The image above is the per-play outcome distribution for "
                "each of your legal plays in this decision. Each row is one "
                "legal play; the histogram is the PDF of your final Q-score "
                "under sampled hidden-hand worlds. μ is the mean Q; p_make "
                "is the probability that play reaches the win threshold "
                "(vertical blue line). Use the shapes — bimodal tails, where "
                "the mass sits, where the threshold cuts — to decide.\n\n"
                + user_text
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Runner.                                                                      #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class CandlewaxRunResult:
    final_play: int
    events: list[dict[str, Any]]
    cost_usd: float
    num_turns: int
    duration_ms: int
    usage: dict[str, Any]
    is_error: bool
    mode: str


DEFAULT_MODEL = "claude-opus-4-7"

_DISALLOWED_BUILTINS = [
    "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task", "WebFetch",
    "WebSearch", "NotebookEdit", "TodoWrite", "SlashCommand", "MultiEdit",
    "KillShell", "BashOutput",
]


async def run_decision_candlewax(
    game_state: Any,
    legal_plays: list[int],
    *,
    mode: Literal["plain", "multimodal"],
    image_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 6,
    max_budget_usd: float = 0.40,
    decision_idx: int = 0,
    thinking_tokens: int | None = 4000,
) -> CandlewaxRunResult:
    """One decision through Opus. ``image_path`` required iff mode='multimodal'."""
    if mode == "multimodal":
        if image_path is None:
            raise ValueError("multimodal mode requires image_path")
        image_bytes = Path(image_path).read_bytes()
    else:
        image_bytes = None

    _CTX.bind(game_state)
    system_content, user_text = _render_prompt(game_state, legal_plays)
    user_content = _user_content_for_mode(user_text, image_bytes=image_bytes)

    mcp = _build_mcp_server()

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_content,
        mcp_servers={MCP_SERVER_NAME: mcp},
        allowed_tools=list(_MCP_TOOL_NAMES),
        disallowed_tools=list(_DISALLOWED_BUILTINS),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        setting_sources=[],
        max_thinking_tokens=thinking_tokens,
    )

    events: list[dict[str, Any]] = []
    events.append({
        "event": "prompt", "role": "system",
        "content": system_content, "decision_idx": decision_idx,
    })
    # User content: if it's a list (multimodal), serialize the text block only
    # to keep the trace small; the image lives on disk.
    if isinstance(user_content, list):
        text_only = next(
            (b["text"] for b in user_content if b.get("type") == "text"), ""
        )
        image_refs = [
            {"media_type": b["source"]["media_type"], "bytes": len(image_bytes)}
            for b in user_content if b.get("type") == "image"
        ]
        events.append({
            "event": "prompt", "role": "user",
            "content": text_only, "decision_idx": decision_idx,
            "images": image_refs,
        })
    else:
        events.append({
            "event": "prompt", "role": "user",
            "content": user_content, "decision_idx": decision_idx,
        })

    result_payload: dict[str, Any] = {
        "cost_usd": 0.0, "num_turns": 0, "duration_ms": 0,
        "usage": {}, "is_error": False,
    }

    async def _prompt_stream():
        # Must use streaming mode so the SDK MCP server can round-trip the
        # commit_play tool call back into this process.
        yield {"type": "user", "message": {"role": "user", "content": user_content}}

    async for msg in query(prompt=_prompt_stream(), options=options):
        if isinstance(msg, AssistantMessage):
            for blk in msg.content:
                if isinstance(blk, TextBlock):
                    if blk.text.strip():
                        events.append({"event": "assistant_text", "content": blk.text})
                elif isinstance(blk, ThinkingBlock):
                    events.append({"event": "thinking", "content": blk.thinking})
                elif isinstance(blk, ToolUseBlock):
                    events.append({
                        "event": "tool_call", "tool": blk.name,
                        "args": blk.input, "tool_use_id": blk.id,
                    })
        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, ToolResultBlock):
                        events.append({
                            "event": "tool_result",
                            "tool_use_id": blk.tool_use_id,
                            "content": blk.content,
                            "is_error": bool(blk.is_error),
                        })
        elif isinstance(msg, ResultMessage):
            result_payload = {
                "cost_usd": float(msg.total_cost_usd or 0.0),
                "num_turns": int(msg.num_turns or 0),
                "duration_ms": int(msg.duration_ms or 0),
                "usage": dict(msg.usage) if msg.usage else {},
                "is_error": bool(msg.is_error),
            }

    if _CTX.final_play is not None:
        events.append({"event": "commit", "final_play": int(_CTX.final_play)})
    events.append({"event": "result", **result_payload})

    return CandlewaxRunResult(
        final_play=int(_CTX.final_play) if _CTX.final_play is not None else -1,
        events=events,
        cost_usd=result_payload["cost_usd"],
        num_turns=result_payload["num_turns"],
        duration_ms=result_payload["duration_ms"],
        usage=result_payload["usage"],
        is_error=result_payload["is_error"],
        mode=mode,
    )

"""Haiku 4.5 agent over Burl's in-process tool surface.

Uses the Claude Agent SDK's in-process MCP server (`create_sdk_mcp_server`) to
expose burl/tools/{engine,eq_distribution}.py directly as tools — no subprocess,
no network hop for tool calls. The model runs via Claude Code's bundled CLI.

One decision = one `query()` call. We bind the current `game_state` to a
module-level holder before invoking the agent; the tool handlers read from it.
This is safe because decisions are run sequentially in this spike.

Trace shape (one event dict per line in the decision JSONL):

    {"event": "prompt", "role": "system" | "user", "content": str, "decision_idx": int}
    {"event": "assistant_text", "content": str}
    {"event": "thinking",       "content": str}
    {"event": "tool_call",      "tool": str, "args": dict, "tool_use_id": str}
    {"event": "tool_result",    "tool_use_id": str, "content": <json>, "is_error": bool}
    {"event": "commit",         "final_play": int}
    {"event": "result",         "cost_usd": float, "num_turns": int,
                                 "duration_ms": int, "usage": dict, "is_error": bool}
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

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

# Opus 4.7 emits parallel tool_use blocks in one assistant message; the Agent
# SDK spawns a concurrent handler task per block (query.py:196), and the
# MCP low-level server's shared tool cache isn't concurrency-safe — races
# wedge the CLI's MCP channel ("Stream closed", or silent termination at
# turn 1). Serialize in-process MCP dispatch with a process-global lock.
import asyncio as _asyncio
from claude_agent_sdk._internal.query import Query as _Query

_MCP_DISPATCH_LOCK = _asyncio.Lock()
_orig_handle_sdk_mcp = _Query._handle_sdk_mcp_request

async def _locked_handle_sdk_mcp(self, server_name, message):
    async with _MCP_DISPATCH_LOCK:
        return await _orig_handle_sdk_mcp(self, server_name, message)

_Query._handle_sdk_mcp_request = _locked_handle_sdk_mcp

from burl.tools import engine as engine_tools
from burl.tools import eq_distribution as eq_tools
from burl.tools.eq_distribution import ConditionUnreachable, OutcomeDistribution


# --------------------------------------------------------------------------- #
# Per-decision state holder — tools read the game_state from here.            #
# --------------------------------------------------------------------------- #


class _DecisionContext:
    """Single-slot holder. Set before each query(); tools pull from it."""

    def __init__(self) -> None:
        self.game_state: Any = None
        self.final_play: int | None = None

    def bind(self, game_state: Any) -> None:
        self.game_state = game_state
        self.final_play = None


_CTX = _DecisionContext()


# --------------------------------------------------------------------------- #
# Tool handlers. Each returns the MCP content shape the SDK expects.          #
# --------------------------------------------------------------------------- #


def _text(payload: Any) -> dict[str, Any]:
    """Wrap a JSON-safe payload as an MCP text content block."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _outcome_to_dict(od: OutcomeDistribution) -> dict[str, Any]:
    return {
        "play": int(od.play),
        "mean": round(float(od.mean), 3),
        "stdev": round(float(od.stdev), 3),
        "p_make": round(float(od.p_make), 4),
        "n_samples": int(od.n_samples),
        "min_q": round(float(od.min_q), 1),
        "max_q": round(float(od.max_q), 1),
        "percentiles": {str(k): round(float(v), 1) for k, v in od.percentiles.items()},
        "is_offense": bool(od.is_offense),
        "pdf_bins": [round(float(x), 6) for x in od.pdf_bins.tolist()],
        "distribution_shape": str(od.distribution_shape),
        "modes": [
            {"center": round(float(m["center"]), 2), "mass": round(float(m["mass"]), 4)}
            for m in od.modes
        ],
        "gap_between_modes": round(float(od.gap_between_modes), 2),
        "suggested_counterfactuals": list(od.suggested_counterfactuals),
        "spike_drivers": list(od.spike_drivers),
    }


@tool("is_legal", "Return {legal, reason} for playing domino_id right now.",
      {"domino_id": int})
async def _is_legal(args: dict[str, Any]) -> dict[str, Any]:
    legal, reason = engine_tools.is_legal(_CTX.game_state, int(args["domino_id"]))
    return _text({"legal": bool(legal), "reason": reason})


@tool("is_trump", "Return True if domino_id belongs to the declared trump suit.",
      {"domino_id": int})
async def _is_trump(args: dict[str, Any]) -> dict[str, Any]:
    return _text(bool(engine_tools.is_trump(_CTX.game_state, int(args["domino_id"]))))


@tool("unseen", "Return sorted domino_ids not in your hand and not yet played.", {})
async def _unseen(args: dict[str, Any]) -> dict[str, Any]:
    return _text(sorted(engine_tools.unseen(_CTX.game_state)))


@tool("void_audit",
      "Return True if player_seat is known void in this suit from play history. "
      "player_seat: 0=me 1=left opp 2=partner 3=right opp. suit: 0..6 pip, 7=called.",
      {"player_seat": int, "suit": int})
async def _void_audit(args: dict[str, Any]) -> dict[str, Any]:
    return _text(bool(engine_tools.void_audit(
        _CTX.game_state, int(args["player_seat"]), int(args["suit"]),
    )))


@tool("trump_declared",
      "Return the declared trump suit (blanks/ones/.../sixes/doubles-trump/doubles-suit/notrump).",
      {})
async def _trump_declared(args: dict[str, Any]) -> dict[str, Any]:
    return _text(engine_tools.trump_declared(_CTX.game_state))


@tool("eq_outcome_distribution",
      "Return outcome distribution (mean, stdev, p_make, pdf_bins, percentiles) "
      "for playing `play` (a domino_id), sampled over n_samples hidden-hand worlds.",
      {"play": int, "n_samples": int})
async def _eq_outcome_distribution(args: dict[str, Any]) -> dict[str, Any]:
    n = int(args.get("n_samples", 10) or 10)
    od = eq_tools.eq_outcome_distribution(_CTX.game_state, int(args["play"]), n_samples=n)
    return _text(_outcome_to_dict(od))


@tool("conditional_outcome",
      "Like eq_outcome_distribution but restricted to worlds satisfying `assumption`. "
      "assumption: {\"player\": abs_seat_int, \"holds\": domino_id_int} or "
      "{\"player\": abs_seat_int, \"void_in_suit\": suit_int}.",
      {"play": int, "assumption": dict, "n_samples": int})
async def _conditional_outcome(args: dict[str, Any]) -> dict[str, Any]:
    n = int(args.get("n_samples", 10) or 10)
    try:
        od = eq_tools.conditional_outcome(
            _CTX.game_state, int(args["play"]), args["assumption"], n_samples=n,
        )
    except ConditionUnreachable as e:
        return _text({"error": "ConditionUnreachable", "reason": str(e)})
    return _text(_outcome_to_dict(od))


@tool("commit_play",
      "Commit to the final play and end the decision. domino_id must be an "
      "integer from your hand. If the engine rejects it as illegal, you will "
      "be told and may call commit_play again with a different domino.",
      {"domino_id": int})
async def _commit_play(args: dict[str, Any]) -> dict[str, Any]:
    dom = int(args["domino_id"])
    legal, reason = engine_tools.is_legal(_CTX.game_state, dom)
    if not legal:
        return _text({"ok": False, "reason": reason, "committed": False})
    _CTX.final_play = dom
    return _text({"ok": True, "committed": dom,
                  "note": "Decision recorded. No more tool calls needed."})


_TOOL_HANDLERS = [
    _is_legal, _is_trump, _unseen, _void_audit, _trump_declared,
    _eq_outcome_distribution, _conditional_outcome, _commit_play,
]

MCP_SERVER_NAME = "burl"
_MCP_TOOL_NAMES = tuple(f"mcp__{MCP_SERVER_NAME}__{h.name}" for h in _TOOL_HANDLERS)


def build_mcp_server() -> Any:
    """Fresh in-process SDK MCP server with all 8 Burl tools."""
    return create_sdk_mcp_server(
        name=MCP_SERVER_NAME, version="0.1.0", tools=list(_TOOL_HANDLERS),
    )


# --------------------------------------------------------------------------- #
# Prompt rendering — same vocabulary as agent_runner_native for comparability. #
# --------------------------------------------------------------------------- #


from burl.harness.agent_runner_native import render_native_messages  # noqa: E402


_SYSTEM_SUFFIX = """

# How you call tools in this environment

You have access to the following tools via the `burl` MCP server. The tool
names you will see are `mcp__burl__<name>`. Call them directly; each call
returns a JSON payload as the tool result:

- `is_legal(domino_id)` — is this play legal?
- `is_trump(domino_id)` — is this a trump?
- `unseen()` — list of dominoes that are neither in your hand nor played.
- `void_audit(player_seat, suit)` — has this seat been proven void in this suit?
  (player_seat: 0=me, 1=left opp, 2=partner, 3=right opp. suit: 0..6 pip, 7=called.)
- `trump_declared()` — name of the current declaration.
- `eq_outcome_distribution(play, n_samples=10)` — outcome distribution for playing `play`.
- `conditional_outcome(play, assumption, n_samples=10)` — same, restricted to worlds
  where `assumption` holds. Use shape `{"player": abs_seat, "holds": dom_id}`
  or `{"player": abs_seat, "void_in_suit": suit}`.
- `commit_play(domino_id)` — commit your final choice. ENDS the decision.

Do NOT read any files, run any shell commands, or use any non-`mcp__burl__*`
tool. After committing with `commit_play`, stop — a brief one-line confirmation
is fine, but no further tool calls are needed."""


# --------------------------------------------------------------------------- #
# Runner.                                                                      #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class HaikuRunResult:
    """Outcome of one `run_decision_haiku` call."""
    final_play: int
    events: list[dict[str, Any]]
    cost_usd: float
    num_turns: int
    duration_ms: int
    usage: dict[str, Any]
    is_error: bool


DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_DISALLOWED_BUILTINS = [
    "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task", "WebFetch",
    "WebSearch", "NotebookEdit", "TodoWrite", "SlashCommand", "MultiEdit",
    "KillShell", "BashOutput",
]


async def run_decision_haiku(
    game_state: Any,
    *,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
    max_budget_usd: float = 0.10,
    decision_idx: int = 0,
    thinking_tokens: int | None = None,
) -> HaikuRunResult:
    """Run one Burl decision via a Claude model through the Agent SDK.

    ``model`` can be any model string the CLI accepts; default is Haiku 4.5.
    ``thinking_tokens`` enables Anthropic extended thinking at that budget;
    None disables thinking (matches the original Haiku spike behavior).

    The MCP server is rebuilt per call so there's no lingering state. Events
    are captured in the order the SDK emits them.
    """
    _CTX.bind(game_state)

    from burl.harness.agent_runner import _current_player, _visible_history

    me_abs = _current_player(game_state)
    hand_remaining = [d for d in game_state.hands[me_abs] if d not in game_state.played]
    history = _visible_history(game_state)
    system_content, user_content = render_native_messages(
        game_state, hand_remaining, history,
    )
    system_content = system_content + _SYSTEM_SUFFIX

    mcp = build_mcp_server()

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
    events.append({
        "event": "prompt", "role": "user",
        "content": user_content, "decision_idx": decision_idx,
    })

    result_payload: dict[str, Any] = {
        "cost_usd": 0.0, "num_turns": 0, "duration_ms": 0,
        "usage": {}, "is_error": False,
    }

    async def _prompt_stream():
        # Streaming mode is required for SDK MCP servers to round-trip tool
        # calls back into this process. `query()` with a plain string falls
        # back to one-shot --print mode, which kills the MCP RPC channel.
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
                        "event": "tool_call",
                        "tool": blk.name,
                        "args": blk.input,
                        "tool_use_id": blk.id,
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

    return HaikuRunResult(
        final_play=int(_CTX.final_play) if _CTX.final_play is not None else -1,
        events=events,
        cost_usd=result_payload["cost_usd"],
        num_turns=result_payload["num_turns"],
        duration_ms=result_payload["duration_ms"],
        usage=result_payload["usage"],
        is_error=result_payload["is_error"],
    )

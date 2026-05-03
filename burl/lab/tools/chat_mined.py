"""ToolSpecs mined from the burl-chat improvised-tool library.

The implementations stay in ``burl.chat.server.tools_library`` for now. This
module makes them first-class ``ToolSpec`` values so burl-lab can render their
protocol phrases, advertise them, and journal their evidence the same way it
does for base tools.
"""

from __future__ import annotations

from typing import Any

from burl.lab.core.tool import ToolResult, ToolSpec


def _payload_to_result(payload: dict) -> ToolResult:
    return ToolResult(
        evidence={
            "prose": str(payload.get("prose", "")),
            "structured": payload.get("structured", {}),
        },
        next_tools=(),
    )


def _state_brief_impl(ctx: Any, args: dict) -> ToolResult:
    from burl.chat.server.tools_library.state_brief import tool

    return _payload_to_result(tool(ctx, **args))


def _legal_plays_impl(ctx: Any, args: dict) -> ToolResult:
    from burl.chat.server.tools_library.legal_plays import tool

    return _payload_to_result(tool(ctx, **args))


def _board_snapshot_impl(ctx: Any, args: dict) -> ToolResult:
    from burl.chat.server.tools_library.board_snapshot import tool

    return _payload_to_result(tool(ctx, **args))


def _play_brief_impl(ctx: Any, args: dict) -> ToolResult:
    from burl.chat.server.tools_library.play_brief import tool

    return _payload_to_result(tool(ctx, play=int(args["play"])))


_NO_ARGS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


STATE_BRIEF = ToolSpec(
    name="state_brief",
    description=(
        "State synthesis in Burl's preferred labeled format: [GAME STATE], "
        "[CONTEXT & GOAL], [PROTOCOL]. Pure rule-based first read."
    ),
    params=_NO_ARGS_SCHEMA,
    example="state_brief()",
    protocol_role="first_read",
    protocol_phrase=(
        "Call `state_brief()` first when you need the current decision in "
        "labeled sections before deeper belief or outcome tools."
    ),
    impl=_state_brief_impl,
)


BOARD_SNAPSHOT = ToolSpec(
    name="board_snapshot",
    description=(
        "One-read state synthesis: hand, trump hierarchy, current trick, bid "
        "math, loose count, and trick history."
    ),
    params=_NO_ARGS_SCHEMA,
    example="board_snapshot()",
    protocol_role="first_read",
    protocol_phrase=(
        "Call `board_snapshot()` when you want a fuller board view before "
        "choosing which candidate plays to evaluate."
    ),
    impl=_board_snapshot_impl,
)


LEGAL_PLAYS = ToolSpec(
    name="legal_plays",
    description=(
        "Names the led suit and lists legal versus illegal dominoes under "
        "follow-suit. Says what is legal, not what to play."
    ),
    params=_NO_ARGS_SCHEMA,
    example="legal_plays()",
    protocol_role="diagnostic",
    protocol_phrase=(
        "Call `legal_plays()` before committing when you are following suit, "
        "after an illegal-play rejection, or whenever legality is uncertain."
    ),
    impl=_legal_plays_impl,
)


PLAY_BRIEF = ToolSpec(
    name="play_brief",
    description=(
        "Outcome distribution for one candidate play, rendered with a headline, "
        "mode labels, catalysts, and risk profile. Reuses explore_game cache."
    ),
    params={
        "type": "object",
        "properties": {
            "play": {
                "type": "integer",
                "description": "domino_id of the candidate play to summarize",
            },
        },
        "required": ["play"],
        "additionalProperties": False,
    },
    example="play_brief(play=14)",
    protocol_role="candidate_eval",
    protocol_phrase=(
        "Call `play_brief(play=X)` to read a labeled outcome summary for a "
        "candidate play; compare candidates by calling it once per play."
    ),
    impl=_play_brief_impl,
)


CHAT_MINED_TOOLS = (
    STATE_BRIEF,
    BOARD_SNAPSHOT,
    LEGAL_PLAYS,
    PLAY_BRIEF,
)


__all__ = [
    "BOARD_SNAPSHOT",
    "CHAT_MINED_TOOLS",
    "LEGAL_PLAYS",
    "PLAY_BRIEF",
    "STATE_BRIEF",
]

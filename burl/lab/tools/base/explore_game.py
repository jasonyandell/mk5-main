"""``explore_game`` ToolSpec.

Samples the outcome distribution for a candidate play: mean Q, p_make,
distribution shape, mode chart with IF/THEN catalysts. Called once per
candidate play the model wants to evaluate.
"""

from __future__ import annotations

from typing import Any

from burl.lab.core.tool import ToolResult, ToolSpec


def _impl(ctx: Any, args: dict) -> ToolResult:
    """Wrap the wax_museum tool_explore_game adapter into a ToolResult."""
    from burl.wax_museum.tools import tool_explore_game

    play = int(args["play"])
    payload = tool_explore_game(ctx, play=play)
    return ToolResult(
        evidence={
            "prose": payload["prose"],
            "structured": payload.get("structured", {}),
        },
        next_tools=(),
    )


EXPLORE_GAME = ToolSpec(
    name="explore_game",
    description=(
        "Sample the outcome distribution for a candidate play. Returns mean Q, "
        "p_make, distribution shape, and IF/THEN catalysts naming the dominoes "
        "whose location decides the outcome."
    ),
    params={
        "type": "object",
        "properties": {
            "play": {
                "type": "integer",
                "description": "domino_id of the candidate play to evaluate",
            },
        },
        "required": ["play"],
        "additionalProperties": False,
    },
    example="explore_game(play=14)",
    protocol_role="candidate_eval",
    protocol_phrase=(
        "Call `explore_game(play=X)` for each candidate play to sample its "
        "outcome distribution."
    ),
    impl=_impl,
)

"""``belief_trajectory`` ToolSpec.

Reads Gus's calibrated belief head for the current decision: per-domino
posterior P(L,P,R) for each opponent seat, plus shift-since-last and CLS
attention. Designed to be called once per turn before the model considers
candidate plays.
"""

from __future__ import annotations

from typing import Any

from burl.lab.core.tool import ToolResult, ToolSpec


def _impl(ctx: Any, args: dict) -> ToolResult:
    """Wrap the wax_museum belief_trajectory adapter into a ToolResult."""
    from burl.wax_museum.tools import tool_belief_trajectory

    payload = tool_belief_trajectory(ctx)
    return ToolResult(
        evidence={
            "prose": payload["prose"],
            "structured": payload.get("structured", {}),
        },
        next_tools=(),
    )


BELIEF_TRAJECTORY = ToolSpec(
    name="belief_trajectory",
    description=(
        "Per-domino posterior P(L,P,R) for each opponent seat, plus "
        "shift-since-last and CLS attention. Calibrated belief head from Gus."
    ),
    params={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    example="belief_trajectory()",
    protocol_role="first_read",
    protocol_phrase=(
        "Call `belief_trajectory()` once per turn to read the belief state "
        "before considering candidate plays."
    ),
    impl=_impl,
    requires_context=True,
)

"""``commit_play`` ToolSpec.

Terminal tool: records the model's chosen domino and advances the harness
to the ``post_turn`` phase. After commit, no further tool calls are
expected within the same decision.
"""

from __future__ import annotations

from typing import Any

from burl.lab.core.tool import ToolResult, ToolSpec


def _impl(ctx: Any, args: dict) -> ToolResult:
    """Record the committed domino on ctx and signal the post_turn phase."""
    domino_id = int(args["domino_id"])

    # Record on ctx if it accepts a final_play attribute. We don't validate
    # legality here — that's a phase concern; phases/in_run.py is responsible
    # for refusing illegal commits and re-prompting.
    if hasattr(ctx, "final_play"):
        ctx.final_play = domino_id

    prose = (
        f"COMMIT: domino_id={domino_id} recorded. Decision complete; "
        "no further tool calls needed."
    )
    structured = {"committed": domino_id}
    return ToolResult(
        evidence={"prose": prose, "structured": structured},
        next_tools=(),
        next_phase="post_turn",
    )


COMMIT_PLAY = ToolSpec(
    name="commit_play",
    description=(
        "Commit to the final play and end the decision. domino_id must be an "
        "integer from the current legal set; phase guards may refuse and "
        "re-prompt if not."
    ),
    params={
        "type": "object",
        "properties": {
            "domino_id": {
                "type": "integer",
                "description": "domino_id to play this turn",
            },
        },
        "required": ["domino_id"],
        "additionalProperties": False,
    },
    example="commit_play(domino_id=14)",
    protocol_role="commit",
    protocol_phrase=(
        "When you've decided, call `commit_play(domino_id=X)` once to play it. "
        "Do not call other tools after committing."
    ),
    impl=_impl,
    requires_context=True,
)

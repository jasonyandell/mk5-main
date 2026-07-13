"""Stub ``post_turn`` phase.

Renders the committed-session journal as segments; no engine re-entry; one
option: ``start_new_session`` (→ ``pre_game``).

The real ``post_turn`` (post-commit Q&A surface) is a separate workstream
the user will design — see the wiki for the eventual scope.  This stub is
just enough to make the post-commit phase reachable so the server doesn't
500 on the follow-up /api/move after a commit.
"""

from __future__ import annotations

from burl.lab.core.arrow import Trace
from burl.lab.core.tool import Registry
from burl.lab.core.transcript import (
    Frame,
    Move,
    Option,
    State,
    UserChoice,
)


class _PostTurnPhase:
    name = "post_turn"

    def render(self, state: State, registry: Registry | None = None) -> Frame:
        timing = {
            "wall_ms": 0,
            "tok_cum_in": state.cum_tok_in,
            "tok_cum_out": state.cum_tok_out,
            "tok_per_s": 0.0,
        }
        return Frame(
            phase=self.name,
            segments=list(state.segments),
            active_tools=list(state.active_tools),
            advertised=list(state.advertised),
            timing=timing,
        )

    def options(self, state: State, registry: Registry | None = None) -> list[Option]:
        return [
            Option(
                name="start_new_session",
                label="Start new session",
                args_schema={"type": "object", "properties": {}},
            ),
        ]

    async def handle(
        self,
        state: State,
        move: Move,
        registry: Registry | None = None,
    ) -> Trace[str]:
        if isinstance(move, UserChoice) and move.option_name == "start_new_session":
            return Trace(output="pre_game")
        return Trace()


POST_TURN = _PostTurnPhase()


__all__ = ["POST_TURN"]

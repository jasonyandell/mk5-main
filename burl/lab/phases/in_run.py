"""``in_run`` phase: model is generating; user can interject, abort, or
pull a HATEOAS-surfaced tool into the advertised set.

The render shows live segments + a timing ribbon.  Options let the user
interject text, abort, or select a tool that the previous ToolResult
surfaced via ``next_tools``.

Phase handlers return config Moves in a ``Trace``. The server journals those
Moves; there is no in-memory side state.
"""

from __future__ import annotations

from burl.lab.core.arrow import Trace
from burl.lab.core.tool import Registry
from burl.lab.core.transcript import (
    AdvertisedSet,
    EngineCommit,
    Frame,
    Move,
    Option,
    State,
    UserChoice,
    UserText,
    now_stamp,
)


class _InRunPhase:
    name = "in_run"

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
        opts: list[Option] = [
            Option(
                name="interject",
                label="Send a user message",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            Option(
                name="abort",
                label="Abort the run",
                args_schema={"type": "object", "properties": {}},
            ),
        ]
        # Surface tools that are ``active`` but not yet ``advertised`` —
        # those came from a tool result's ``next_tools`` and the user gets
        # to decide whether to bring them into scope (HATEOAS).
        surfaced = [n for n in state.active_tools if n not in state.advertised]
        for name in surfaced:
            opts.append(
                Option(
                    name="select_tool",
                    label=f"Advertise surfaced tool: {name}",
                    args_schema={
                        "type": "object",
                        "properties": {"name": {"const": name}},
                        "required": ["name"],
                    },
                )
            )
        return opts

    async def handle(
        self,
        state: State,
        move: Move,
        registry: Registry | None = None,
    ) -> Trace[str]:
        """Return journalable effects and optionally signal a phase transition.

        Observes the terminal ``EngineCommit`` Move that ``drive`` yields
        on a commit-role tool dispatch, and returns
        ``output="post_turn"``.  The server is responsible for
        journaling PhaseExit/PhaseEnter from that decision — drive is
        engine-shaped, the phase is transition-shaped.
        """
        new_moves: list[Move] = []
        next_phase: str | None = None

        if isinstance(move, EngineCommit):
            # Drive emitted (and journaled) EngineCommit. Terminal signal
            # for in_run; the server will journal PhaseExit/PhaseEnter
            # based on this returned next_phase.
            next_phase = "post_turn"

        elif isinstance(move, UserText):
            return Trace()

        elif isinstance(move, UserChoice):
            opt = move.option_name
            args = dict(move.args)

            if opt == "interject":
                text = str(args.get("text", ""))
                new_moves.append(UserText(stamp=now_stamp(state), text=text))

            elif opt == "abort":
                next_phase = "pre_game"

            elif opt == "select_tool":
                name = str(args.get("name", ""))
                if name and name not in state.advertised:
                    new_advertised = list(state.advertised) + [name]
                    new_moves.append(
                        AdvertisedSet(stamp=now_stamp(state), names=new_advertised)
                    )

        return Trace(events=tuple(new_moves), output=next_phase)


IN_RUN = _InRunPhase()


__all__ = ["IN_RUN"]

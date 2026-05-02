"""Phase Protocol + module-level PHASES registry.

A Phase is a pure state machine vertex. Each phase implements:

- `name: str`
- `render(state) -> Frame`           — view for the UI
- `options(state) -> list[Option]`   — what the user can pick next
- `async handle(state, move) -> (state', next_phase | None)`

Phases are HARNESS-PRIVATE — the model sees only the messages the phase
produces, never the phase identifier or transition logic.
"""

from __future__ import annotations

from typing import Awaitable, Protocol, runtime_checkable

from .transcript import Frame, Move, Option, State


@runtime_checkable
class Phase(Protocol):
    name: str

    def render(self, state: State) -> Frame: ...

    def options(self, state: State) -> list[Option]: ...

    async def handle(
        self, state: State, move: Move
    ) -> tuple[State, str | None]: ...


PHASES: dict[str, Phase] = {}


def register(p: Phase) -> None:
    """Register a Phase under its `name`. Last registration wins."""
    PHASES[p.name] = p


__all__ = ["Phase", "PHASES", "register"]

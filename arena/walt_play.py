"""Walt — the scenario player — as an arena PlayPolicy.

Walt = R_live[R_model[U]]: the live seat plans a contingent policy over a
sample of hidden-hand scenarios, modeling the other seats as the same
construction one rung down, bottoming out in random legal play. See
`scenario/planner.py` for the mathematics and `scenario/` for the parts.

Spec: walt[:k<level>][,n<outer worlds>][,m<inner worlds>][,h<outer horizon>]
          [,g<inner horizon>][,u=make|points]
Defaults: k=1 (the usual two-response form), n=16, m=4, h=full, g=4 plays.

The player is bid-aware (the contract threshold is the utility) and
score-blind. It runs on the CPU, one decision at a time; `nodes` after a
call is the cell count of the last decision, for timing work.
"""
from __future__ import annotations

from typing import Sequence

from forge.zeb.types import ZebGameState

from scenario.planner import Belief, Config, Rung, Walt


class WaltPlay:
    def __init__(self, level: int = 1, n_outer: int = 16, n_inner: int = 4,
                 horizon: int | None = None, horizon_inner: int | None = 4,
                 utility: str = "make", seed: int = 0, belief: Belief | None = None):
        rungs = tuple([Rung(n_inner, horizon_inner)] * level + [Rung(n_outer, horizon)])
        self.walt = Walt(Config(rungs, utility, seed), belief=belief)
        self.last_values: list[dict[int, float]] = []

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        slots = []
        self.last_values = []
        for state, bid in zip(states, bid_values):
            tile, q = self.walt.decide(state, int(bid))
            self.last_values.append(q)
            seat = (state.trick_leader + len(state.current_trick)) % 4
            slots.append(list(state.hands[seat]).index(tile))
        return slots

    def __repr__(self) -> str:
        c = self.walt.config
        return f"WaltPlay(level={c.level}, rungs={list(c.rungs)}, utility={c.utility!r})"


def parse_walt_spec(arg: str) -> WaltPlay:
    kw: dict = {}
    for t in (p for p in arg.split(",") if p):
        if t.startswith("u="):
            kw["utility"] = t[2:]
        elif t[0] in "knmhg" and t[1:].isdigit():
            key = {"k": "level", "n": "n_outer", "m": "n_inner",
                   "h": "horizon", "g": "horizon_inner"}[t[0]]
            kw[key] = int(t[1:])
        elif t in ("hfull", "gfull"):
            kw["horizon" if t[0] == "h" else "horizon_inner"] = None
        else:
            raise ValueError(f"unknown walt option {t!r}")
    return WaltPlay(**kw)

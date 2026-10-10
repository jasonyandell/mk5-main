"""Walt — the scenario player — as an arena PlayPolicy.

Walt = R_live[R_model[U]]: the live seat plans a contingent policy over a
sample of hidden-hand scenarios, modeling the other seats as the same
construction one rung down, bottoming out in random legal play. See
`scenario/planner.py` for the mathematics and `scenario/` for the parts.

Spec: walt[:k<level>][,n<outer worlds>][,m<inner worlds>][,h<outer horizon>]
          [,g<inner horizon>][,u=make|points][,b=<belief>][,e=<eps>][,py]
Defaults: k=1 (the usual two-response form), n=16, m=4, h=full, g=4 plays,
uniform belief.

Beliefs (`b=`, joined with `+`, each a likelihood factor on every scenario):
  dice     the hidden seats' discretionary plays under random legal play
           (a play with n options weighs 1/n — forced-looking plays count more)
  field    the same plays under the rung-0 player: eps/n + (1-eps)·[it would
           have played that tile]
  auction  the declaration: eps/7 + (1-eps)·[the bidder's full hand in this
           scenario would have called this trump] — exact for the bid30 and
           heuristic bidders, which declare `best_trump` deterministically

The player is bid-aware (the contract threshold is the utility) and
score-blind. By default it runs the compiled core (`scenario/core`), which
plans a whole arena round in parallel on every core; `py` selects the pure
Python planner, the referee, one decision at a time. `nodes` after a call
is the cell count of the last round, for timing work.
"""
from __future__ import annotations

from typing import Sequence

from forge.zeb.types import ZebGameState

from arena.hand_metrics import best_trump
from scenario.planner import Belief, Config, Rung, Situation, Tilt, Walt, World


def declaration_belief(eps: float = 0.1) -> Belief:
    """The auction as evidence: in each scenario, would the bidder's dealt
    hand have declared the trump that was declared? Deterministic under the
    arena's declaring rule (`best_trump` with no length floor), softened to
    eps/7 + (1-eps)·[match]; no evidence when the viewer is the bidder."""

    def belief(sit: Situation, worlds: list[World]) -> list[float]:
        if sit.bidder == sit.viewer:
            return [1.0] * len(worlds)
        shown = sit.played_by(sit.bidder)
        out = []
        for hands in worlds:
            full = hands[sit.bidder] | shown
            e = best_trump(tuple(t for t in range(28) if (full >> t) & 1), 0)
            assert e is not None
            out.append(eps / 7 + (1.0 - eps) * (1.0 if e.trump == sit.decl_id else 0.0))
        return out

    return belief


class WaltPlay:
    def __init__(self, level: int = 1, n_outer: int = 16, n_inner: int = 4,
                 horizon: int | None = None, horizon_inner: int | None = 4,
                 utility: str = "make", seed: int = 0, belief: Belief | None = None,
                 tilt: Tilt = Tilt(), compiled: bool = True):
        rungs = tuple([Rung(n_inner, horizon_inner)] * level + [Rung(n_outer, horizon)])
        config = Config(rungs, utility, seed, tilt)
        if compiled:
            from scenario.compiled import CompiledWalt
            self.walt = CompiledWalt(config, belief=belief)
        else:
            self.walt = Walt(config, belief=belief)
        self.last_values: list[dict[int, float]] = []

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        if hasattr(self.walt, "decide_batch"):
            decisions = self.walt.decide_batch(states, [int(b) for b in bid_values])
        else:
            decisions = [self.walt.decide(s, int(b)) for s, b in zip(states, bid_values)]
        self.last_values = [q for _, q in decisions]
        slots = []
        for state, (tile, _) in zip(states, decisions):
            seat = (state.trick_leader + len(state.current_trick)) % 4
            slots.append(list(state.hands[seat]).index(tile))
        return slots

    def __repr__(self) -> str:
        c = self.walt.config
        engine = type(self.walt).__name__
        belief = "" if self.walt.belief is None else ", +external belief"
        return (f"WaltPlay(level={c.level}, rungs={list(c.rungs)}, utility={c.utility!r}, "
                f"tilt={c.tilt.plays}{belief}, {engine})")


def parse_walt_spec(arg: str) -> WaltPlay:
    kw: dict = {}
    beliefs: set[str] = set()
    eps = 0.1
    for t in (p for p in arg.split(",") if p):
        if t.startswith("u="):
            kw["utility"] = t[2:]
        elif t.startswith("b="):
            beliefs |= set(t[2:].split("+"))
        elif t.startswith("e="):
            eps = float(t[2:])
        elif t[0] in "knmhg" and t[1:].isdigit():
            key = {"k": "level", "n": "n_outer", "m": "n_inner",
                   "h": "horizon", "g": "horizon_inner"}[t[0]]
            kw[key] = int(t[1:])
        elif t in ("hfull", "gfull"):
            kw["horizon" if t[0] == "h" else "horizon_inner"] = None
        elif t == "py":
            kw["compiled"] = False
        else:
            raise ValueError(f"unknown walt option {t!r}")
    unknown = beliefs - {"dice", "field", "auction"}
    if unknown:
        raise ValueError(f"unknown walt belief {sorted(unknown)}")
    if "dice" in beliefs and "field" in beliefs:
        raise ValueError("dice and field are two scorings of the same plays; pick one")
    plays = "dice" if "dice" in beliefs else "field" if "field" in beliefs else "none"
    kw["tilt"] = Tilt(plays, eps)
    if "auction" in beliefs:
        kw["belief"] = declaration_belief(eps)
    return WaltPlay(**kw)

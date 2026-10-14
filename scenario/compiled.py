"""The compiled scenario player: `scenario_core` (Rust, `scenario/core`)
behind the planner's interface.

`CompiledWalt` decides and analyzes exactly as `scenario.planner.Walt` does
— same tile, same action values, same per-scenario outcomes, bit for bit;
`scenario/tests/test_compiled.py` holds it to that — and adds the batch
forms, which plan a whole arena round of decisions in parallel on all cores
with the GIL released. `mold` is `scenario.atlas.mold`, compiled.

The belief seam is unchanged: the config's draw runs inside the core, and
`belief(situation, worlds) -> weights` is called once per decision, in
Python, on the live rung's scenarios (the core samples them, hands them
over, and multiplies the weights returned into the posterior's).

Build: `python -m pip install ./scenario/core` (needs the Rust toolchain).
"""
from __future__ import annotations

import math
from typing import Sequence

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState

from scenario.field import CANON
from scenario.planner import Analysis, Belief, Config, Situation, Step, situation_from_state
from scenario.record import remaining_hand
from scenario.rules import rules_for, tiles_of

try:
    import scenario_core
except ImportError as e:  # pragma: no cover — a build step, not a code path
    raise ImportError(
        "scenario_core is not built; run `python -m pip install ./scenario/core`"
    ) from e

MAX_WORLDS: int = scenario_core.MAX_WORLDS


class CompiledWalt:
    """A particular Walt — a Config plus a belief — run by the compiled core."""

    def __init__(self, config: Config, belief: Belief | None = None, threads: int = 0):
        self.config = config
        self.belief = belief
        self.threads = threads
        self._core = scenario_core.Core(
            [(r.n_worlds, r.horizon, r.floor) for r in config.rungs], config.utility, config.seed,
            None if config.field is None else config.field.to_bytes(), config.field_eps,
            config.draw, config.draw_budget,
            None if config.convention is None else
            (config.convention.name, config.convention.send, config.convention.read),
            None if config.reading is None else
            (config.reading.likelihood, config.reading.margin, config.reading.eps),
            config.tie,
        )
        self.nodes = 0      # cells evaluated in the last decision or batch
        self.proposals = 0  # posterior or reader: uniform proposals drawn in the last decision or batch
        self.fallbacks = 0  # posterior or reader: decisions of the last batch planned over the uniform
        self.sent = 0       # convention: decisions of the last batch whose tile the sender's rule chose
        self.draws: list[tuple[int, int]] = []   # per state of the last batch: (proposals, fell back); (0, 0) when forced

    def decide(self, state: ZebGameState, bid: int) -> tuple[int, dict[int, float]]:
        return self.decide_batch([state], [bid])[0]

    def decide_batch(
        self, states: Sequence[ZebGameState], bids: Sequence[int],
    ) -> list[tuple[int, dict[int, float]]]:
        """One decision per state, planned in parallel."""
        return [
            (tile, {tile: math.nan}) if res is None else (res[0], dict(res[1]))
            for tile, res in self._planned("solve", states, bids)
        ]

    def analyze(self, state: ZebGameState, bid: int, line: bool = False) -> Analysis:
        return self.analyze_batch([state], [bid], line)[0]

    def analyze_batch(self, states: Sequence[ZebGameState], bids: Sequence[int],
                      line: bool = False) -> list[Analysis]:
        """One `Analysis` per state, planned in parallel; a forced move holds
        no scenarios, as in `Walt.analyze`. With `line`, each carries the
        plan's line (`Step`s), as `Walt.analyze(line=True)` does."""
        return [
            Analysis(tile, {tile: math.nan}, [], [], {tile: ()}, {tile: ()} if line else None)
            if res is None else Analysis(
                res[0], dict(res[1]), list(res[3]), res[4], {t: tuple(o) for t, o in res[5]},
                None if res[6] is None else {
                    t: tuple(Step(tuple(plays), move, tuple(cell), floored) for plays, move, cell, floored in steps)
                    for t, steps in res[6]
                },
            )
            for tile, res in self._planned("analyze", states, bids, line=line)
        ]

    def mold(self, hands: Sequence[int], decl_id: int, leader: int,
             max_layer: int | None = None) -> list[int]:
        """`scenario.atlas.mold`, on all cores."""
        return self._core.mold(decl_id, tuple(hands), leader, max_layer, self.threads)

    def _planned(self, method: str, states: Sequence[ZebGameState], bids: Sequence[int],
                 **options) -> list[tuple]:
        """(tile, the core's result) per state, with the result None where the
        move is forced and the core is not asked; `method` is the core's
        `solve` or `analyze`, called with `options`."""
        out: list[tuple | None] = [None] * len(states)
        reqs: list[tuple] = []
        sits: list[tuple[int, Situation]] = []
        for i, (state, bid) in enumerate(zip(states, bids)):
            sit = situation_from_state(state, bid)
            rec, viewer = sit.record, sit.viewer
            if viewer != current_player(state):
                raise AssertionError("record seat disagrees with engine")
            own = remaining_hand(state, viewer)
            legal = tiles_of(rules_for(state.decl_id).legal(own, rec.led))
            if len(legal) == 1:
                out[i] = (legal[0], None)
                continue
            if self.config.field is not None and state.decl_id not in CANON:
                raise ValueError(f"the field net plays pip-trump declarations only, not {state.decl_id}")
            reqs.append((
                state.decl_id, viewer, own, rec.played, rec.leader, list(rec.trick),
                rec.banked, rec.forbid, sit.bid, sit.bidder, list(sit.plays),
            ))
            sits.append((i, sit))
        self.nodes = self.proposals = self.fallbacks = self.sent = 0
        self.draws = [(0, 0)] * len(states)
        if reqs:
            weights: list[list[float] | None] = [None] * len(reqs)
            if self.belief is not None:
                for j, worlds in enumerate(self._core.sample_roots(reqs)):
                    weights[j] = [float(w) for w in self.belief(sits[j][1], worlds)]
            for (i, _), res in zip(sits, getattr(self._core, method)(reqs, weights, self.threads, **options)):
                out[i] = (res[0], res)
                self.nodes += res[2]
                self.proposals += res[-3]
                self.fallbacks += res[-2]
                self.sent += res[-1]
                self.draws[i] = (res[-3], int(res[-2]))
        assert all(o is not None for o in out)
        return out  # type: ignore[return-value]

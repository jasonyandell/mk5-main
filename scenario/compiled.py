"""The compiled scenario player: `scenario_core` (Rust, `scenario/core`)
behind the planner's interface.

`CompiledWalt` decides exactly as `scenario.planner.Walt` does — same tile,
same action values, bit for bit; `scenario/tests/test_compiled.py` holds it
to that — and adds `decide_batch`, which plans a whole arena round of
decisions in parallel on all cores with the GIL released.

The belief seam is unchanged: `belief(record, viewer, worlds) -> weights`
is called once per decision, in Python, on the live rung's scenarios (the
core samples them, hands them over, and plans with the weights returned).

Build: `python -m pip install ./scenario/core` (needs the Rust toolchain).
"""
from __future__ import annotations

import math
from typing import Sequence

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState

from scenario.planner import Belief, Config
from scenario.record import Record, record_from_state, remaining_hand
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
            [(r.n_worlds, r.horizon) for r in config.rungs], config.utility, config.seed,
        )
        self._decls: set[int] = set()
        self.nodes = 0  # cells evaluated in the last decide/decide_batch

    def _rules(self, decl_id: int) -> None:
        if decl_id not in self._decls:
            r = rules_for(decl_id)
            self._core.set_rules(
                decl_id, list(r.led_suit), list(r.follow_bits),
                [list(row) for row in r.rank], list(r.count),
            )
            self._decls.add(decl_id)

    def decide(self, state: ZebGameState, bid: int) -> tuple[int, dict[int, float]]:
        return self.decide_batch([state], [bid])[0]

    def decide_batch(
        self, states: Sequence[ZebGameState], bids: Sequence[int],
    ) -> list[tuple[int, dict[int, float]]]:
        """One decision per state, planned in parallel."""
        out: list[tuple[int, dict[int, float]] | None] = [None] * len(states)
        reqs: list[tuple] = []
        recs: list[tuple[int, Record, int]] = []
        for i, (state, bid) in enumerate(zip(states, bids)):
            rec = record_from_state(state)
            viewer = rec.seat
            if viewer != current_player(state):
                raise AssertionError("record seat disagrees with engine")
            own = remaining_hand(state, viewer)
            legal = tiles_of(rules_for(state.decl_id).legal(own, rec.led))
            if len(legal) == 1:
                out[i] = (legal[0], {legal[0]: math.nan})
                continue
            self._rules(state.decl_id)
            reqs.append((
                state.decl_id, viewer, own, rec.played, rec.leader, list(rec.trick),
                rec.banked, rec.forbid, int(bid), state.bidder % 2,
            ))
            recs.append((i, rec, viewer))
        self.nodes = 0
        if reqs:
            weights: list[list[float] | None] = [None] * len(reqs)
            if self.belief is not None:
                for j, worlds in enumerate(self._core.sample_roots(reqs)):
                    _, rec, viewer = recs[j]
                    weights[j] = [float(w) for w in self.belief(rec, viewer, worlds)]
            for (i, _, _), (tile, q, nodes) in zip(recs, self._core.solve(reqs, weights, self.threads)):
                out[i] = (tile, dict(q))
                self.nodes += nodes
        assert all(o is not None for o in out)
        return out  # type: ignore[return-value]

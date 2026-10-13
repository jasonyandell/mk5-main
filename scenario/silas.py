"""Silas at Zeb's decisions: the frame a seat's questions are asked of.

Silas (`silas/`, compiled into `scenario_core` by the `silas` feature) asks
typed relational questions of the worlds a seat cannot tell apart. Its
classes are `scenario_core`'s own — `SilasQuery`, `SilasPolicy`,
`SilasBelief`, `SilasSummary`, `SilasEstimate`, `SilasAuthority`,
`silas_compare`, `SilasError` — and run on Zeb's rules (`ZebPhysics`,
`scenario/core/src/physics.rs`). This module supplies the one thing Python
holds and Rust does not: the decision. `frame(sit, own)` turns Zeb's
`Situation` and the viewer's remaining hand into a `SilasFrame`.

The record keeps each seat's voids as the tiles it provably lacks
(`Record.forbid`); Silas keeps the suits it has shown void in. `shown_voids`
reads those suits off the plays, and `frame` checks that they forbid exactly
the record's tiles.
"""
from __future__ import annotations

from scenario_core import SilasFrame

from scenario.planner import Situation
from scenario.rules import rules_for


def shown_voids(sit: Situation) -> tuple[int, int, int, int]:
    """Per seat, the led suits it has failed to follow: bit k for Zeb's led
    suit k (7 = the called suit)."""
    rules = rules_for(sit.decl_id)
    voids = [0, 0, 0, 0]
    lead = None
    for i, (seat, tile) in enumerate(sit.plays):
        if i % 4 == 0:
            lead = tile
        elif not rules.follows(tile, lead):
            voids[seat] |= 1 << rules.led_suit[lead]
    return (voids[0], voids[1], voids[2], voids[3])


def frame(sit: Situation, own: int) -> SilasFrame:
    """The Silas frame of the seat to play at `sit`, holding `own` (its
    remaining hand, a tile mask)."""
    rules, rec = rules_for(sit.decl_id), sit.record
    voids = shown_voids(sit)
    for seat in range(4):
        shown = 0
        for k in range(8):
            if voids[seat] >> k & 1:
                shown |= rules.follow_bits[k]
        if shown != rec.forbid[seat]:
            raise AssertionError(f"seat {seat}'s shown voids disagree with the record")
    sizes = tuple(rec.hand_size(s) for s in range(4))
    return SilasFrame(sit.decl_id, sit.viewer, own, rec.played, rec.leader, list(rec.trick), voids, sizes)

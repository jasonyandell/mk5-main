# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Silas at Zeb's decisions: the frame a seat's questions are asked of.

Silas (`silas/`, compiled into `scenario_core` by the `silas` feature) asks
typed relational questions of the worlds a seat cannot tell apart. Its
classes are `scenario_core`'s own — `SilasQuery`, `SilasPolicy`,
`SilasBelief`, `SilasSummary`, `SilasEstimate`, `SilasAuthority`,
`silas_compare`, `SilasError` — and run on Zeb's rules (`ZebPhysics`,
`scenario/core/src/physics.rs`). This module supplies the one thing Python
holds and Rust does not: the decision. `frame(sit, own)` turns Zeb's
`Situation` and the viewer's remaining hand into a `SilasFrame`.

A `SilasFrame` is built from the hand's public record alone: the contract
(the bidder and the bid) and every play, attributed to its seat. Silas reads
the rest off the plays, and `frame` checks that reading against Zeb's own
`Record`: the played set, the leader, the open trick, the banked points, each
seat's remaining size, and each seat's voids (the record keeps them as the
tiles a seat provably lacks, `Record.forbid`; Silas keeps the suits it has
shown void in).
"""
from __future__ import annotations

from scenario_core import SilasFrame

from scenario.planner import Situation
from scenario.rules import rules_for


def frame(sit: Situation, own: int) -> SilasFrame:
    """The Silas frame of the seat to play at `sit`, holding `own` (its
    remaining hand, a tile mask)."""
    rules, rec = rules_for(sit.decl_id), sit.record
    f = SilasFrame(sit.decl_id, sit.viewer, own, sit.bidder, sit.bid, list(sit.plays))
    forbid = [0, 0, 0, 0]
    for seat in range(4):
        for k in range(8):
            if f.voids[seat] >> k & 1:
                forbid[seat] |= rules.follow_bits[k]
    zeb = (rec.played, rec.leader, rec.trick, rec.banked, rec.forbid, tuple(rec.hand_size(s) for s in range(4)))
    silas = (f.played, f.leader, tuple(f.trick), f.banked, tuple(forbid), tuple(f.sizes))
    if zeb != silas:
        raise AssertionError(f"Silas reads the record as {silas}, Zeb as {zeb}")
    return f

# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Constellations over Zeb's late hand: what a lookup keyed on them may hold.

A constellation (`scenario/core/src/constellation.rs`) is a position with
the names rubbed out: the live tiles and their standings among the living,
who holds them (or, for a seat, which seats the record has shown void of
them), counts, the open trick, and what the contract still needs. Two
positions with one constellation are the same remaining game under a tile
bijection (texas-42's Lemma E). That is true whatever came before: the
remaining game never reads a dead tile.

So any value that is a fold over the remaining game, by an operator that
itself reads only constellations, is a function of the constellation. Zeb's
plan is such a fold (one action per cell of the viewer's worlds, values
added over what the viewer can distinguish, the other seats played by the
field) *when its field is*. The deployed field net is not: it reads pips and
the dead tiles (`scenario.field`), so a plan against it can differ between
two positions with one constellation. `substitutions` builds such pairs and
`Ladder` builds the field that does read only constellations.

The ladder is the scenario player's own ladder in the late hand, exact: a
level-n seat plans, as one-rung Zeb over its whole fiber (enumerated, so no
floor and no sample), against level n-1 in the other three seats. Level 0 is
a base field. With a structural base (`first`: the canonically first legal
tile) every level is a function of the constellation, and its decisions are
stored in a table keyed by it, choice ties broken by canonical index. With
the net as base (`net`) the levels are keyed by the exact information state
instead. A seat's decision at every level is an information-state choice:
nothing here plays one move per world.
"""
from __future__ import annotations

from dataclasses import dataclass

import scenario_core

from scenario.fiber import Fiber
from scenario.field import FieldNet
from scenario.planner import Config, Rung, Walt
from scenario.record import Record
from scenario.rules import Rules, tiles_of

ENUMERATE = 1 << 12     # worlds a ladder plan may hold: every fiber with three tricks left or fewer


def form(rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int,
         hands: tuple[int, int, int, int] | None = None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(the canonical form, the tiles in canonical order): the seat form of
    `seat` holding `own`, or with `hands` the world form."""
    words, order, _ = scenario_core.constellation(
        rules.decl_id, seat, own, rec.played, rec.leader, list(rec.trick), rec.banked,
        rec.forbid, bid, bidder_team, True, hands)
    return tuple(words), tuple(order)


def automorphisms(rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int) -> int:
    """How many relabelings carry the seat's constellation to itself (1: none but the identity)."""
    return scenario_core.constellation(
        rules.decl_id, seat, own, rec.played, rec.leader, list(rec.trick), rec.banked,
        rec.forbid, bid, bidder_team, True, None)[2]


def _swap(mask: int, a: int, b: int) -> int:
    """`mask` with tiles a and b exchanged."""
    ia, ib = (mask >> a) & 1, (mask >> b) & 1
    if ia != ib:
        mask ^= (1 << a) | (1 << b)
    return mask


def substitute(rec: Record, hands: tuple[int, int, int, int], live: int, dead: int
               ) -> tuple[Record, tuple[int, int, int, int]]:
    """The position with a live tile and a dead one exchanged: the dead tile
    is now held where the live one was, the live one has been played where
    the dead one was, and every void the record shows moves with them."""
    rec2 = Record(_swap(rec.played, live, dead), rec.leader, rec.trick, rec.banked,
                  tuple(_swap(f, live, dead) for f in rec.forbid))
    return rec2, tuple(_swap(h, live, dead) for h in hands)


def substitutions(rules: Rules, rec: Record, hands: tuple[int, int, int, int], bid: int,
                  bidder_team: int) -> list[tuple[int, int, Record, tuple[int, int, int, int]]]:
    """Every (live, dead, record, hands) exchange that leaves the seat to
    play's constellation unchanged: the same remaining game, a different
    history."""
    seat = rec.seat
    here = form(rules, seat, hands[seat], rec, bid, bidder_team)[0]
    table = 0
    for t in rec.trick:
        table |= 1 << t
    out = []
    live_tiles = tiles_of(((1 << 28) - 1) & ~rec.played)
    for d in live_tiles:
        for e in tiles_of(rec.played & ~table):
            rec2, hands2 = substitute(rec, hands, d, e)
            if form(rules, seat, hands2[seat], rec2, bid, bidder_team)[0] == here:
                out.append((d, e, rec2, hands2))
    return out


class _First:
    """The structural base field: the canonically first legal tile."""

    def move(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int) -> int:
        legal = rules.legal(own, rec.led)
        order = form(rules, seat, own, rec, bid, bidder_team)[1]
        return next(t for t in order if (legal >> t) & 1)


@dataclass
class Entry:
    """One stored decision: the tile chosen, as a canonical index when the
    level is keyed by constellation, and the plan's value per legal move."""

    choice: int
    q: dict[int, float]


class Level:
    """One rung of the late-hand ladder, as a field for the rung above: a
    seat at this level plans as one-rung Zeb over its enumerated fiber
    against `below`. Decisions are stored in `table`, keyed by the seat's
    constellation (`structural`) or by its exact information state."""

    def __init__(self, below, structural: bool, utility: str = "make"):
        self.below = below
        self.structural = structural
        self.walt = Walt(Config((Rung(ENUMERATE, None, 1),), utility, field=below))
        self.table: dict[tuple, Entry] = {}
        self.planned = 0     # decisions planned (table misses)
        self.looked_up = 0   # decisions answered from the table
        self.orbit_ties = 0  # planned decisions whose best moves tie inside a symmetric constellation
        self.exact: set = set()   # exact information states asked (structural levels)
        self.symmetric_hits = 0   # table answers a plain cache on the exact state would have missed

    def key(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int):
        if self.structural:
            return form(rules, seat, own, rec, bid, bidder_team)
        return (rules.decl_id, seat, own, rec.key(), bid, bidder_team), None

    def plan(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int
             ) -> dict[int, float]:
        """The plan's value per legal tile at this information state (no table)."""
        sizes = tuple(rec.hand_size(s) for s in range(4))
        if Fiber(seat, own, rec.played, sizes, rec.forbid).size > ENUMERATE:
            raise ValueError("the ladder plans enumerated fibers only (three tricks left or fewer)")
        return self.walt._invocation(0, seat, own, rec, rules, bid, bidder_team).action_values()

    def decide(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int
               ) -> tuple[int, dict[int, float]]:
        """(tile, q by tile): from the table when this state's key is in it."""
        key, order = self.key(rules, seat, own, rec, bid, bidder_team)
        entry = self.table.get(key)
        if entry is None:
            legal = tiles_of(rules.legal(own, rec.led))
            if len(legal) == 1:
                q = {legal[0]: float("nan")}
            else:
                q = self.plan(rules, seat, own, rec, bid, bidder_team)
            maximize = seat % 2 == bidder_team
            rank = (lambda t: order.index(t)) if order is not None else (lambda t: t)
            best = max(q, key=lambda t: ((q[t] if maximize else -q[t]) if len(q) > 1 else 0.0, -rank(t)))
            stored = {rank(t): v for t, v in q.items()}
            if (self.structural and sum(v == q[best] for v in q.values()) > 1
                    and automorphisms(rules, seat, own, rec, bid, bidder_team) > 1):
                self.orbit_ties += 1
            entry = Entry(rank(best), stored)
            self.table[key] = entry
            self.planned += 1
        else:
            self.looked_up += 1
            if self.structural and (rules.decl_id, seat, own, rec.key(), bid, bidder_team) not in self.exact:
                self.symmetric_hits += 1
        if self.structural:
            self.exact.add((rules.decl_id, seat, own, rec.key(), bid, bidder_team))
        tile_of = (lambda i: order[i]) if order is not None else (lambda i: i)
        return tile_of(entry.choice), {tile_of(i): v for i, v in entry.q.items()}

    def move(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int) -> int:
        return self.decide(rules, seat, own, rec, bid, bidder_team)[0]


class Ladder:
    """Levels 0..n over a base field: `base="first"` (structural, every
    level keyed by constellation) or a `FieldNet` (every level keyed by the
    exact information state, since the net reads what a constellation drops)."""

    def __init__(self, base: str | FieldNet = "first", utility: str = "make"):
        self.structural = base == "first"
        self.utility = utility
        self.levels: list = [_First() if self.structural else base]

    def level(self, n: int):
        """The field at level n (level 0 is the base)."""
        while len(self.levels) <= n:
            self.levels.append(Level(self.levels[-1], self.structural, self.utility))
        return self.levels[n]

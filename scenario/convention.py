# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Conventions: a joint deviation of two partners, a sender and a reader.

A convention is a rule a seat may follow at a gated position: a predicate
on the mover's chair alone — its seat's role (the bidder, or the bidder's
partner), its own remaining hand, and the public record — that, where it
holds, narrows the mover's legal tiles to an allowed set. Both halves read
the same rule:

- **Send.** At the live decision the plan runs as usual (its q is reported
  unchanged), and where the gate holds the seat plays the plan's best tile
  among the allowed ones (the tie rule within the set) instead of the
  plan's best overall. `Walt.sent` says the override was in force.
- **Read.** The live seat draws its worlds from the rule-only posterior: a
  world is kept iff every gated decision its *partner* has made so far,
  replayed with the hand that world gives the partner, played an allowed
  tile (likelihood 1 or 0; ungated decisions and the opponents' are not
  filtered). It is the posterior draw with this likelihood in place of the
  field's (`Walt.convention_likelihood`); it composes with the uniform draw
  only — the field's likelihood would model the same partner moves a second
  time, by a different model.

A gate that leaves every legal tile allowed is no gate (`None`): the rule
fires only where it constrains, so a forced move never fires it. The
modeled seats inside the plan do not follow the convention: the overlay is
the live seat's alone, and the field it plans against is unchanged.

Trumps are the tiles with trump power (`Rules.rank` tier 2, rank ≥ 32): a
pip suit's seven, the seven doubles under doubles-trump, none under
no-trump. Trumps are ordered by that rank. Count is `Rules.count`.

The conventions (Winning 42, `scratch/book/CANDIDATES.md`):

`handoff` — C2, the table's high-trump handoff. Gate: the mover is the
bidder's partner, the bidder led the open trick with a trump, and the
highest unplayed trump is in the mover's hand. Allowed: that trump alone
(even when a lower trump would win, even when the lead already wins).
Read: after the partner followed such a lead, the worlds where he held the
top unplayed trump and played anything else are deleted.

`handoff-singleton` — the same sender; the book's reading (page 25): a
partner who plays the top trump had no other. Read gate as above; allowed:
the top trump when it is the partner's only trump, else any trump but the
top. So a top trump from partner deletes every world where he held another.

`biggest` — C3, biggest count first. Gate: the mover is the bidder's
partner, the bidder led the open trick, the mover cannot follow, the lead
is winning the trick so far and is a trump or the highest unplayed tile of
its suit, and the mover holds two or more count tiles of different values.
Allowed: every tile but the count tiles smaller than the largest held — if
he gives count, it is his largest. Read: a donation of a smaller count
deletes the worlds where the partner held a larger one.

`donation` — C1, count only on a guaranteed trick. Two gates.
(a) The mover is the bidder, on lead; the partner is shown void in trump
(the record's voids); a trump is still unplayed outside the mover's hand
(so with the opponents); the highest unplayed trump is the mover's.
Allowed: that trump — he leads it. (b) The mover is the bidder's partner,
the bidder led a trump, and the mover holds no trump. If the lead was the
highest trump unplayed when it was led (every higher trump fell in an
earlier trick), allowed: the mover's largest count tiles (gate only when
he holds count); otherwise allowed: his non-count tiles (gate only when he
holds one). Read: the bidder deletes worlds where the partner sluffed
non-count on a top-trump lead while holding count, or count on a lower
lead while holding a non-count; the partner deletes worlds where the
bidder held the top trump at gate (a) and led anything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from scenario.record import Record
from scenario.rules import Rules, bit, mask_of, rules_for, tiles_of

TRUMP_RANK = 32  # trick rank tier 2: a tile with trump power

Gate = Callable[[Rules, int, int, int, Record], "int | None"]


def trumps(rules: Rules) -> int:
    """The tiles with trump power under this declaration."""
    return _trumps(rules.decl_id)


@lru_cache(maxsize=None)
def _trumps(decl_id: int) -> int:
    rules = rules_for(decl_id)
    return mask_of(t for t in range(28) if rules.rank[7][t] >= TRUMP_RANK)


def _top(rules: Rules, mask: int) -> int | None:
    """The highest trump in `mask` (a set of trumps), or None when empty."""
    return max(tiles_of(mask), key=lambda t: rules.rank[7][t], default=None)


def _counts(rules: Rules, hand: int) -> int:
    return mask_of(t for t in tiles_of(hand) if rules.count[t])


def _partner(seat: int) -> int:
    return (seat + 2) % 4


def _handoff_top(rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
    """The handoff's gate: the top unplayed trump, when the mover is the
    bidder's partner following the bidder's trump lead and holds it."""
    if seat != _partner(bidder) or not rec.trick or rec.leader != bidder:
        return None
    t = trumps(rules)
    if not (t >> rec.trick[0]) & 1:
        return None
    top = _top(rules, t & ~rec.played)
    return top if top is not None and (own >> top) & 1 else None


def _handoff(rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
    top = _handoff_top(rules, seat, bidder, own, rec)
    return None if top is None else bit(top)


def _singleton(rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
    top = _handoff_top(rules, seat, bidder, own, rec)
    if top is None:
        return None
    others = own & trumps(rules) & ~bit(top)
    return others if others else bit(top)


def _biggest(rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
    if seat != _partner(bidder) or not rec.trick or rec.leader != bidder:
        return None
    lead = rec.trick[0]
    suit = rules.led_suit[lead]
    if own & rules.follow_bits[suit]:
        return None
    rank = rules.rank[suit]
    if any(rank[t] > rank[lead] for t in rec.trick[1:]):
        return None                                        # the lead is not winning
    if not (trumps(rules) >> lead) & 1 and any(
            rank[t] > rank[lead] for t in tiles_of(rules.follow_bits[suit] & ~rec.played)):
        return None                                        # a higher tile of the suit is out
    counts = _counts(rules, own)
    if counts.bit_count() < 2:
        return None
    most = max(rules.count[t] for t in tiles_of(counts))
    return own & ~mask_of(t for t in tiles_of(counts) if rules.count[t] < most)


def _donation(rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
    t = trumps(rules)
    if not t:
        return None
    partner = _partner(bidder)
    if seat == bidder and not rec.trick:
        out = t & ~rec.played
        if t & ~rec.forbid[partner] or not out & ~own:
            return None
        top = _top(rules, out)
        return bit(top) if (own >> top) & 1 else None
    if seat != partner or not rec.trick or rec.leader != bidder:
        return None
    lead = rec.trick[0]
    if not (t >> lead) & 1 or own & t:
        return None
    earlier = rec.played & ~mask_of(rec.trick)
    top_lead = all(rules.rank[7][x] < rules.rank[7][lead] for x in tiles_of(t & ~earlier & ~bit(lead)))
    counts = _counts(rules, own)
    if top_lead:
        if not counts:
            return None
        most = max(rules.count[x] for x in tiles_of(counts))
        return mask_of(x for x in tiles_of(counts) if rules.count[x] == most)
    rest = own & ~counts
    return rest if rest else None


SEND: dict[str, Gate] = {
    "handoff": _handoff, "handoff-singleton": _handoff, "biggest": _biggest, "donation": _donation,
}
READ: dict[str, Gate] = {
    "handoff": _handoff, "handoff-singleton": _singleton, "biggest": _biggest, "donation": _donation,
}
NAMES = tuple(SEND)


def _gated(gate: Gate, rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
    allowed = gate(rules, seat, bidder, own, rec)
    if allowed is None or allowed == rules.legal(own, rec.led):
        return None
    assert allowed and allowed & ~rules.legal(own, rec.led) == 0, "a rule allows legal tiles only"
    return allowed


@dataclass(frozen=True)
class Convention:
    """A named convention and which halves the live seat runs: S = send
    only, R = read only, SR = both."""

    name: str
    send: bool = True
    read: bool = True

    def __post_init__(self) -> None:
        if self.name not in NAMES:
            raise ValueError(f"unknown convention {self.name!r}; known: {', '.join(NAMES)}")
        if not (self.send or self.read):
            raise ValueError("a convention sends, reads, or both")

    @property
    def arm(self) -> str:
        return ("S" if self.send else "") + ("R" if self.read else "")

    def sends(self, rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
        """The allowed tiles where the sender's gate holds for the mover, else None."""
        return _gated(SEND[self.name], rules, seat, bidder, own, rec)

    def reads(self, rules: Rules, seat: int, bidder: int, own: int, rec: Record) -> int | None:
        """The allowed tiles the reader expects of `seat` holding `own`, else None."""
        return _gated(READ[self.name], rules, seat, bidder, own, rec)

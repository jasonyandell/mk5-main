"""atlas/narrate.py — coordinate deltas as sentences.

Given a step ``before -> after`` (after == transition(before, tile)), emit
the events the step made true: the play, any observed void, a banked trick,
and the role promotions that the shrinking outstanding set forced on the
viewer's hand ("the 6-4 became a walker").  Pure functions over the two
coordinates; this is the family-vocabulary surface and the count-fate
referees' hook.

Roles only promote, so a promotion event is monotone and never retracted —
narration is a faithful log of the lattice walk, not a heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW

from atlas.roles import boss_mask, outstanding, walker_mask

_SUIT_NAMES = ("blanks", "ones", "twos", "threes", "fours", "fives", "sixes",
               "trumps")


def tile_name(tile: int) -> str:
    """Family notation for a domino: high pip, low pip (e.g. '6-4')."""
    return f"{DOMINO_HIGH[tile]}-{DOMINO_LOW[tile]}"


def suit_name(led_suit: int) -> str:
    """Name of a led-suit domain value (0..6 pip suits, 7 = the called suit)."""
    return _SUIT_NAMES[led_suit]


@dataclass(frozen=True)
class Event:
    """A single coordinate-delta fact. ``kind`` is one of: 'play',
    'void', 'banked', 'walker', 'boss'."""

    kind: str
    seat: int
    tile: int = -1
    suit: int = -1
    points: int = 0

    def sentence(self) -> str:
        if self.kind == "play":
            return f"seat {self.seat} played the {tile_name(self.tile)}"
        if self.kind == "void":
            return f"seat {self.seat} shown void in {suit_name(self.suit)}"
        if self.kind == "banked":
            return (f"seat {self.seat} won the trick and banked "
                    f"{self.points} point{'s' if self.points != 1 else ''}")
        if self.kind == "walker":
            return f"the {tile_name(self.tile)} became a walker"
        if self.kind == "boss":
            return f"the {tile_name(self.tile)} became a boss"
        raise ValueError(f"unknown event kind {self.kind!r}")


def _played_tile(before, after) -> tuple[int, int]:
    """(seat, tile) of the play that took ``before`` to ``after``."""
    seat = before.acting_seat
    added = int(after.played[seat]) & ~int(before.played[seat])
    if added == 0 or (added & (added - 1)):
        raise ValueError("before/after do not differ by exactly one play")
    return seat, added.bit_length() - 1


def _promotions(before, after, kind: str, mask_fn) -> list[Event]:
    decl = before.decl_id
    hb, ha = int(before.viewer_hand), int(after.viewer_hand)
    if kind == "walker":
        was = mask_fn(hb, outstanding(before, hb), decl)
        now = mask_fn(ha, outstanding(after, ha), decl)
    else:  # boss: bosses of the live set (out | hand), restricted to the hand
        was = mask_fn(outstanding(before, hb) | hb, decl) & hb
        now = mask_fn(outstanding(after, ha) | ha, decl) & ha
    gained = now & ~was
    return [Event(kind=kind, seat=after.viewer, tile=t)
            for t in range(28) if (gained >> t) & 1]


def narrate(before, after) -> list[Event]:
    """The events that ``before -> after`` made true, in reading order:
    the play, any void, a banked trick, then the viewer's role promotions."""
    seat, tile = _played_tile(before, after)
    events = [Event(kind="play", seat=seat, tile=tile)]

    for s in range(4):
        gained = int(after.voids[s]) & ~int(before.voids[s])
        for ls in range(8):
            if (gained >> ls) & 1:
                events.append(Event(kind="void", seat=s, suit=ls))

    banked = sum(after.team_points) - sum(before.team_points)
    if banked > 0 and not after.current_trick:
        events.append(Event(kind="banked", seat=after.trick_leader, points=banked))

    events += _promotions(before, after, "walker", walker_mask)
    events += _promotions(before, after, "boss", boss_mask)
    return events

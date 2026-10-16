# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Position cards: one position, in the viewer's words, for a reader.

A card holds exactly the viewer's information and nothing more: the trump,
the bid and who holds it, the trick number and the viewer's place in it,
the viewer's hand, the open trick by relative seat, the points banked, the
count already gone, and the voids each seat has shown. No hidden hand, no
value, no activation. Tiles are written high pip first, `6-4`.
"""
from __future__ import annotations

from forge.oracle.declarations import DECL_ID_TO_NAME
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW

from lem.probe.sample import Sample
from scenario.record import empty_record
from scenario.rules import rules_for, tiles_of

SUITS = ("blanks", "ones", "twos", "threes", "fours", "fives", "sixes")
REL = {0: "you", 1: "left", 2: "partner", 3: "right"}


def tile(t: int) -> str:
    return f"{DOMINO_HIGH[t]}-{DOMINO_LOW[t]}"


def card(sample: Sample, r: int) -> str:
    decl, viewer, own, bidder, bid, plays = sample.frame_args(r)
    rules = rules_for(decl)
    rec = empty_record(bidder)
    for _, t in plays:
        rec = rec.advance(t, rules)
    trump = DECL_ID_TO_NAME.get(decl, str(decl))
    me = viewer
    side = "your side" if me % 2 == bidder % 2 else "the other side"
    who = REL[(bidder - me) % 4]
    trick_no = rec.plays_so_far() // 4 + 1
    place = ("lead", "second", "third", "last")[len(rec.trick)]
    hand = " ".join(tile(t) for t in sorted(tiles_of(own), key=lambda t: -rules.rank[7][t]))
    lines = [
        f"Trump: {trump}. Bid {bid} by {who} ({side}).",
        f"Trick {trick_no} of 7, you play {place}. Your hand: {hand}.",
    ]
    if rec.trick:
        plays_txt = []
        for i, t in enumerate(rec.trick):
            seat = (rec.leader + i) % 4
            plays_txt.append(f"{REL[(seat - me) % 4]} {tile(t)}")
        w = (rec.leader + rules.winner_offset(tuple(rec.trick) + (rec.trick[0],) * (4 - len(rec.trick)))) % 4
        # the winner so far: recompute over the open trick only
        ranks = rules.rank[rules.led_suit[rec.trick[0]]]
        best = max(range(len(rec.trick)), key=lambda i: (ranks[rec.trick[i]], -i))
        w = (rec.leader + best) % 4
        lines.append(f"On the table: {', '.join(plays_txt)}; {REL[(w - me) % 4]} is winning.")
    else:
        lines.append(f"{REL[(rec.leader - me) % 4].capitalize()} leads." if rec.leader != me else "You lead.")
    us, them = rec.banked[me % 2], rec.banked[1 - me % 2]
    lines.append(f"Points so far: you {us}, them {them}.")
    gone = [tile(t) for t in tiles_of(rec.played) if rules.count[t]]
    lines.append("Count gone: " + (" ".join(gone) if gone else "none") + ".")
    voids = []
    for seat in range(4):
        if seat == me:
            continue
        shown = [SUITS[s] if s < 7 else "trump" for s in range(8)
                 if rec.forbid[seat] & rules.follow_bits[s] == rules.follow_bits[s] and rules.follow_bits[s]]
        shown = [s for s in shown if s != "trump" or decl < 7]
        if shown:
            voids.append(f"{REL[(seat - me) % 4]} out of {', '.join(dict.fromkeys(shown))}")
    lines.append("Shown voids: " + ("; ".join(voids) if voids else "none") + ".")
    return "\n".join(lines)

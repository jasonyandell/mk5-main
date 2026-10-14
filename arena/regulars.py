# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The regulars' table (#149): a reader that reads the auction, and remembers.

The book's opponents are regulars: they learn how you bid, hand by hand, and
read your next bid by it. This module is that reader's model of a bidder and
its memory, built on Zeb's soft play-reading (`Config.reading`): the auction
joins the reading's likelihood as a table over each opposing seat's dealt
hand (`scenario.planner.Situation.auction`).

**The bidder model.** An opposing seat bids by the arena's heuristic rule
(`arena.bidders.HeuristicBidder`): at its one turn, with `target` the lowest
legal point bid (at most 42), it bids the target when its hand has a pip
suit of three tiles and min(42, ceiling - c) >= target, else passes, where c
is its caution. With probability `tremble` it acts at random instead, so
no world vanishes:

    P(action | hand, c) = tremble/2 + (1 - tremble)·[action is the rule's]

The hand enters only through its auction slot (`ceiling_slot`), so for a
belief pi over the caution grid the evidence of one auction on one seat is
a table over slots, T[slot] = sum_c pi(c)·P(action | slot, c). A turn with
no point bid left to make, or a bid other than the target (the rule never
makes one), is no evidence.

**The memory.** `memory=False` is R0: pi is flat on `GRID` every hand.
`memory=True` is R: pi is a posterior over a *fixed* caution per seat, kept
for one game. After each hand (every hand is played out, so each seat's
dealt tiles are public by then) every seat's pi is multiplied by
P(its action | its revealed hand, c) and renormalized. The model assumes a
seat keeps one style; a seat that varies its caution does not fit it, and
that is the book's "throw 'em off" (S-0080-b): the regular who has learned
you is misled when you change.

What this reader does not do: it does not change its own bidding or play in
response (no exploitation), it does not learn a seat's play (the reading's
eps is fixed), it does not read the declaration or its partner's bid, and
its memory lasts one game.

The fence: every input is public when it is used — the auction of this hand,
and the tiles of hands already finished ([[strategy-fusion-fence]]).
"""
from __future__ import annotations

from typing import Sequence

from forge.zeb.types import ZebGameState

from arena.auction import MIN_BID, ONE_MARK, PASS, bidding_order
from arena.engine import HandRecord
from scenario.planner import AUCTION_SLOTS, Auction, ceiling_slot
from scenario.rules import mask_of

GRID = (-10, -5, 0, 5, 10)   # the cautions a regular considers; ceilings move in steps of 5
TREMBLE = 0.1                # the probability a bidder acts at random


def turns(bids: Sequence[int], dealer: int) -> list[tuple[int, int, int | None]]:
    """Each seat's turn in a public auction: (seat, its bid or PASS, the
    target it faced — the lowest legal point bid, None when none was left)."""
    high = 0
    out = []
    for seat in bidding_order(dealer):
        target = max(MIN_BID, high + 1)
        out.append((seat, bids[seat], target if target <= ONE_MARK else None))
        if bids[seat] != PASS:
            high = bids[seat]
    return out


def bids_by_rule(slot: int, caution: int, target: int) -> bool:
    """Whether the heuristic bidder with this caution bids `target` holding a hand in `slot`."""
    return slot > 0 and min(ONE_MARK, slot - 1 - caution) >= target


def action_likelihood(action: int, target: int | None, slot: int, caution: int,
                      tremble: float = TREMBLE) -> float | None:
    """P(action | slot, caution) under the trembling rule; None when the turn is no evidence."""
    if target is None or action not in (PASS, target):
        return None
    rule = bids_by_rule(slot, caution, target) == (action == target)
    return tremble / 2 + (1.0 - tremble) * rule


class BidderBelief:
    """A regular's belief about one seat's caution: weights over `grid`."""

    def __init__(self, grid: tuple[int, ...] = GRID, tremble: float = TREMBLE):
        self.grid = grid
        self.tremble = tremble
        self.pi = [1.0 / len(grid)] * len(grid)

    def table(self, action: int, target: int | None) -> tuple[float, ...] | None:
        """The auction's evidence on this seat's dealt hand, by slot; None when there is none."""
        if target is None or action not in (PASS, target):
            return None
        return tuple(
            sum(p * action_likelihood(action, target, slot, c, self.tremble) for p, c in zip(self.pi, self.grid))
            for slot in range(AUCTION_SLOTS)
        )

    def learn(self, action: int, target: int | None, slot: int) -> None:
        """Bayes on a finished hand: the seat's action, the target it faced, its revealed slot."""
        like = [action_likelihood(action, target, slot, c, self.tremble) for c in self.grid]
        if like[0] is None:
            return
        post = [p * x for p, x in zip(self.pi, like)]
        total = sum(post)
        self.pi = [x / total for x in post]


class RegularsPlay:
    """A Zeb that reads the opposing seats' auction (`walt`: a `WaltPlay` with
    a reading). `memory` keeps a `BidderBelief` per (game, seat) across the
    hands of a game (R); without it the belief is flat every hand (R0)."""

    def __init__(self, walt, memory: bool, grid: tuple[int, ...] = GRID, tremble: float = TREMBLE):
        if walt.walt.config.reading is None:
            raise ValueError("the auction is read with a reading's draw: the walt needs a reading")
        self.walt = walt
        self.memory = memory
        self.grid = grid
        self.tremble = tremble
        self.remembers = memory
        self.beliefs: dict[tuple[tuple[int, int], int], BidderBelief] = {}

    def belief(self, game: tuple[int, int] | None, seat: int) -> BidderBelief:
        if not self.memory or game is None:
            return BidderBelief(self.grid, self.tremble)
        key = (game, seat)
        if key not in self.beliefs:
            self.beliefs[key] = BidderBelief(self.grid, self.tremble)
        return self.beliefs[key]

    def auction(self, state: ZebGameState, game: tuple[int, int] | None) -> Auction:
        """The auction's evidence on the two opposing seats' dealt hands, from the viewer's chair."""
        viewer = (state.trick_leader + len(state.current_trick)) % 4
        out: list[tuple[float, ...] | None] = [None] * 4
        for seat, action, target in turns(state.bid_state.bids, state.dealer):
            if seat % 2 != viewer % 2:
                out[seat] = self.belief(game, seat).table(action, target)
        return tuple(out)

    def observe(self, game: tuple[int, int], hand: HandRecord) -> None:
        """A finished hand: each seat's dealt tiles are the tiles it played.
        A shaker forced to open (`forced`, a pass-out rule) is no evidence."""
        for seat, action, target in turns(hand.bids, hand.dealer):
            if hand.forced and seat == hand.dealer:
                continue
            dealt = mask_of(t for s, t in hand.plays if s == seat)
            self.belief(game, seat).learn(action, target, ceiling_slot(dealt))

    def choose(self, states: Sequence[ZebGameState], bid_values: Sequence[int],
               marks: Sequence[tuple[int, int]] | None = None, marks_to_win: int = 7,
               games: Sequence[tuple[int, int]] | None = None) -> list[int]:
        keys = list(games) if games is not None else [None] * len(states)
        return self.walt.choose(states, bid_values, marks, marks_to_win,
                                auctions=[self.auction(s, g) for s, g in zip(states, keys)])

    def __repr__(self) -> str:
        return (f"RegularsPlay(memory={self.memory}, grid={self.grid}, tremble={self.tremble}, "
                f"{self.walt!r})")

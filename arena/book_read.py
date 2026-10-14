# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book-aware reader (#147): Zeb whose belief about two named seats is that
they play by the book.

At each of its decisions the reader weighs the live rung's worlds by the
likelihood that the read seats' past plays are the book's, as the book player
(`arena/book_play.py`) decides them: at each read play the stack's verdict at
that seat's own frame of the time (its hand in the world plus the tiles it has
played since, and the record before the play) is the deciding policy's set,
narrowed as `book_play.decide` narrows it, and

    consistent  the play weighs 1 when the set holds it, else 0;
    uniform     1/|set| when the set holds it, else 0 (restricted choice).

A play no rule decides is silence (weight 1), as Silas reads a fallback
(`silas/src/reading.rs`): the book says nothing there, and the book player's
fallback (Zeb's tile, or the lowest) is not the book's. The least-id pick the
book player makes inside a set is a tie-break, not read, as in Silas. A forced
play is not evidence. Unread seats are uniform, as plain Zeb holds them. The
weight is a product of per-seat factors, each a function of that seat's hand
in the world, so it is cached by (seat, hand).

Which seats are read is a choice (`seats`): the two opponents (the reader of
#147, `BookReader`), or the partner (#150: the book player reading its own
partner by the book, `BookPlay(read=...)`). A read seat's calls may be read
too (`calls`), under `bid-only-enough`, the book team's bidding (the arena's
`heuristic`, call for call): the world's hand for the seat, dealt, must make
the seat's call at its turn and, if it declared, its declaration, as Silas
reads the bid (`silas/README.md` § The record as evidence). The shaker's 30
into three passes is silence: it may be the forced opening, which the record
does not show.

The weights enter through the planner's belief seam (`Walt(belief=...)`,
`CompiledWalt`): the core samples the live rung's worlds as plain Zeb does,
uniform on the seat's cell, and multiplies these weights in. Where the fiber
is no larger than the sample (enumerated), the worlds are the whole fiber and
the weights normalized are the exact posterior, Silas's `exact_given` count.
Where it is sampled, the weighted sample is a self-normalized importance
estimate of that posterior on the uniform draws: no exact draw, and its
effective sample size is logged per decision (`log`). Where no world of the
sample explains the record, the reader plans over the uniform sample, as plain
Zeb, and says so (`fell_back`).

Information-state only (STRATEGY-FUSION FENCE, CLAUDE.md): the likelihood is
the book's verdict at a frame (the seat's own hand and the public record), and
every value is Zeb's plan over the reader's own cell. Nothing reads the deal.

Spec (arena `--team-*` play): `bookread:<consistent|uniform>/<walt options>`
(the opponents read); the partner read is the book player's option
`read=<likelihood>[+calls]` (`arena/book_play.py`).
"""
from __future__ import annotations

from typing import Sequence

from forge.zeb.types import ZebGameState

from functools import lru_cache

from arena.auction import BidContext, bidding_order, legal_bids
from arena.bidders import HeuristicBidder
from arena.book_play import FOLLOW_ORDER, LEAD_ORDER, NO_COUNT, compiled
from arena.walt_play import WaltPlay, parse_walt_spec
from scenario.planner import Situation, World
from scenario.rules import bit, rules_for

LIKELIHOODS = ("consistent", "uniform")
SEATS = ("opponents", "partner")
CACHE_CAP = 1 << 20   # verdicts kept across decisions before the cache is cleared


def verdict(decl: int, seat: int, hand: int, bidder: int, bid: int,
            plays: tuple[tuple[int, int], ...]) -> int | None:
    """The book stack's deciding set at `seat`'s decision holding `hand` after
    `plays`, as a tile mask, or None where no rule fires. Equal to
    `book_play.ask`'s `allowed`: policies are asked in the phase's order and
    stop at the first that fires, unless it only forbids count, in which case
    the later firing ones narrow it (`book_play.decide`)."""
    from scenario_core import SilasFrame
    f = SilasFrame(decl, seat, hand, bidder, bid, list(plays))
    left: int | None = None
    for name in (LEAD_ORDER if len(plays) % 4 == 0 else FOLLOW_ORDER):
        layer, tiles = compiled(name, decl).allowed(f)
        if layer == "fallback":
            continue
        mask = 0
        for t in tiles:
            mask |= bit(t)
        if left is None:
            left = mask
            if (name, layer) not in NO_COUNT:
                return left
        elif left & mask:
            left &= mask
    return left


def read_seats(viewer: int, seats: str) -> tuple[int, ...]:
    """The seats `viewer` reads: its two opponents, or its partner."""
    if seats not in SEATS:
        raise ValueError(f"bookread: the read seats are one of {SEATS}, not {seats!r}")
    return ((viewer + 1) % 4, (viewer + 3) % 4) if seats == "opponents" else ((viewer + 2) % 4,)


_BIDDER = HeuristicBidder()   # bid-only-enough, call for call (arena/test_book_play.py)


@lru_cache(maxsize=1 << 18)
def calls_weight(seat: int, dealt: int, dealer: int, bids: tuple[int, ...], bidder: int, bid: int,
                 decl: int) -> float:
    """1 when `bid-only-enough` holding `dealt` (a tile mask) makes `seat`'s
    call in the auction `bids` (seat order) and, as the declarer, its
    declaration; else 0. The shaker's 30 into three passes is silence for
    the call (it may be forced)."""
    order = bidding_order(dealer)
    before = order[:order.index(seat)]
    high, high_seat = 0, -1
    for s in before:
        if bids[s] > high:
            high, high_seat = bids[s], s
    hand = tuple(t for t in range(28) if dealt >> t & 1)
    if not (seat == dealer and high == 0 and bids[seat] == 30):
        ctx = BidContext(hand=hand, seat=seat, dealer=dealer,
                         bids=tuple(bids[s] if s in before else -1 for s in range(4)),
                         high_bid=high, high_seat=high_seat, legal=legal_bids(high))
        if _BIDDER.bid(ctx, None) != bids[seat]:
            return 0.0
    if seat == bidder and _BIDDER.declare(hand, bid, None) != decl:
        return 0.0
    return 1.0


def book_weights(sit: Situation, worlds: list[World], bid: int, likelihood: str,
                 seats: str = "opponents", auction: tuple[int, tuple[int, ...]] | None = None,
                 ) -> tuple[list[float], dict]:
    """The weights over `worlds` that the viewer's read `seats` played by the
    book (normalized; the uniform where none explains the record), and the
    decision's log row. `bid` is the bid as the table says it: the book's
    frame needs 84, not the 42 points the planner's contract is. With
    `auction` (the dealer and the calls in seat order), the read seats' calls
    are read too (`calls_weight`)."""
    rules = rules_for(sit.decl_id)
    read = read_seats(sit.viewer, seats)
    ev = []   # (seat, play index, tile, the tiles that seat had played before)
    before = [0, 0, 0, 0]
    for i, (seat, tile) in enumerate(sit.plays):
        if seat in read:
            ev.append((seat, i, tile, before[seat]))
        before[seat] |= bit(tile)
    factors: dict[tuple[int, int], float] = {}

    def factor(seat: int, now: int) -> float:
        got = factors.get((seat, now))
        if got is not None:
            return got
        w, mine = 1.0, sit.played_by(seat)
        for s, i, tile, had in ev:
            if s != seat:
                continue
            hand = (now | mine) & ~had
            led = sit.plays[i - i % 4][1] if i % 4 else None
            if rules.legal(hand, led).bit_count() < 2:
                continue
            key = (sit.decl_id, seat, hand, sit.bidder, bid, sit.plays[:i])
            if key not in _VERDICTS:
                if len(_VERDICTS) >= CACHE_CAP:
                    _VERDICTS.clear()
                _VERDICTS[key] = verdict(sit.decl_id, seat, hand, sit.bidder, bid, sit.plays[:i])
            a = _VERDICTS[key]
            if a is None:
                continue
            if not (a >> tile) & 1:
                w = 0.0
                break
            if likelihood == "uniform":
                w /= a.bit_count()
        if w and auction is not None:
            w *= calls_weight(seat, now | mine, auction[0], auction[1], sit.bidder, bid, sit.decl_id)
        factors[(seat, now)] = w
        return w

    weights = []
    for h in worlds:
        x = 1.0
        for seat in read:
            x *= factor(seat, h[seat])
        weights.append(x)
    total, sq = sum(weights), sum(x * x for x in weights)
    fell_back = total == 0.0
    row = dict(trick=len(sit.plays) // 4, worlds=len(worlds),
               ess=0.0 if fell_back else total * total / sq, fell_back=fell_back)
    return ([1.0] * len(worlds) if fell_back else [x / total for x in weights]), row


_VERDICTS: dict[tuple, int | None] = {}   # verdicts by (decl, seat, hand, bidder, bid, plays), per process


def _weigh(job: tuple) -> tuple[list[float], dict]:
    return book_weights(*job)


class BookBelief:
    """The belief seam's callable (`book_weights` at each decision), with a
    batch form (`many`) that spreads a round's decisions over `workers`
    processes. Each decision's weights are a pure function of it, so the
    split changes nothing. `table` maps a decision's key to what the
    `Situation` does not carry: the bid as the table says it, the dealer and
    the calls (seat order)."""

    def __init__(self, likelihood: str = "consistent", workers: int = 8, seats: str = "opponents",
                 calls: bool = False):
        if likelihood not in LIKELIHOODS:
            raise ValueError(f"bookread: the likelihood is one of {LIKELIHOODS}, not {likelihood!r}")
        read_seats(0, seats)
        self.likelihood, self.workers, self.seats, self.calls = likelihood, workers, seats, calls
        self.table: dict[tuple, tuple[int, int, tuple[int, ...]]] = {}
        self.log: list[dict] = []
        self._pool = None

    def note(self, states: Sequence[ZebGameState], bid_values: Sequence[int]) -> None:
        """Record, for this round's decisions, what the weights need beyond the situation."""
        from scenario.planner import situation_from_state
        from scenario.record import remaining_hand
        self.table.clear()
        for s, b in zip(states, bid_values):
            sit = situation_from_state(s, int(b))
            self.table[_key(sit, remaining_hand(s, sit.viewer))] = (
                int(b), int(s.dealer), tuple(int(x) for x in s.bid_state.bids))

    def _job(self, sit: Situation, worlds: list[World]) -> tuple:
        raw, dealer, bids = self.table.get(_key(sit, worlds[0][sit.viewer]), (sit.bid, 0, ()))
        auction = (dealer, bids) if self.calls and bids else None
        return sit, worlds, raw, self.likelihood, self.seats, auction

    def __call__(self, sit: Situation, worlds: list[World]) -> list[float]:
        weights, row = book_weights(*self._job(sit, worlds))
        self.log.append(row)
        return weights

    def many(self, sits: Sequence[Situation], worlds: Sequence[list[World]]) -> list[list[float]]:
        jobs = [self._job(s, w) for s, w in zip(sits, worlds)]
        if self.workers > 1 and len(jobs) > 1:
            if self._pool is None:
                import multiprocessing
                self._pool = multiprocessing.get_context("spawn").Pool(self.workers)
            out = self._pool.map(_weigh, jobs, chunksize=1)
        else:
            out = [_weigh(j) for j in jobs]
        self.log.extend(row for _, row in out)
        return [w for w, _ in out]


def _key(sit: Situation, own: int) -> tuple:
    return (sit.decl_id, sit.viewer, sit.bidder, sit.plays, own)


class BookReader:
    """Zeb reading its opponents by the book, as an arena `PlayPolicy`."""

    def __init__(self, likelihood: str, zeb: WaltPlay):
        if zeb.walt.belief is not None or zeb.walt.config.draw != "uniform":
            raise ValueError("bookread: the reader is plain Zeb's draw with the book's belief, nothing else")
        self.belief = BookBelief(likelihood)
        zeb.walt.belief = self.belief
        self.zeb = zeb

    def choose(self, states: Sequence[ZebGameState], bid_values: Sequence[int],
               marks: Sequence[tuple[int, int]] | None = None, marks_to_win: int = 7) -> list[int]:
        self.belief.note(states, bid_values)
        return self.zeb.choose(states, bid_values, marks, marks_to_win)

    def __repr__(self) -> str:
        return f"BookReader({self.belief.likelihood}, {self.zeb!r})"


def parse_bookread_spec(arg: str) -> BookReader:
    """`<consistent|uniform>/<walt options>`, e.g. `consistent/k0,n512,F=net.npz,f=16`."""
    likelihood, _, walt = arg.partition("/")
    return BookReader(likelihood or "consistent", parse_walt_spec(walt))

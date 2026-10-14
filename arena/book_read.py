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
play is not evidence. Unread seats (the reader's partner) are uniform, as
plain Zeb holds them. The weight is a product of per-seat factors, each a
function of that seat's hand in the world, so it is cached by (seat, hand).

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

Spec (arena `--team-*` play): `bookread:<consistent|uniform>/<walt options>`.
"""
from __future__ import annotations

from typing import Sequence

from forge.zeb.types import ZebGameState

from arena.book_play import FOLLOW_ORDER, LEAD_ORDER, NO_COUNT, compiled
from arena.walt_play import WaltPlay, parse_walt_spec
from scenario.planner import Situation, World
from scenario.rules import bit, rules_for

LIKELIHOODS = ("consistent", "uniform")
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


def book_weights(sit: Situation, worlds: list[World], bid: int, likelihood: str) -> tuple[list[float], dict]:
    """The weights over `worlds` that the viewer's two opponents played by the
    book (normalized; the uniform where none explains the record), and the
    decision's log row. `bid` is the bid as the table says it: the book's
    frame needs 84, not the 42 points the planner's contract is."""
    rules = rules_for(sit.decl_id)
    read = ((sit.viewer + 1) % 4, (sit.viewer + 3) % 4)
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
        factors[(seat, now)] = w
        return w

    weights = [factor(read[0], h[read[0]]) * factor(read[1], h[read[1]]) for h in worlds]
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
    split changes nothing. `raw_bid` maps a decision's key to the bid as the
    table says it."""

    def __init__(self, likelihood: str = "consistent", workers: int = 8):
        if likelihood not in LIKELIHOODS:
            raise ValueError(f"bookread: the likelihood is one of {LIKELIHOODS}, not {likelihood!r}")
        self.likelihood, self.workers = likelihood, workers
        self.raw_bid: dict[tuple, int] = {}
        self.log: list[dict] = []
        self._pool = None

    def _job(self, sit: Situation, worlds: list[World]) -> tuple:
        return sit, worlds, self.raw_bid.get(_key(sit, worlds[0][sit.viewer]), sit.bid), self.likelihood

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
        from scenario.planner import situation_from_state
        from scenario.record import remaining_hand
        raw = self.belief.raw_bid
        raw.clear()
        for s, b in zip(states, bid_values):
            sit = situation_from_state(s, int(b))
            raw[_key(sit, remaining_hand(s, sit.viewer))] = int(b)
        return self.zeb.choose(states, bid_values, marks, marks_to_win)

    def __repr__(self) -> str:
        return f"BookReader({self.belief.likelihood}, {self.zeb!r})"


def parse_bookread_spec(arg: str) -> BookReader:
    """`<consistent|uniform>/<walt options>`, e.g. `consistent/k0,n512,F=net.npz,f=16`."""
    likelihood, _, walt = arg.partition("/")
    return BookReader(likelihood or "consistent", parse_walt_spec(walt))

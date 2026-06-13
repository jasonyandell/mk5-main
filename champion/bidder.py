"""Auction v0: Gus-backed risk-budget bidding (rung #21).

Roberson Ch 2 as an executable policy ("bid only enough", `supported` at
wave 2.B.2), with the willingness number coming from simulation instead of
static arithmetic: P(make) per (trump, threshold) from Gus playing all four
seats (`gus/bidding/simulate.py`), scored by a pluggable marks utility
(`champion/utility.py` — MarkEV for the score-blind v0 criterion,
MarksToSeven for the rung-#27 score-conditioned one).

The policy at its one turn to speak: walk the legal raises from the
cheapest up and take the first whose utility beats `margin`; pass
otherwise. For point bids utility is monotone non-increasing in the bid,
so this is the minimum positive-utility bid; from far behind the walk can
step past negative point bids onto a positive two-mark gamble, which is
the score-conditioned desperation bid emerging rather than being authored.

A static prefilter (the wave-2.B risk-budget arithmetic plus a doubles
count) skips simulation on hands no trump structure could carry — the
same hands the simulator would price below every threshold.
"""
from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence

from arena.auction import PASS, BidContext, contract_points
from arena.hand_metrics import best_trump
from forge.oracle.tables import DOMINO_IS_DOUBLE

from .utility import BidUtility, MarkEV

# hand -> {decl_id: bidding team's final points, one per simulated world}
PointsEvaluator = Callable[[tuple[int, ...]], Mapping[int, Sequence[int]]]


def _p_make(points: Sequence[int], threshold: int) -> float:
    return sum(1 for p in points if p >= threshold) / len(points)


class GusBidder:
    """BidPolicy over a simulated P(make) table, one evaluation per hand."""

    def __init__(
        self,
        evaluator: PointsEvaluator,
        utility: BidUtility | None = None,
        *,
        prefilter_min_trumps: int = 3,
        margin: float = 0.0,
    ):
        self.evaluator = evaluator
        self.utility = utility or MarkEV()
        self.prefilter_min_trumps = prefilter_min_trumps
        self.margin = margin
        self._cache: dict[tuple[int, ...], dict[int, list[int]]] = {}

    def _points(self, hand: tuple[int, ...]) -> dict[int, list[int]]:
        key = tuple(sorted(hand))
        if key not in self._cache:
            self._cache[key] = {
                decl: list(pts) for decl, pts in self.evaluator(key).items()
            }
        return self._cache[key]

    def _worth_evaluating(self, hand: tuple[int, ...]) -> bool:
        if best_trump(hand, self.prefilter_min_trumps) is not None:
            return True
        return sum(1 for d in hand if DOMINO_IS_DOUBLE[d]) >= 3

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        if not self._worth_evaluating(ctx.hand):
            return PASS
        table = self._points(ctx.hand)
        for value in ctx.legal:
            threshold = contract_points(value)
            p = max(_p_make(pts, threshold) for pts in table.values())
            u = self.utility.value(
                p, value,
                team=ctx.team, marks=ctx.marks, marks_to_win=ctx.marks_to_win,
            )
            if u > self.margin:
                return value
        return PASS

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        table = self._points(hand)  # cached unless the bid was forced
        threshold = contract_points(bid)
        return max(
            table,
            key=lambda decl: (
                _p_make(table[decl], threshold),
                sum(table[decl]) / len(table[decl]),
            ),
        )

    def __repr__(self) -> str:
        return (
            f"GusBidder(utility={self.utility!r}, margin={self.margin}, "
            f"prefilter_min_trumps={self.prefilter_min_trumps})"
        )


class GusPointsEvaluator:
    """The default evaluator: Gus in all four seats, batched over trumps.

    Pip trumps 0-6 plus doubles-trump (7); deterministic per hand via a
    fixed deal seed, so paired-seed reruns hit the GusBidder cache.
    """

    def __init__(
        self,
        adapter: str | None = None,
        device: str | None = None,
        n_samples: int = 32,
        seed: int = 42,
    ):
        from gus.bidding.evaluate import ADAPTER_DEFAULT, TRUMP_IDS, load_gus
        from gus.bidding.simulate import _pick_device, simulate_all_gus_batch

        self._simulate = simulate_all_gus_batch
        self._trump_ids = TRUMP_IDS
        self.adapter = str(adapter or ADAPTER_DEFAULT)
        self.device = device or _pick_device()
        self.n_samples = n_samples
        self.seed = seed
        self._model, self._is_voids = load_gus(self.adapter, self.device)

    def __call__(self, hand: tuple[int, ...]) -> dict[int, list[int]]:
        pts = self._simulate(
            self._model, self._is_voids, list(hand),
            self._trump_ids, self.n_samples, self.device, self.seed,
        )
        return {decl: t.tolist() for decl, t in pts.items()}

    def __repr__(self) -> str:
        return (
            f"GusPointsEvaluator(adapter={self.adapter!r}, "
            f"n_samples={self.n_samples}, device={self.device!r})"
        )

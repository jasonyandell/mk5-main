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

With ``maximize=True`` (rung #31) the policy instead takes the utility-
*maximizing* legal bid. For one-mark point bids the argmax is still the
minimum (utility is monotone), but a hand strong enough that the two-mark
contract's ``2 * P(make 42)`` beats the one-mark ``P(make 30)`` now climbs
to 84 — so the bid *level* tracks hand strength and bid magnitude becomes a
live auction signal rather than a flat 30/31 (it is otherwise near-dead, the
belief feature's #24 magnitude channel having nothing to learn from).

A static prefilter (the wave-2.B risk-budget arithmetic plus a doubles
count) skips simulation on hands no trump structure could carry — the
same hands the simulator would price below every threshold.
"""
from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence

from arena.auction import MIN_BID, ONE_MARK, PASS, BidContext, contract_points
from arena.hand_metrics import best_trump
from forge.oracle.tables import DOMINO_IS_DOUBLE

from .utility import BidUtility, MarkEV

# hand -> {decl_id: bidding team's final points, one per simulated world}
PointsEvaluator = Callable[[tuple[int, ...]], Mapping[int, Sequence[int]]]
# hand -> {decl_id: {threshold: P(make)}} — the table the policy actually reads;
# the distilled bid_net (rung #22) supplies this directly, skipping simulation.
PMakeFn = Callable[[tuple[int, ...]], Mapping[int, Mapping[int, float]]]

# Point thresholds a contract can demand: 30..42 (a mark bid caps at all 42).
THRESHOLDS = tuple(range(MIN_BID, ONE_MARK + 1))


def _p_make(points: Sequence[int], threshold: int) -> float:
    return sum(1 for p in points if p >= threshold) / len(points)


class GusBidder:
    """BidPolicy over a simulated P(make) table, one evaluation per hand."""

    def __init__(
        self,
        evaluator: PointsEvaluator | None = None,
        utility: BidUtility | None = None,
        *,
        prefilter_min_trumps: int = 3,
        margin: float = 0.0,
        pmake_fn: PMakeFn | None = None,
        maximize: bool = False,
    ):
        if evaluator is None and pmake_fn is None:
            raise ValueError("GusBidder needs an `evaluator` or a `pmake_fn`")
        self.evaluator = evaluator
        self.pmake_fn = pmake_fn
        self.utility = utility or MarkEV()
        self.prefilter_min_trumps = prefilter_min_trumps
        self.margin = margin
        self.maximize = maximize
        self._cache: dict[tuple[int, ...], dict[int, list[int]]] = {}
        self._pm_cache: dict[tuple[int, ...], dict[int, dict[int, float]]] = {}

    def _points(self, hand: tuple[int, ...]) -> dict[int, list[int]]:
        key = tuple(sorted(hand))
        if key not in self._cache:
            self._cache[key] = {
                decl: list(pts) for decl, pts in self.evaluator(key).items()
            }
        return self._cache[key]

    def _pmake(self, hand: tuple[int, ...]) -> dict[int, dict[int, float]]:
        """{decl: {threshold: P(make)}}, from the net if given else simulated
        points. One evaluation per hand, cached across bid and declaration."""
        key = tuple(sorted(hand))
        if key not in self._pm_cache:
            if self.pmake_fn is not None:
                self._pm_cache[key] = {
                    decl: dict(row) for decl, row in self.pmake_fn(key).items()
                }
            else:
                self._pm_cache[key] = {
                    decl: {t: _p_make(pts, t) for t in THRESHOLDS}
                    for decl, pts in self._points(key).items()
                }
        return self._pm_cache[key]

    def _worth_evaluating(self, hand: tuple[int, ...]) -> bool:
        if best_trump(hand, self.prefilter_min_trumps) is not None:
            return True
        return sum(1 for d in hand if DOMINO_IS_DOUBLE[d]) >= 3

    def _utility_of(self, value: int, pm: Mapping[int, Mapping[int, float]],
                    ctx: BidContext) -> float:
        """Marks utility of bidding `value`, under the best declaration's P(make)."""
        threshold = contract_points(value)
        p = max(row[threshold] for row in pm.values())
        return self.utility.value(
            p, value,
            team=ctx.team, marks=ctx.marks, marks_to_win=ctx.marks_to_win,
        )

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        if not self._worth_evaluating(ctx.hand):
            return PASS
        pm = self._pmake(ctx.hand)
        if self.maximize:
            # Rung #31: take the utility-maximizing legal bid (ties keep the
            # cheapest, "bid only enough" among equals); PASS if none clears.
            best_val, best_u = PASS, self.margin
            for value in ctx.legal:
                u = self._utility_of(value, pm, ctx)
                if u > best_u:
                    best_u, best_val = u, value
            return best_val
        # Rung #21: the minimum positive-utility bid (cheapest that clears).
        for value in ctx.legal:
            if self._utility_of(value, pm, ctx) > self.margin:
                return value
        return PASS

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        pm = self._pmake(hand)  # cached unless the bid was forced
        threshold = contract_points(bid)
        if self.pmake_fn is None:
            # Break ties on P(make) by higher mean points (needs the points path).
            pts = self._points(hand)
            return max(
                pm,
                key=lambda decl: (pm[decl][threshold], sum(pts[decl]) / len(pts[decl])),
            )
        return max(pm, key=lambda decl: pm[decl][threshold])

    def __repr__(self) -> str:
        src = "net" if self.pmake_fn is not None else "sim"
        mode = "max" if self.maximize else "min"
        return (
            f"GusBidder(source={src}, mode={mode}, utility={self.utility!r}, "
            f"margin={self.margin}, prefilter_min_trumps={self.prefilter_min_trumps})"
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


class NetPointsEvaluator:
    """P(make) table from the distilled bid_net (rung #22) — <1ms per hand.

    Returns {decl_id: {threshold: P(make)}} for the GusBidder `pmake_fn` path,
    replacing the per-hand Gus simulation (≈1s) with a single MLP forward pass.
    The net lives in `champion/bid_net.py`; default weights `champion/bid_net.pt`.
    Covers 9 declarations (EVAL_DECLS, including no-trump=9) × thresholds 30..42.
    """

    def __init__(self, model_path: str = "champion/bid_net.pt", device: str = "cpu"):
        import torch

        from champion.bid_net import (
            BID_THRESHOLDS, BidNet, EVAL_DECLS, FEATURE_DIM, featurize_hand,
        )

        self._torch = torch
        self._featurize = featurize_hand
        self._decls = list(EVAL_DECLS)
        self._thresholds = list(BID_THRESHOLDS)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.model = BidNet(in_dim=ckpt.get("feature_dim", FEATURE_DIM))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.model_path = model_path

    def __call__(self, hand: tuple[int, ...]) -> dict[int, dict[int, float]]:
        x = self._featurize(tuple(hand)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            table = self.model(x)[0].cpu()  # [9, 13]
        return {
            self._decls[i]: {
                self._thresholds[j]: float(table[i, j])
                for j in range(len(self._thresholds))
            }
            for i in range(len(self._decls))
        }

    def __repr__(self) -> str:
        return f"NetPointsEvaluator(model_path={self.model_path!r}, device={self.device!r})"

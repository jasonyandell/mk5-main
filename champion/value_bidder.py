"""ValueBidder (jud v0, GitHub #32): the value-native auction policy.

Where `GusBidder`/`net:wp` price a contract by a double-dummy oracle's P(make),
this bidder prices it by the REALIZED-value head `champion.margin_net.MarginNet`
— a categorical over the declaring team's captured points, trained on arena
outcomes. That is the whole jud v0 move: the pricing path stops asking "would a
perfect declarer make this?" and starts asking "how often did *these* players
actually take the points?", dissolving the #26 over-bidder ("the bidder writes
checks the player can't cash") without touching the play side (`lens:ev`).

The policy at its one turn to speak (Roberson's "bid only enough", rung #21's
minimum-positive-utility convention — rung #31 measured bid-magnitude
maximization dead):

  1.  Build the P(pts ≥ threshold) table ONCE per hand via
      ``MarginNet.pmake_table`` at the hypothetical completed auction ("suppose I
      win and lead trick 1"). The auction is canonicalized inside pmake_table
      (`canonical_auction`: level-blind, later seats masked) — the ValueBidder
      never bypasses it, so serving features are byte-identical to the training
      corpus (see `champion/test_value_bidder.py`).
  2.  Walk the legal raises from cheapest up; take the FIRST whose score-
      conditioned marks utility (`champion.utility`) clears the margin. For point
      bids the exceedance P(pts ≥ bid) is monotone non-increasing in the bid, so
      the first positive-utility bid is the minimum positive-utility bid; a strong
      hand can still step past negative point bids onto a positive two-mark 84.
  3.  Declare the argmax-exceedance declaration at the chosen contract's
      threshold, cached from bid time (`declare` reuses it).

There is deliberately NO ``pmake_scale`` in this path: the realized-value head is
already calibrated to what the players cash, so rescaling P(make) — the #26
optimism-correction knob the oracle bidder needed — would be double-counting.
There is also no static hand prefilter: the value head prices every hand, and a
hand no trump structure can carry simply exceeds no threshold with positive
utility and passes on its own.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping

import torch

from arena.auction import PASS, BidContext, BidPolicy, contract_points
from champion.margin_net import FEATURE_DIM, MarginNet
from champion.utility import BidUtility, MarksToSeven

_DEFAULT_MODEL = Path("champion/margin_net.pt")


def load_margin_net(model_path: str | Path = _DEFAULT_MODEL, device: str = "cpu") -> MarginNet:
    """Load a trained MarginNet checkpoint (mirrors ``margin_net.evaluate``)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = MarginNet(in_dim=ckpt.get("feature_dim", FEATURE_DIM)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


class ValueBidder(BidPolicy):
    """A bidder that prices contracts with the realized-value head.

    ``utility`` defaults to ``MarksToSeven`` (score-conditioned marks WP), matching
    ``net:wp``. The MarginNet forward is a sub-millisecond MLP pass, so the whole
    table is (re)built per hand with a small cache to serve ``declare`` from the
    same evaluation that chose the bid.
    """

    def __init__(
        self,
        model: MarginNet,
        utility: BidUtility | None = None,
        *,
        margin: float = 0.0,
    ) -> None:
        self.model = model
        self.utility = utility or MarksToSeven()
        self.margin = margin
        self.model.eval()
        # (sorted hand, canonical-query bids, seat, dealer) -> {decl: {threshold: P}}
        self._table_cache: dict[tuple, dict[int, dict[int, float]]] = {}
        # (sorted hand, won bid value) -> argmax declaration, cached at bid time
        # so declare() returns exactly the declaration the winning bid was priced
        # under (mirrors GusBidder/BeliefBidder).
        self._decl_cache: dict[tuple[tuple[int, ...], int], int] = {}

    # -- pricing --------------------------------------------------------------

    @staticmethod
    def _query_bids(ctx: BidContext) -> tuple[int, ...]:
        """BidContext bids → canonical-query form: -1 (not-yet) and PASS both → 0.

        ``canonical_auction`` clamps earlier-seat pass/not-yet to 0, stamps the
        bidder's own seat with the constant CANON_BID, and masks later seats to 0
        regardless — so this mapping makes the ValueBidder's decision-time view
        featurize byte-identically to the corpus view of the same auction.
        """
        return tuple(b if b > 0 else 0 for b in ctx.bids)

    def _table(self, ctx: BidContext) -> dict[int, dict[int, float]]:
        """{decl: {threshold 30..42: P(pts ≥ threshold)}} at the hypothetical root."""
        bids = self._query_bids(ctx)
        key = (tuple(sorted(ctx.hand)), bids, ctx.seat, ctx.dealer)
        if key not in self._table_cache:
            self._table_cache[key] = self.model.pmake_table(
                ctx.hand, bids, ctx.seat, ctx.dealer,
            )
        return self._table_cache[key]

    def _utility_of(
        self, value: int, table: Mapping[int, Mapping[int, float]], ctx: BidContext,
    ) -> float:
        """Marks utility of bidding ``value`` under the best declaration's P(make)."""
        threshold = contract_points(value)
        p = max(row[threshold] for row in table.values())
        return self.utility.value(
            p, value,
            team=ctx.team, marks=ctx.marks, marks_to_win=ctx.marks_to_win,
        )

    @staticmethod
    def _best_decl(value: int, table: Mapping[int, Mapping[int, float]]) -> int:
        """argmax-exceedance declaration at the contract's threshold."""
        threshold = contract_points(value)
        return max(table, key=lambda d: table[d][threshold])

    # -- BidPolicy interface --------------------------------------------------

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        table = self._table(ctx)
        # Minimum positive-utility bid (cheapest legal raise that clears margin).
        for value in ctx.legal:
            if self._utility_of(value, table, ctx) > self.margin:
                self._decl_cache[(tuple(sorted(ctx.hand)), int(value))] = (
                    self._best_decl(value, table)
                )
                return value
        return PASS

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        """The declaration the winning bid was priced under, cached at bid time.

        Cold path (a forced-open bid the arena never issues to this bidder, or a
        call without a matching ``bid``): re-price the single won value at an empty
        auction and take its argmax declaration.
        """
        cached = self._decl_cache.get((tuple(sorted(hand)), int(bid)))
        if cached is not None:
            return cached
        ctx = BidContext(
            hand=tuple(hand), seat=0, dealer=0, bids=(-1, -1, -1, -1),
            high_bid=0, high_seat=-1, legal=(int(bid),),
        )
        return self._best_decl(int(bid), self._table(ctx))

    def __repr__(self) -> str:
        return f"ValueBidder(utility={self.utility!r}, margin={self.margin})"

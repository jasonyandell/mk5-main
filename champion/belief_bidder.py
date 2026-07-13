"""Belief-conditioned bidder — the keystone of champion rung #26.

The self-play loop closes only if the bidder *changes when the belief model
changes*. This policy makes that true: for each candidate contract it builds the
HYPOTHETICAL COMPLETED AUCTION ("suppose I win this contract and lead trick 1"),
runs the same batched oracle E[Q] path the play side uses
(``arena.lens_play.LensPlay``), and importance-weights the oracle's sampled
opponent worlds by the #24 auction-belief posterior (``champion.belief``). From
the belief-weighted E[Q] PDF it derives P(make) per contract, scores each by the
score-conditioned marks utility (``champion.utility``), and bids the
utility-maximizing contract.

Two locked design choices keep this honest (see the spec / [[champion]] #26):

1.  The ORACLE drives E[points]; the belief is ONLY *soft* importance weights
    (``uniform_mix=0.1``). With ``belief_model=None`` the weights are exactly
    uniform 1/M and this degrades to the pure-oracle bidder — never depend on
    belief sharpness. The belief at bid time is measured-soft (seat-ESS ~2.86/3).

2.  Only ``suppose-I-WIN`` candidates are built. The bidding seat is the auction
    winner (relative seat 0, ``is_winner=1``) and LEADS trick 1, byte-identical
    to how the #26 corpus is generated (``forge.cli.generate_eq_from_snapshots``
    makes the winner lead trick 1 via ``GameStateTensor.from_deals(bidders=...)``).
    suppose-opponent-wins is out-of-distribution and is never constructed.

The oracle Q depends only on (decl, bidder, my hand, worlds) — NOT on the bid
value (the GameStateTensor carries decl + bidder, not the bid). So one oracle
forward over the distinct candidate declarations suffices; the bid value enters
only through the belief feature (``win_bid_norm``) and the P(make) threshold.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

import torch

from arena.auction import PASS, BidContext, BidPolicy, contract_points
from arena.hand_metrics import best_trump
from champion.belief import belief_weights_for_worlds, effective_sample_size
from champion.utility import BidUtility, MarksToSeven
from forge.bidding.schema import EVAL_DECLS
from forge.eq.generate.actions import _p_make_from_pdf
from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.model import query_model
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.oracle.tables import DOMINO_IS_DOUBLE
from forge.zeb.eq_player import zeb_states_to_game_state_tensor
from forge.zeb.types import BidState, GamePhase, ZebGameState

# Placeholder opponent hand: never read by the oracle (worlds are re-sampled) and
# never read by the belief (it reads only my hand + bid_state + decl). The values
# are dummies of the right length so ZebGameState/GameStateTensor pack cleanly.
_OPP_PLACEHOLDER: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


class BeliefBidder(BidPolicy):
    """A bidder whose contract choice moves with the belief model.

    Constructor mirrors the spec exactly. ``belief_model`` may be None (the pure
    oracle degrade path: uniform world weights). ``oracle_model`` is the already
    loaded arena oracle — it is NOT reloaded here.
    """

    def __init__(
        self,
        belief_model,
        oracle_model,
        *,
        is_voids: bool = True,
        device: str = "mps",
        n_samples: int = 32,
        utility: BidUtility | None = None,
        maximize: bool = True,
        uniform_mix: float = 0.1,
        tau: float = 1.0,
        prefilter_min_trumps: int = 3,
        margin: float = 0.0,
        pmake_scale: float = 1.0,
    ):
        self.belief_model = belief_model
        self.oracle_model = oracle_model
        self.is_voids = is_voids
        self.device = device
        self.n_samples = n_samples
        self.utility = utility or MarksToSeven()
        self.maximize = maximize
        self.uniform_mix = uniform_mix
        self.tau = tau
        self.prefilter_min_trumps = prefilter_min_trumps
        self.margin = margin
        # Optimism correction (#26): the oracle E[Q] is double-dummy (perfect play
        # by all four seats), so its P(make) runs systematically optimistic vs the
        # realized PIMC rate. NOTE: pmake_scale=0.70 is a TUNED KNOB, not a measured
        # constant — the prose "0.58/0.83 gap at bid 30" was never computed. The real
        # gap, measured from existing data (champion/optimism_meter.py -> optimism_gap.json),
        # is oracle 0.64 / realized 0.52 at bid 30 (ratio ~0.81) and is strongly
        # BID-DEPENDENT (realized make-rate falls 0.52 -> 0.11 across bids 30-42), so a
        # single global scalar is the wrong SHAPE. Scaling P(make) at utility time
        # nudges toward achievable contracts without touching the cached raw oracle
        # P(make). 1.0 = raw double-dummy (default).
        self.pmake_scale = pmake_scale

        if self.belief_model is not None:
            self.belief_model.eval()
        self.oracle_model.eval()

        self.decls = list(EVAL_DECLS)  # [0,1,2,3,4,5,6,7,9]
        self._sampler: WorldSamplerMRV | None = None
        self._tokenizer: GPUTokenizer | None = None
        # P(make) cache: (sorted hand, bids, decl, value) -> float. Score is NOT
        # part of the key — it enters only via the utility at call time.
        self._pmake_cache: dict[tuple, float] = {}
        self.last_ess: float | None = None

    # -- GPU capacity (mirror LensPlay._ensure_capacity) ----------------------

    def _ensure_capacity(self, n_games: int) -> None:
        if self._sampler is None or self._sampler.max_games < n_games:
            self._sampler = WorldSamplerMRV(
                max_games=n_games, max_samples=self.n_samples, device=self.device,
            )
        batch_needed = n_games * self.n_samples
        if self._tokenizer is None or self._tokenizer.max_batch < batch_needed:
            self._tokenizer = GPUTokenizer(max_batch=batch_needed, device=self.device)

    # -- prefilter (reuse GusBidder._worth_evaluating) ------------------------

    def _worth_evaluating(self, hand: tuple[int, ...]) -> bool:
        if best_trump(hand, self.prefilter_min_trumps) is not None:
            return True
        return sum(1 for d in hand if DOMINO_IS_DOUBLE[d]) >= 3

    # -- hypothetical completed-auction state ---------------------------------

    def hypothetical_state(
        self, hand: tuple[int, ...], bids: tuple[int, ...], my_seat: int,
        cand_decl: int, cand_value: int,
    ) -> ZebGameState:
        """Build the PLAYING-phase state for "suppose I win (cand_decl, cand_value)".

        The state is byte-identical (for what the oracle + belief read) to the
        #26 corpus winner-leads-trick-1 setup:
          * phase=PLAYING, play_history=(), current_trick=(), played=frozenset()
          * trick_leader=my_seat, bidder=my_seat, decl_id=cand_decl
          * bid_state stamps my_seat=cand_value, EARLIER real bids at their seats,
            LATER seats (who speak after me) = PASS(0); high_bidder=my_seat.
        ``current_player`` then returns my_seat (empty trick), so the belief reads
        "I won, about to lead trick 1" at relative seat 0 (is_winner=1).
        """
        stamped = [0, 0, 0, 0]
        for s in range(4):
            if s == my_seat:
                stamped[s] = cand_value
            elif s < len(bids) and bids[s] is not None and int(bids[s]) > 0:
                # An earlier real positive bid — keep it at its seat (the auction
                # feature attributes the bid to the seat that made it). Later /
                # passed / not-yet seats stay at PASS(0).
                stamped[s] = int(bids[s])
        bid_state = BidState(
            bids=tuple(stamped), high_bidder=my_seat, high_bid=cand_value,
        )

        hands: list[tuple[int, ...]] = [_OPP_PLACEHOLDER] * 4
        hands[my_seat] = tuple(hand)

        return ZebGameState(
            hands=tuple(hands),
            dealer=0,
            phase=GamePhase.PLAYING,
            bid_state=bid_state,
            decl_id=int(cand_decl),
            bidder=int(my_seat),
            played=frozenset(),
            play_history=(),
            current_trick=(),
            trick_leader=int(my_seat),
            team_points=(0, 0),
        )

    # -- core: belief-weighted P(make) for every candidate --------------------

    def _pmake_table(
        self, hand: tuple[int, ...], ctx: BidContext, values: Sequence[int],
    ) -> dict[int, dict[int, float]]:
        """Return {decl: {value: P(make)}} for the candidate grid.

        ONE oracle forward over the distinct declarations (Q depends only on
        decl + bidder + my hand + worlds, not the bid value); the belief weights
        and the P(make) threshold then vary per (decl, value).
        """
        my_seat = ctx.seat
        bids = ctx.bids
        decls = self.decls

        # Cache hit for the whole grid?
        key = lambda d, v: (tuple(sorted(hand)), tuple(bids), int(d), int(v))
        if all(key(d, v) in self._pmake_cache for d in decls for v in values):
            return {d: {v: self._pmake_cache[key(d, v)] for v in values} for d in decls}

        # One hypothetical state per declaration (winner leads trick 1). The bid
        # value used for the *state* is irrelevant to the oracle Q; we use ctx's
        # high value as a representative so the state is well-formed.
        rep_value = max(values)
        states = [
            self.hypothetical_state(hand, bids, my_seat, d, rep_value) for d in decls
        ]
        n = len(states)
        self._ensure_capacity(n)

        gst = zeb_states_to_game_state_tensor(states, self.device)
        with torch.no_grad():
            worlds = sample_worlds_batched(gst, self._sampler, self.n_samples)
            deals = build_hypothetical_deals(gst, worlds)
            tokens, masks = tokenize_batched(gst, deals, self._tokenizer)
            q = query_model(
                self.oracle_model, tokens, masks, gst, self.n_samples, self.device,
            ).view(n, self.n_samples, 7)

            legal = gst.legal_actions()  # [n, 7] — all 7 leads legal at trick 1
            bidder = gst.bidder.long()
            cur = gst.current_player.long()

            table: dict[int, dict[int, float]] = {}
            ess_accum = []
            for value in values:
                # Belief weights depend on (decl, value): the auction feature reads
                # win_bid_norm. Re-stamp the bid value into each state for this value.
                if self.belief_model is not None:
                    val_states = [
                        self.hypothetical_state(hand, bids, my_seat, d, value)
                        for d in decls
                    ]
                else:
                    val_states = states  # unused for uniform weights
                w = belief_weights_for_worlds(
                    self.belief_model, self.is_voids, val_states, worlds, self.device,
                    uniform_mix=self.uniform_mix, tau=self.tau,
                )  # [n, M]
                if self.belief_model is not None:
                    ess_accum.append(float(effective_sample_size(w).mean().item()))

                e_q_pdf = compute_eq_pdf(q, weights=w)  # [n, 7, 85]
                bid_values = [int(value)] * n
                p_make = _p_make_from_pdf(e_q_pdf, bidder, cur, bid_values)  # [n, 7]

                # P(make) at the action the line play would actually take: the
                # oracle-argmax legal lead. Restrict to LEGAL leads (mask illegal
                # to -inf) so we never optimistically read an unplayable slot.
                masked = p_make.clone()
                masked[~legal] = float("-inf")
                lead_idx = masked.argmax(dim=1)  # [n]
                p_at_lead = p_make.gather(1, lead_idx.view(n, 1)).squeeze(1)  # [n]

                for i, d in enumerate(decls):
                    pv = float(p_at_lead[i].item())
                    table.setdefault(d, {})[value] = pv
                    self._pmake_cache[key(d, value)] = pv

        if ess_accum:
            self.last_ess = sum(ess_accum) / len(ess_accum)
        return table

    # -- BidPolicy interface --------------------------------------------------

    def _utility_of(
        self, value: int, table: Mapping[int, Mapping[int, float]], ctx: BidContext,
    ) -> float:
        """Marks utility of bidding ``value`` under the best declaration's P(make).

        Applies the optimism-correction scale at utility time (the cache keeps the
        raw double-dummy P(make) so it stays reusable across scales)."""
        p = max(table[d][value] for d in table)
        p = min(1.0, max(0.0, p * self.pmake_scale))
        return self.utility.value(
            p, value, team=ctx.team, marks=ctx.marks, marks_to_win=ctx.marks_to_win,
        )

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        if not self._worth_evaluating(ctx.hand):
            return PASS
        values = list(ctx.legal)
        if not values:
            return PASS
        table = self._pmake_table(ctx.hand, ctx, values)

        if self.maximize:
            # Rung #31 structure (mirror GusBidder.bid maximize): utility-MAX legal
            # bid; ties keep the cheapest; PASS if none clears the margin.
            best_val, best_u = PASS, self.margin
            for value in values:
                u = self._utility_of(value, table, ctx)
                if u > best_u:
                    best_u, best_val = u, value
            return best_val
        # Minimum positive-utility bid (cheapest that clears the margin).
        for value in values:
            if self._utility_of(value, table, ctx) > self.margin:
                return value
        return PASS

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        """The argmax-P(make) declaration at the won value.

        Reuses the P(make) cache populated by ``bid``. ``bid`` here is the value
        actually won; if for some reason it is uncached (e.g. a forced open) we
        re-evaluate the single-value grid first.
        """
        # The threshold a contract demands is contract_points(bid), but the cache
        # is keyed by the bid value itself, so look up the won value directly.
        sorted_hand = tuple(sorted(hand))
        # Recover the bids tuple from any cached entry for this hand, else assume
        # a fresh auction (declare can be called without a matching bid() only via
        # the forced-open path, which the arena does not use with this bidder).
        candidates = {
            (k[2], k[3]): v for k, v in self._pmake_cache.items()
            if k[0] == sorted_hand and k[3] == int(bid)
        }
        if not candidates:
            # Cold path: evaluate the single won value with an empty auction.
            ctx = BidContext(
                hand=tuple(hand), seat=0, dealer=0, bids=(-1, -1, -1, -1),
                high_bid=0, high_seat=-1, legal=(int(bid),),
            )
            table = self._pmake_table(tuple(hand), ctx, [int(bid)])
            return max(self.decls, key=lambda d: table[d][int(bid)])
        return max(self.decls, key=lambda d: candidates[(d, int(bid))])

    def __repr__(self) -> str:
        tag = "uniform" if self.belief_model is None else f"belief(tau={self.tau})"
        mode = "max" if self.maximize else "min"
        return (
            f"BeliefBidder(weights={tag}, mode={mode}, utility={self.utility!r}, "
            f"n_samples={self.n_samples}, uniform_mix={self.uniform_mix}, "
            f"margin={self.margin}, pmake_scale={self.pmake_scale})"
        )

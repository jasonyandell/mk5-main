"""Tests for the belief-conditioned bidder (rung #26 keystone).

Four tests, cheapest insurance first:

1. (no GPU) Feature correctness of the hypothetical completed-auction state.
2. (GPU)   Pure-oracle degrade: belief_model=None -> uniform weights -> P(make)
           matches the plain-oracle (uniform e_q_pdf) P(make).
3. (GPU)   The loop CAN move: a real belief adapter shifts at least one bid away
           from the hand-only (net:wp) bidder. A no-op would be a CRITICAL find.
4. (GPU)   Sanity: strong hands bid, trash passes; every bid is legal or PASS.

GPU tests are skipped when neither MPS nor CUDA is available.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from arena.auction import PASS, BidContext, legal_bids
from champion.belief_bidder import BeliefBidder
from forge.bidding.schema import EVAL_DECLS
from forge.oracle.rng import deal_from_seed
from gus.model.auction import N_AUCTION_FEATURES, auction_feature_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BELIEF_ADAPTER = PROJECT_ROOT / "scratch/champion-run/run24_measure/v2_auction_s0.pt"
DEFAULT_ORACLE = (
    PROJECT_ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
)

_NAMES = [f"{a}-{b}" for a in range(7) for b in range(a + 1)]


def _did(name: str) -> int:
    a, b = (int(x) for x in name.split("-"))
    return _NAMES.index(f"{max(a, b)}-{min(a, b)}")


# A strong fives hand (passes the prefilter); a hopeless trash hand (prefiltered).
STRONG_HAND = tuple(_did(n) for n in ("5-5", "5-4", "5-2", "5-0", "6-4", "3-2", "1-0"))
TRASH_HAND = tuple(_did(n) for n in ("6-5", "6-4", "3-2", "3-1", "2-1", "5-0", "4-0"))


def _device() -> str | None:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _ctx(hand, *, bids=(-1, -1, -1, -1), high_bid=0, high_seat=-1, marks=(0, 0)):
    return BidContext(
        hand=hand, seat=0, dealer=3, bids=bids,
        high_bid=high_bid, high_seat=high_seat, legal=legal_bids(high_bid),
        marks=marks, marks_to_win=7,
    )


# --------------------------------------------------------------------------- #
# Test 1: feature correctness (no GPU, no models)                             #
# --------------------------------------------------------------------------- #

def _bidder_no_models() -> BeliefBidder:
    """A BeliefBidder we can call hypothetical_state on without loading models."""
    return BeliefBidder.__new__(BeliefBidder)


def test_hypothetical_state_auction_feature():
    """The hypothetical state's auction feature is shape (28,), encodes
    is_winner=1 at relative seat 0, and the decl one-hot for cand_decl."""
    b = _bidder_no_models()
    my_seat = 0
    cand_decl = 4  # fours
    cand_value = 35
    # An earlier real bid: seat 3 bid 32 before me; seats 1,2 speak after me.
    bids = (-1, -1, -1, 32)
    st = BeliefBidder.hypothetical_state(
        b, STRONG_HAND, bids, my_seat, cand_decl, cand_value,
    )

    # State invariants (winner leads trick 1).
    from forge.zeb.game import current_player
    from forge.zeb.types import GamePhase

    assert st.phase == GamePhase.PLAYING
    assert st.play_history == ()
    assert st.current_trick == ()
    assert st.played == frozenset()
    assert st.trick_leader == my_seat
    assert st.bidder == my_seat
    assert st.decl_id == cand_decl
    assert st.bid_state.high_bidder == my_seat
    assert st.bid_state.high_bid == cand_value
    assert st.bid_state.bids[my_seat] == cand_value
    assert st.bid_state.bids[3] == 32  # earlier real bid kept at its seat
    assert current_player(st) == my_seat  # empty trick -> leader leads

    # Auction feature, current-player POV (= my_seat).
    cp = current_player(st)
    feat = auction_feature_vector(
        st.bid_state.bids, st.bid_state.high_bidder, st.bid_state.high_bid,
        st.decl_id, cp,
    )
    assert feat.shape == (N_AUCTION_FEATURES,)
    assert N_AUCTION_FEATURES == 28
    # Relative seat 0 = me; is_winner flag lives at index 4*0 + 3 = 3.
    assert float(feat[3]) == 1.0, "is_winner must be 1 at relative seat 0"
    # No other seat is the winner.
    for r in range(1, 4):
        assert float(feat[4 * r + 3]) == 0.0
    # decl one-hot at 18 + cand_decl.
    assert float(feat[18 + cand_decl]) == 1.0
    assert float(feat[18:28].sum()) == 1.0  # exactly one decl set
    # My bid level (35) is normalized > 0 at relative seat 0, slot 0.
    assert float(feat[0]) > 0.0
    # win_bid_norm (global tail [16]) > 0 for a 35 contract.
    assert float(feat[16]) > 0.0


def test_hypothetical_state_only_suppose_i_win():
    """Across all eval decls, the bidding seat is always the winner (rel seat 0)."""
    b = _bidder_no_models()
    from forge.zeb.game import current_player

    for my_seat in range(4):
        for decl in EVAL_DECLS:
            st = BeliefBidder.hypothetical_state(
                b, STRONG_HAND, (-1, -1, -1, -1), my_seat, decl, 30,
            )
            assert st.bidder == my_seat
            cp = current_player(st)
            assert cp == my_seat
            feat = auction_feature_vector(
                st.bid_state.bids, st.bid_state.high_bidder,
                st.bid_state.high_bid, st.decl_id, cp,
            )
            assert float(feat[3]) == 1.0  # I am the winner at rel seat 0


# --------------------------------------------------------------------------- #
# GPU fixtures: load the oracle once                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def oracle_and_device():
    dev = _device()
    if dev is None:
        pytest.skip("no MPS/CUDA device")
    if not DEFAULT_ORACLE.exists():
        pytest.skip(f"oracle missing: {DEFAULT_ORACLE}")
    from forge.zeb.eval.loading import load_oracle

    model = load_oracle(str(DEFAULT_ORACLE), dev)
    return model, dev


# --------------------------------------------------------------------------- #
# Test 2: pure-oracle degrade matches plain-oracle P(make)                    #
# --------------------------------------------------------------------------- #

def test_pure_oracle_degrade_matches_plain_oracle(oracle_and_device):
    """belief_model=None -> uniform world weights -> the bidder's P(make) for a
    contract equals the plain-oracle (uniform e_q_pdf) P(make) at the same lead."""
    model, dev = oracle_and_device
    torch.manual_seed(0)

    bidder = BeliefBidder(
        belief_model=None, oracle_model=model, device=dev, n_samples=32,
    )
    hand = STRONG_HAND
    ctx = _ctx(hand)
    table = bidder._pmake_table(hand, ctx, [30])
    bidder_pmake = max(table[d][30] for d in table)

    # Independently recompute the plain-oracle P(make) for the SAME states with
    # uniform weights (no belief), reusing the same seeded sampler.
    from forge.eq.generate.actions import _p_make_from_pdf
    from forge.eq.generate.deals import build_hypothetical_deals
    from forge.eq.generate.eq_compute import compute_eq_pdf
    from forge.eq.generate.model import query_model
    from forge.eq.generate.sampling import sample_worlds_batched
    from forge.eq.generate.tokenization import tokenize_batched
    from forge.zeb.eq_player import zeb_states_to_game_state_tensor

    decls = bidder.decls
    states = [bidder.hypothetical_state(hand, ctx.bids, 0, d, 30) for d in decls]
    n = len(states)
    bidder._ensure_capacity(n)
    torch.manual_seed(0)  # same sampler draw as the bidder used (seed reset)
    # NOTE: the bidder consumed RNG; recompute end-to-end with a fresh draw and
    # compare the *uniform-vs-uniform* path, which is deterministic given worlds.
    gst = zeb_states_to_game_state_tensor(states, dev)
    with torch.no_grad():
        worlds = sample_worlds_batched(gst, bidder._sampler, bidder.n_samples)
        deals = build_hypothetical_deals(gst, worlds)
        tokens, masks = tokenize_batched(gst, deals, bidder._tokenizer)
        q = query_model(model, tokens, masks, gst, bidder.n_samples, dev).view(
            n, bidder.n_samples, 7
        )
        e_q_pdf_uniform = compute_eq_pdf(q)  # weights=None -> uniform 1/M
        legal = gst.legal_actions()
        p_make = _p_make_from_pdf(e_q_pdf_uniform, gst.bidder.long(),
                                  gst.current_player.long(), [30] * n)
        masked = p_make.clone()
        masked[~legal] = float("-inf")
        lead = masked.argmax(dim=1)
        p_at_lead = p_make.gather(1, lead.view(n, 1)).squeeze(1)
    plain_pmake = float(p_at_lead.max().item())

    # The two paths both use uniform weights; they should agree closely. Worlds
    # are re-sampled (MRV is stochastic), so allow Monte-Carlo tolerance.
    assert abs(bidder_pmake - plain_pmake) < 0.10, (
        f"degrade path {bidder_pmake:.3f} != plain oracle {plain_pmake:.3f}"
    )


def test_pure_oracle_uses_uniform_weights(oracle_and_device):
    """Direct check: with belief_model=None the weight tensor is exactly 1/M."""
    from champion.belief import belief_weights_for_worlds

    model, dev = oracle_and_device
    bidder = BeliefBidder(belief_model=None, oracle_model=model, device=dev, n_samples=16)
    states = [bidder.hypothetical_state(STRONG_HAND, (-1, -1, -1, -1), 0, 5, 30)]
    bidder._ensure_capacity(1)
    from forge.eq.generate.sampling import sample_worlds_batched
    from forge.zeb.eq_player import zeb_states_to_game_state_tensor

    gst = zeb_states_to_game_state_tensor(states, dev)
    with torch.no_grad():
        worlds = sample_worlds_batched(gst, bidder._sampler, bidder.n_samples)
    w = belief_weights_for_worlds(None, True, states, worlds, dev)
    assert torch.allclose(w, torch.full_like(w, 1.0 / bidder.n_samples))


# --------------------------------------------------------------------------- #
# Test 3: the loop CAN move (belief-conditioned bid != hand-only bid)         #
# --------------------------------------------------------------------------- #

def test_belief_bidder_differs_from_hand_only(oracle_and_device):
    """A real belief adapter must shift at least one bid vs the hand-only net:wp
    bidder. If it NEVER differs over 20 hands, the self-play loop is a no-op
    (CRITICAL) — surface that explicitly."""
    model, dev = oracle_and_device
    if not BELIEF_ADAPTER.exists():
        pytest.skip(f"belief adapter missing: {BELIEF_ADAPTER}")

    from champion.belief import load_belief
    from champion.bidder import GusBidder, NetPointsEvaluator
    from champion.utility import MarksToSeven

    belief_model, is_voids = load_belief(str(BELIEF_ADAPTER), dev)
    belief_bidder = BeliefBidder(
        belief_model, model, is_voids=is_voids, device=dev, n_samples=32,
        utility=MarksToSeven(), maximize=True,
    )
    hand_only = GusBidder(
        pmake_fn=NetPointsEvaluator(), utility=MarksToSeven(), maximize=True,
    )

    rng = random.Random(0)
    n_diff = 0
    n_eval = 0
    diffs = []
    for seed in range(20):
        hand = tuple(deal_from_seed(seed)[0])
        # Inject a realistic earlier auction so the belief feature has signal:
        # seat 3 (right of me) bid 31, seats 1,2 not yet / passed.
        ctx = _ctx(hand, bids=(-1, -1, -1, 31), high_bid=31, high_seat=3)
        b_belief = belief_bidder.bid(ctx, rng)
        b_hand = hand_only.bid(ctx, rng)
        # Only count hands where at least one bidder acts (else trivially equal).
        if b_belief == PASS and b_hand == PASS:
            continue
        n_eval += 1
        if b_belief != b_hand:
            n_diff += 1
            diffs.append((seed, b_belief, b_hand))

    assert n_eval > 0, "no hand triggered a bid for either bidder"
    # CRITICAL if zero: the belief never changes the contract.
    assert n_diff > 0, (
        f"CRITICAL: belief-conditioned bid identical to hand-only over "
        f"{n_eval} acting hands — the self-play loop would be a no-op. "
        f"belief ess={belief_bidder.last_ess}"
    )
    print(f"belief differs on {n_diff}/{n_eval} acting hands; examples {diffs[:5]}; "
          f"ess={belief_bidder.last_ess}")


# --------------------------------------------------------------------------- #
# Test 4: sanity — strong bids, trash passes, bids always legal               #
# --------------------------------------------------------------------------- #

def test_strong_bids_trash_passes(oracle_and_device):
    model, dev = oracle_and_device
    if not BELIEF_ADAPTER.exists():
        pytest.skip(f"belief adapter missing: {BELIEF_ADAPTER}")
    from champion.belief import load_belief

    belief_model, is_voids = load_belief(str(BELIEF_ADAPTER), dev)
    bidder = BeliefBidder(
        belief_model, model, is_voids=is_voids, device=dev, n_samples=32,
    )
    rng = random.Random(0)

    # Trash is prefiltered -> PASS without touching the oracle.
    assert bidder.bid(_ctx(TRASH_HAND), rng) == PASS

    # Strong fives hand should bid (positive utility somewhere).
    b_strong = bidder.bid(_ctx(STRONG_HAND), rng)
    assert b_strong != PASS, "a strong fives hand should bid"
    assert b_strong in _ctx(STRONG_HAND).legal


def test_all_bids_legal_or_pass(oracle_and_device):
    model, dev = oracle_and_device
    if not BELIEF_ADAPTER.exists():
        pytest.skip(f"belief adapter missing: {BELIEF_ADAPTER}")
    from champion.belief import load_belief

    belief_model, is_voids = load_belief(str(BELIEF_ADAPTER), dev)
    bidder = BeliefBidder(
        belief_model, model, is_voids=is_voids, device=dev, n_samples=16,
    )
    rng = random.Random(0)
    for seed in range(8):
        hand = tuple(deal_from_seed(seed)[0])
        ctx = _ctx(hand)
        b = bidder.bid(ctx, rng)
        assert b == PASS or b in ctx.legal
        if b != PASS:
            assert bidder.declare(hand, b, rng) in EVAL_DECLS

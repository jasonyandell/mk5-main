"""JudNet (jud v1) — unified realized-value organ. CPU, seconds.

The load-bearing tests are the two identities:

  1. TRAIN/SERVE byte-identity — every info-state of a real hand featurizes
     byte-identically from the snapshot corpus (`featurize_snapshot` at a play
     prefix) and from the live engine state the arena serves (`featurize_state`
     mid-replay). If this drifts, the head is served out of distribution and
     every price is wrong.
  2. BID-TIME identity — the jud path with an empty play history reproduces
     `margin_net.featurize` exactly on the shared 91 dims: bid-time is
     play-time with no plays, one featurization, no special case.

The rest pins the play-block encoding on a hand-built trick, the dataset
expansion/dedup, a training smoke, and the ValueBidder compatibility of
`pmake_table`.
"""
from __future__ import annotations

import json

import torch

from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.match import run_match, snapshot_rows
from arena.play import RandomPlay
from champion import margin_net
from champion.jud_net import (
    AUCTION_DIM,
    FEATURE_DIM,
    GLOBAL_DIM,
    HAND_DIM,
    JudDataset,
    JudNet,
    N_POINTS,
    PER_DOMINO,
    PLAY_DIM,
    featurize,
    featurize_snapshot,
    featurize_state,
    hand_samples,
)
from forge.zeb.game import apply_action, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState


# --------------------------------------------------------------------- #
#  Fixtures                                                               #
# --------------------------------------------------------------------- #

def _tiny_result(n_games=2, base_seed=7):
    return run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=n_games, cfg=ArenaConfig(marks_to_win=2, base_seed=base_seed),
    )


def _initial_state(snap) -> ZebGameState:
    """Rebuild the hand's opening PLAYING state exactly as arena.engine does."""
    return ZebGameState(
        hands=tuple(tuple(h) for h in snap["hands"]),
        dealer=snap["dealer"],
        phase=GamePhase.PLAYING,
        bid_state=BidState(
            bids=tuple(snap["bids"]),
            high_bidder=snap["bidder"],
            high_bid=snap["bid_value"],
        ),
        decl_id=snap["decl_id"],
        bidder=snap["bidder"],
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=snap["bidder"],
        team_points=(0, 0),
    )


# --------------------------------------------------------------------- #
#  1. Train/serve byte-identity over a real match                         #
# --------------------------------------------------------------------- #

def test_featurize_state_matches_snapshot_prefix_over_real_match():
    """Replay every snapshot hand through the zeb engine; at every play step
    the mover's PRE-move state (decision) and POST-move state (the child
    JudPlay prices) must featurize byte-identically to the snapshot-prefix
    path the dataset trains on."""
    snaps = snapshot_rows(_tiny_result(n_games=2))
    assert snaps
    for snap in snaps:
        state = _initial_state(snap)
        for k, (seat, domino) in enumerate(snap["plays"]):
            x_train = featurize_snapshot(snap, step=k, seat=seat)
            x_serve = featurize_state(state)  # mover's POV, pre-move
            assert torch.equal(x_train, x_serve), (snap["seed"], k, "pre")

            slot = snap["hands"][seat].index(domino)
            state = apply_action(state, slot)
            x_train = featurize_snapshot(snap, step=k + 1, seat=seat)
            x_serve = featurize_state(state, seat=seat)  # post-move, same POV
            assert torch.equal(x_train, x_serve), (snap["seed"], k, "post")
        assert is_terminal(state)
        assert state.team_points[snap["bidder"] % 2] == snap["bidder_team_pts"]


# --------------------------------------------------------------------- #
#  2. Bid-time identity with margin_net (empty history, seat = bidder)    #
# --------------------------------------------------------------------- #

def test_empty_history_reproduces_margin_net_featurization():
    snaps = snapshot_rows(_tiny_result(n_games=2, base_seed=13))
    assert snaps
    for snap in snaps:
        x_jud = featurize_snapshot(snap, step=0)  # POV defaults to the bidder
        x_margin = margin_net.featurize_snapshot(snap)
        assert torch.equal(x_jud[: HAND_DIM + AUCTION_DIM], x_margin)
        # Root play block: all zeros except the empty-trick fill flag.
        play = x_jud[HAND_DIM + AUCTION_DIM:]
        assert play.shape == (PLAY_DIM,)
        fill0 = HAND_DIM + AUCTION_DIM + PLAY_DIM - GLOBAL_DIM + 3
        assert float(x_jud[fill0]) == 1.0
        assert float(play.sum()) == 1.0


def test_featurization_is_blind_to_winning_bid_level():
    # The v0 leakage defense survives v1: same info-state, two winning levels.
    hand = (0, 1, 2, 3, 4, 5, 6)
    plays = [(3, 21), (0, 0), (1, 7), (2, 14)]
    x_lo = featurize(hand, [0, 0, 0, 30], 3, 0, 5, plays, seat=0)
    x_hi = featurize(hand, [0, 0, 0, 42], 3, 0, 5, plays, seat=0)
    assert torch.equal(x_lo, x_hi)
    assert x_lo.shape == (FEATURE_DIM,)


# --------------------------------------------------------------------- #
#  3. Play-block encoding on a hand-built trick                           #
# --------------------------------------------------------------------- #

def test_play_block_encodes_seat_position_trick_and_points():
    # decl_id 6 (sixes trump). Trick: seat 3 leads 6-6 (id 27), others follow
    # with 6-3 (24), 6-1 (22), 6-4 (25). 6-6 wins; count = 6-4 (10 pts) + trick
    # point = 11 for team 3%2=1, which is the DECLARING team (bidder=3).
    plays = [(3, 27), (0, 24), (1, 22), (2, 25)]
    x = featurize(
        (0, 1, 2, 3, 4, 5, 6), [0, 0, 0, 30], bidder=3, dealer=0, decl_id=6,
        plays=plays, seat=0,
    )
    play = x[HAND_DIM + AUCTION_DIM:]

    for k, (seat, domino) in enumerate(plays):
        block = play[PER_DOMINO * domino: PER_DOMINO * (domino + 1)]
        rel = (seat - 0) % 4  # POV seat 0
        assert float(block[rel]) == 1.0 and float(block[:4].sum()) == 1.0
        assert float(block[4 + k % 4]) == 1.0 and float(block[4:8].sum()) == 1.0
        assert float(block[8]) == 0.0  # trick index 0

    unplayed = play[: PER_DOMINO * 22]  # ids 0..21 untouched
    assert float(unplayed.abs().sum()) == 0.0

    g = 28 * PER_DOMINO
    assert abs(float(play[g + 0]) - 11 / 42) < 1e-6  # declaring team scored 11
    assert float(play[g + 1]) == 0.0
    assert abs(float(play[g + 2]) - 4 / 28) < 1e-6
    assert float(play[g + 3 + 0]) == 1.0  # trick complete → fill 0


def test_play_block_is_pov_relative():
    plays = [(3, 27), (0, 24), (1, 22), (2, 25)]
    args = ((0, 1, 2, 3, 4, 5, 6), [0, 0, 0, 30], 3, 0, 6, plays)
    x0 = featurize(*args, seat=0)
    x2 = featurize(*args, seat=2)
    b0 = x0[HAND_DIM + AUCTION_DIM + PER_DOMINO * 27:][:4]
    b2 = x2[HAND_DIM + AUCTION_DIM + PER_DOMINO * 27:][:4]
    assert float(b0[3]) == 1.0  # seat 3 is my right opponent from seat 0
    assert float(b2[1]) == 1.0  # ... and my left opponent from seat 2


# --------------------------------------------------------------------- #
#  4. Dataset: expansion, dedup, splits, labels                           #
# --------------------------------------------------------------------- #

def test_hand_samples_covers_decisions_and_children_incl_bid_root():
    snap = {"plays": [[(k * 3) % 4, k] for k in range(28)]}
    coords = list(hand_samples(snap))
    assert len(coords) == 56
    movers = {k: p for k, (p, _) in enumerate(snap["plays"])}
    assert set(coords) == {(k, movers[k]) for k in movers} | {
        (k + 1, movers[k]) for k in movers
    }


def test_dataset_from_tiny_match(tmp_path):
    result = _tiny_result(n_games=2)
    snaps = snapshot_rows(result)
    path = tmp_path / "snaps.json"
    path.write_text(json.dumps({"snapshots": snaps, "metadata": {}}))

    ds = JudDataset(path, split="all")
    # Expansion: ≤ 56 rows per snapshot hand; paired halves with identical
    # auctions collapse their step-0 rows, so dedup strictly shrinks the set.
    assert 0 < len(ds) < 56 * len(snaps)
    for x, y in ds:
        assert x.shape == (FEATURE_DIM,)
        assert 0 <= int(y) <= 42

    # Splits partition the deduped rows, whole hands staying together.
    tr = JudDataset(path, split="train")
    va = JudDataset(path, split="val")
    te = JudDataset(path, split="test")
    assert len(tr) + len(va) + len(te) == len(ds)
    assert set(tr.keys) | set(va.keys) | set(te.keys) == set(ds.keys)

    # Rows keyed identically are unique. (Note: paired halves replay the same
    # (seed, hand_idx) with divergent random play, so a DEAL may appear with
    # two different realized labels — two hand instances, two Monte Carlo
    # draws. Only byte-identical info-states dedup, matching MarginDataset.)
    assert len(set(ds.keys)) == len(ds)


def test_dataset_rejects_bid_only_corpus(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"snapshots": [
        {"seed": 1, "hand_idx": 0, "bids": [0, 30, 0, 0], "bidder": 1,
         "dealer": 0, "decl_id": 3, "hands": [[0]] * 4, "bidder_team_pts": 20}
    ]}))
    try:
        JudDataset(path)
        raise AssertionError("expected ValueError for corpus without plays")
    except ValueError as e:
        assert "plays" in str(e)


# --------------------------------------------------------------------- #
#  5. Training smoke + ValueBidder compatibility                          #
# --------------------------------------------------------------------- #

def test_training_smoke_ce_decreases(tmp_path):
    snaps = snapshot_rows(_tiny_result(n_games=2))
    path = tmp_path / "snaps.json"
    path.write_text(json.dumps({"snapshots": snaps}))
    ds = JudDataset(path, split="all")
    xs = torch.stack([x for x, _ in ds])
    ys = torch.stack([y for _, y in ds])

    torch.manual_seed(0)
    model = JudNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    first = last = None
    for step in range(30):
        loss = loss_fn(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss.item()) if step == 0 else first
        last = float(loss.item())
    assert last < first, f"CE did not decrease: {first:.4f} → {last:.4f}"


def test_pmake_table_matches_margin_interface_and_bounds():
    from forge.bidding.schema import EVAL_DECLS

    model = JudNet()
    table = model.pmake_table(
        hand=(0, 1, 2, 3, 4, 5, 6), bids=[0, 0, 0, 30], bidder=3, dealer=0,
    )
    assert set(table) == set(EVAL_DECLS)
    for row in table.values():
        assert set(row) == set(range(30, 43))
        vals = [row[t] for t in range(30, 43)]
        assert all(0.0 <= p <= 1.0 for p in vals)
        assert all(a >= b - 1e-6 for a, b in zip(vals, vals[1:]))


def test_value_bidder_consumes_jud_net():
    import random

    from arena.auction import PASS, BidContext, legal_bids
    from champion.value_bidder import ValueBidder

    bidder = ValueBidder(JudNet())
    ctx = BidContext(
        hand=(0, 7, 14, 21, 2, 9, 16), seat=1, dealer=0, bids=(-1, -1, -1, -1),
        high_bid=0, high_seat=-1, legal=legal_bids(0),
    )
    value = bidder.bid(ctx, random.Random(0))
    assert value == PASS or value in ctx.legal
    if value != PASS:
        assert bidder.declare(ctx.hand, value, random.Random(0)) in EVAL_DECLS_SET


EVAL_DECLS_SET = {0, 1, 2, 3, 4, 5, 6, 7, 9}


def test_jud_net_forward_shape():
    model = JudNet()
    x = torch.zeros(5, FEATURE_DIM)
    assert model(x).shape == (5, N_POINTS)

"""Tests for the otis tied-strategy rollout (issue #49, P7).

Three properties, per the task:

  (a) tied-by-construction — the info-honest actor's action is a pure function of
      its own info-state, so two worlds identical from the actor's view up to a
      divergence ply produce identical actor actions up to that ply. Tested both
      directly (π_me purity) and end-to-end (the first divergence ply of two
      co-rolled worlds is never an actor ply).
  (b) rollout legality — every rolled world produces 28 legal plays whose P1
      identity holds (verified through the fate parser).
  (c) determinism — same seed ⇒ identical world selection and identical outcomes.

The clairvoyant leg and the retention/fusion-gap machinery are exercised too.

CPU only; the student + oracle load in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from otis import tiedroll as TR
from otis.fates import COUNT_TILE_PIPS, NeutralGame, parse_game_fates

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CORPUS = REPO_ROOT / "gus" / "data" / "corpus_v2_eval.pt"
STUDENT = REPO_ROOT / "gus" / "adapters" / "v3_consistency_10000g.pt"
ORACLE = REPO_ROOT / "forge" / "models" / "domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"

NOTRUMP = 9


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tied_policy():
    if not STUDENT.exists():
        pytest.skip("gus student adapter not present")
    return TR.load_tied_policy(str(STUDENT), "cpu")


@pytest.fixture(scope="module")
def clair_policy():
    if not ORACLE.exists():
        pytest.skip("oracle checkpoint not present")
    return TR.load_clairvoyant_policy(str(ORACLE), "cpu", max_batch=8)


@pytest.fixture(scope="module")
def a_decision():
    """A trick-0/1 void-capable decision from the eval corpus (game, d_idx, P)."""
    if not EVAL_CORPUS.exists():
        pytest.skip("eval corpus not present")
    blob = torch.load(str(EVAL_CORPUS), map_location="cpu", weights_only=False)
    games = blob["results"]
    picks = TR._find_void_capable_decisions(games, 1, min_worlds=8)
    gi, d_idx, P = picks[0]
    return games[gi], d_idx, P


# --------------------------------------------------------------------------- #
# (a) tied-by-construction
# --------------------------------------------------------------------------- #


def _synthetic_pair(swap: bool):
    """Two full deals sharing P=0's hand; opponents differ by one swapped tile."""
    P = 0
    p_hand = [0, 1, 2, 3, 4, 5, 6]
    seat1 = [7, 8, 9, 10, 11, 12, 13]
    seat2 = [14, 15, 16, 17, 18, 19, 20]
    seat3 = [21, 22, 23, 24, 25, 26, 27]
    if swap:
        # move a followable low tile between two hidden opponents
        seat1 = [14, 8, 9, 10, 11, 12, 13]
        seat2 = [7, 15, 16, 17, 18, 19, 20]
    deal = [None, None, None, None]
    deal[P] = list(p_hand)
    deal[1] = seat1
    deal[2] = seat2
    deal[3] = seat3
    assert len({d for row in deal for d in row}) == 28
    return deal


def test_pi_me_is_info_state_pure(tied_policy):
    """π_me for the actor is identical across two worlds with the same actor
    info-state (same hand, same — here empty — public prefix)."""
    from forge.eq.game_tensor import GameStateTensor

    deal_a = _synthetic_pair(swap=False)
    deal_b = _synthetic_pair(swap=True)
    deals = [deal_a, deal_b]  # P=0 identical in both
    state = GameStateTensor.from_deals(deals, [NOTRUMP] * 2, device="cpu", bidders=[0, 0])
    cp = state.current_player.long()
    assert int(cp[0]) == 0 and int(cp[1]) == 0  # P=0 leads trick 0 in both

    scores = tied_policy.scores(state, deals, [[], []], cp, NOTRUMP)
    # The actor's info-state is identical ⇒ π_me logits identical row-for-row.
    assert torch.allclose(scores[0], scores[1], atol=1e-5)
    assert int(scores[0].argmax()) == int(scores[1].argmax())


def test_tied_first_divergence_is_never_the_actor(tied_policy):
    """Co-rolling two worlds that share the actor's hand, the FIRST play where the
    two trajectories differ is never the actor's — the actor is info-honest, so it
    cannot be the first to diverge while histories are still identical."""
    deals = [_synthetic_pair(swap=False), _synthetic_pair(swap=True)]
    P = 0
    trajs = TR.roll_worlds(deals, NOTRUMP, bidder=0, prefix=[], P=P, policy=tied_policy)
    assert len(trajs[0]) == 28 and len(trajs[1]) == 28

    # first ply at which the two public trajectories differ
    div = None
    for t in range(28):
        if trajs[0][t] != trajs[1][t]:
            div = t
            break
    if div is None:
        pytest.skip("the two synthetic worlds happened not to diverge under tied play")
    # up to div the trajectories are identical, so the seat at div is well-defined
    seat_at_div = trajs[0][div][0]
    assert seat_at_div != P, (
        f"actor {P} diverged first at ply {div} despite an identical info-state — "
        "tied-by-construction violated"
    )


def test_tied_property_on_real_worlds(tied_policy, a_decision):
    """Same property on two real corpus worlds (guards against synthetic bias)."""
    game, d_idx, P = a_decision
    idx, _, _ = TR.valid_world_indices(game, d_idx, 2, seed=0)
    wh = game.decisions[d_idx].world_hands[idx]
    deals, prefix, P2, decl_id, bidder, M = TR.reconstruct_initial_deals(game, d_idx, wh)
    assert P2 == P and M == 2
    trajs = TR.roll_worlds(deals, decl_id, bidder, prefix, P, tied_policy)
    div = next((t for t in range(28) if trajs[0][t] != trajs[1][t]), None)
    if div is not None:
        assert trajs[0][div][0] != P


# --------------------------------------------------------------------------- #
# (b) rollout legality — 28 plays, identity holds via the fate parser
# --------------------------------------------------------------------------- #


def _assert_legal_completed(deals, decl_id, bidder, P, trajs):
    for m, plays in enumerate(trajs):
        assert len(plays) == 28
        ng = NeutralGame(
            game_id=f"t:{m}", hands=deals[m], decl_id=decl_id,
            bidder=bidder, bid_value=42, plays=plays,
        )
        gf = parse_game_fates(ng)  # asserts the P1 identity internally
        assert gf.team0_points + gf.team1_points == 42
        assert gf.team0_tricks + gf.team1_tricks == 7
        assert gf.team0_count + gf.team1_count == 35
        assert {t.tile for t in gf.tiles} == set(COUNT_TILE_PIPS)


def test_tied_rollout_legality(tied_policy, a_decision):
    game, d_idx, P = a_decision
    idx, _, _ = TR.valid_world_indices(game, d_idx, 6, seed=0)
    wh = game.decisions[d_idx].world_hands[idx]
    deals, prefix, P2, decl_id, bidder, M = TR.reconstruct_initial_deals(game, d_idx, wh)
    trajs = TR.roll_worlds(deals, decl_id, bidder, prefix, P2, tied_policy)
    _assert_legal_completed(deals, decl_id, bidder, P2, trajs)


def test_clairvoyant_rollout_legality(clair_policy, a_decision):
    game, d_idx, P = a_decision
    idx, _, _ = TR.valid_world_indices(game, d_idx, 6, seed=0)
    wh = game.decisions[d_idx].world_hands[idx]
    deals, prefix, P2, decl_id, bidder, M = TR.reconstruct_initial_deals(game, d_idx, wh)
    trajs = TR.roll_worlds(deals, decl_id, bidder, prefix, P2, clair_policy)
    _assert_legal_completed(deals, decl_id, bidder, P2, trajs)


def test_retention_commitment_preserves_legality(tied_policy, a_decision):
    """A retention constraint still yields 28 legal plays with the identity intact."""
    game, d_idx, P = a_decision
    idx, _, _ = TR.valid_world_indices(game, d_idx, 6, seed=0)
    wh = game.decisions[d_idx].world_hands[idx]
    deals, prefix, P2, decl_id, bidder, M = TR.reconstruct_initial_deals(game, d_idx, wh)
    base = TR.roll_worlds(deals, decl_id, bidder, prefix, P2, tied_policy)
    # profile a sloughable junk tile to constrain
    from collections import Counter

    from forge.eq.game_tensor import GameStateTensor

    ct: Counter = Counter()
    for m, plays in enumerate(base):
        st = GameStateTensor.from_deals([deals[m]], [decl_id], device="cpu", bidders=[bidder])
        for (seat, dom) in plays:
            slot = deals[m][seat].index(dom)
            if seat == P2 and TR._actor_is_void(st, 0, P2, decl_id) and not TR._is_trump(dom, decl_id):
                ct[dom] += 1
            st = st.apply_actions(torch.tensor([slot], dtype=torch.long))
    if len(ct) < 2:
        pytest.skip("decision has < 2 sloughable junk tiles to constrain")
    keep, release = [d for d, _ in ct.most_common()][:2]
    commit = TR.Commitment("k", keep_id=keep, release_id=release)
    trajs = TR.roll_worlds(deals, decl_id, bidder, prefix, P2, tied_policy, commit)
    _assert_legal_completed(deals, decl_id, bidder, P2, trajs)


# --------------------------------------------------------------------------- #
# (c) determinism given a seed
# --------------------------------------------------------------------------- #


def test_world_selection_deterministic(a_decision):
    game, d_idx, P = a_decision
    a, na, sa = TR.valid_world_indices(game, d_idx, 8, seed=3)
    b, nb, sb = TR.valid_world_indices(game, d_idx, 8, seed=3)
    assert a == b and na == nb and sa == sb
    c, _, _ = TR.valid_world_indices(game, d_idx, 8, seed=4)
    # a different seed generally yields a different subset
    assert a != c or len(a) < 8


def test_rollout_deterministic(tied_policy, a_decision):
    game, d_idx, P = a_decision
    idx, _, _ = TR.valid_world_indices(game, d_idx, 6, seed=1)
    wh = game.decisions[d_idx].world_hands[idx]
    deals, prefix, P2, decl_id, bidder, M = TR.reconstruct_initial_deals(game, d_idx, wh)
    t1 = TR.roll_worlds(deals, decl_id, bidder, prefix, P2, tied_policy)
    t2 = TR.roll_worlds(deals, decl_id, bidder, prefix, P2, tied_policy)
    assert t1 == t2


def test_pricing_is_paired_and_finite(tied_policy, clair_policy, a_decision):
    """price_retention returns finite deltas; clairvoyant delta of a self-pair
    (keep == release swapped) is antisymmetric under swapping keep/release."""
    game, d_idx, P = a_decision
    idx, _, _ = TR.valid_world_indices(game, d_idx, 8, seed=0)
    wh = game.decisions[d_idx].world_hands[idx]
    w_belief, ess = TR.belief_weights_for_decision(
        tied_policy.student, tied_policy.is_voids, game, d_idx, wh
    )
    base = TR.run_leg(game, d_idx, tied_policy, TR.NONE, idx, w_belief, ess)
    profile = TR.slough_profile(base)
    tiles = [d for d, _ in profile.most_common()][:2]
    if len(tiles) < 2:
        pytest.skip("decision has < 2 sloughable junk tiles to price")
    keep, release = tiles
    p1 = TR.price_retention(game, d_idx, keep, release, tied_policy, clair_policy, idx, w_belief, ess)
    p2 = TR.price_retention(game, d_idx, release, keep, tied_policy, clair_policy, idx, w_belief, ess)
    import math

    for v in (p1.tied_delta, p1.clairvoyant_delta, p1.fusion_gap):
        assert math.isfinite(v)
    # swapping keep/release negates every delta (paired, common random worlds)
    assert p1.tied_delta == pytest.approx(-p2.tied_delta, abs=1e-6)
    assert p1.clairvoyant_delta == pytest.approx(-p2.clairvoyant_delta, abs=1e-6)
    assert p1.fusion_gap == pytest.approx(-p2.fusion_gap, abs=1e-6)


def test_fate_class_encoding():
    """The 8-class fate index round-trips through the name helper."""
    names = [TR.fate_class_name(i) for i in range(TR.N_FATE_CLASSES)]
    assert names[0] == "my_team/led"
    assert names[7] == "their_team/sloughed"
    assert len(set(names)) == 8

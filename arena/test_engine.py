"""Full-game engine tests: CPU-only, random play, real auctions."""
from arena.bidders import Bid30Bidder, HeuristicBidder
from arena.engine import ArenaConfig, run_half
from arena.match import run_match, summarize
from arena.play import RandomPlay


def _cfg(**kw) -> ArenaConfig:
    return ArenaConfig(**{"marks_to_win": 3, "base_seed": 99, **kw})


def _half(a_team=0, n_games=4, **kw):
    return run_half(
        n_games=n_games, a_team=a_team, cfg=_cfg(**kw),
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
    )


def test_games_complete_with_exact_winner_marks():
    for g in _half():
        assert g.marks[g.winner_team] >= 3
        assert g.marks[1 - g.winner_team] < 3
        # marks ledger is consistent hand by hand
        run = [0, 0]
        for h in g.hands:
            run[0] += h.marks_delta[0]
            run[1] += h.marks_delta[1]
            assert tuple(run) == h.marks_after
        assert tuple(run) == g.marks


def test_hand_bookkeeping():
    for g in _half():
        for i, h in enumerate(g.hands):
            assert h.hand_idx == i
            assert sum(h.team_points) == 42
            assert sum(1 for b in h.bids if b > 0) >= 1
            assert h.bid_value == max(h.bids)
            assert h.bids[h.bidder] == h.bid_value
            assert 0 <= h.decl_id <= 6  # arena bidders declare pip trumps


def test_deterministic_replay():
    assert _half() == _half()


def test_paired_halves_replay_identical_deals():
    """Deal seeds and auctions depend only on (base_seed, game, hand), never
    on team assignment — so hand k of game g is the same deal with the same
    auction in both halves. (Game lengths may still differ: play diverges.)"""
    h0, h1 = _half(a_team=0), _half(a_team=1)
    for g0, g1 in zip(h0, h1):
        for a, b in zip(g0.hands, g1.hands):
            assert a.seed == b.seed
            assert a.dealer == b.dealer
            assert a.bids == b.bids
            assert a.bidder == b.bidder
            assert a.decl_id == b.decl_id


def test_bid30_baseline_shape():
    """Bid30 vs Bid30: every contract is 30, won by the seat left of the
    shaker, and never forced or redealt."""
    records = run_half(
        n_games=3, a_team=0, cfg=_cfg(),
        bid_a=Bid30Bidder(), bid_b=Bid30Bidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
    )
    for g in records:
        for h in g.hands:
            assert h.bid_value == 30
            assert h.bidder == (h.dealer + 1) % 4
            assert h.redeals == 0 and not h.forced


def test_dealer_rotates():
    """The shake advances one seat per hand, plus one per pass-out reshake."""
    for g in _half():
        for prev, nxt in zip(g.hands, g.hands[1:]):
            assert nxt.dealer == (prev.dealer + 1 + nxt.redeals) % 4


def _match_kw(**kw):
    """Fresh policies per call: RandomPlay is stateful (one RNG stream)."""
    return {
        "bid_a": HeuristicBidder(), "bid_b": HeuristicBidder(),
        "play_a": RandomPlay(seed=1), "play_b": RandomPlay(seed=2),
        "n_games": 8, "cfg": _cfg(), **kw,
    }


def test_fast_batching_pools_halves():
    """Fast batching preserves everything that is per-game deterministic:
    game order, pairing structure, deal seeds, and opening auctions. Play
    diverges (batch composition regroups the policies' RNG streams), so
    later hands may differ — that is the accepted trade."""
    exact = run_match(**_match_kw())
    fast = run_match(**_match_kw(), fast_batching=True)
    assert [(g.game_idx, g.a_team) for g in fast.games] == \
           [(g.game_idx, g.a_team) for g in exact.games]
    for gf, ge in zip(fast.games, exact.games):
        hf, he = gf.hands[0], ge.hands[0]
        assert (hf.seed, hf.dealer, hf.bids, hf.bidder, hf.decl_id) == \
               (he.seed, he.dealer, he.bids, he.bidder, he.decl_id)
        # completed games with a clean marks ledger
        assert gf.marks[gf.winner_team] >= 3
        run = [0, 0]
        for h in gf.hands:
            run[0] += h.marks_delta[0]
            run[1] += h.marks_delta[1]
        assert tuple(run) == gf.marks


def test_fast_batching_deterministic():
    a = run_match(**_match_kw(), fast_batching=True)
    b = run_match(**_match_kw(), fast_batching=True)
    assert a.games == b.games


def test_match_summary():
    result = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=8, cfg=_cfg(), label_a="ha", label_b="hb",
    )
    s = summarize(result)
    assert s["n_games"] == 8
    assert s["a_wins"] == sum(1 for g in result.games if g.a_won)
    assert 0.0 <= s["a_game_win_rate"] <= 1.0
    assert s["n_hands"] == sum(len(g.hands) for g in result.games)
    assert abs(s["auction"]["a_offense_share"] +
               (s["contracts"]["b_offense"]["contracts"] / s["n_hands"]) - 1.0) < 1e-9
    assert sum(s["auction"]["bid_hist"].values()) == s["n_hands"]

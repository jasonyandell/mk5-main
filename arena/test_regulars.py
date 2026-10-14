"""The regulars' table (#149): the bidder model, the memory, the varied
bidder and the engine's per-game memory hook."""

import random

from arena.auction import PASS, BidContext, legal_bids, run_auction
from arena.bidders import HeuristicBidder, VariedBidder
from arena.engine import ArenaConfig, run_paired
from arena.hand_metrics import best_trump
from arena.play import RandomPlay
from arena.regulars import GRID, TREMBLE, BidderBelief, RegularsPlay, action_likelihood, bids_by_rule, turns
from forge.oracle.rng import deal_from_seed
from scenario.planner import AUCTION_SLOTS, ceiling_slot
from scenario.rules import mask_of


def test_turns_read_the_public_auction():
    # dealer 3: seats bid 0, 1, 2, 3; seat 1 bids 30, seat 2 31, the 84 leaves no point bid
    assert turns((PASS, 30, 31, PASS), 3) == [(0, PASS, 30), (1, 30, 30), (2, 31, 31), (3, PASS, 32)]
    assert turns((84, PASS, PASS, PASS), 3) == [(0, 84, 30), (1, PASS, None), (2, PASS, None), (3, PASS, None)]


def test_the_rule_is_the_heuristic_bidders():
    """`bids_by_rule` is HeuristicBidder's decision, at every caution of the grid and target."""
    rng = random.Random(3)
    checked = 0
    for _ in range(3000):
        hand = tuple(rng.sample(range(28), 7))
        slot = ceiling_slot(mask_of(hand))
        high = rng.choice((0, 30, 31, 33, 36, 40, 41))
        for c in GRID:
            ctx = BidContext(hand=hand, seat=0, dealer=3, bids=(-1, -1, -1, -1), high_bid=high,
                             high_seat=-1 if high == 0 else 1, legal=legal_bids(high))
            bid = HeuristicBidder(caution=c).bid(ctx, rng)
            assert (bid != PASS) == bids_by_rule(slot, c, max(30, high + 1))
            checked += bid != PASS
    assert checked > 500


def test_a_regular_learns_a_fixed_bidder_and_misreads_a_varied_one():
    """Over one game's worth of revealed auctions, the posterior on a caution-0
    bidder lands on the behaviours that fit it; the table it then gives reads
    a bid at 30 as a ceiling of at least 30, softened by the tremble."""
    fixed, flat = BidderBelief(), BidderBelief()
    for h in range(11):
        hand = mask_of(deal_from_seed(1000 + h)[0])
        slot = ceiling_slot(hand)
        fixed.learn(30 if bids_by_rule(slot, 0, 30) else PASS, 30, slot)
    assert fixed.pi[GRID.index(0)] == max(fixed.pi) and abs(sum(fixed.pi) - 1) < 1e-12
    table = fixed.table(30, 30)
    lo, hi = TREMBLE / 2, 1 - TREMBLE / 2
    assert table[0] == lo                                    # no suit of three: a bid is a tremble
    assert abs(table[1 + 42] - hi) < 1e-9                    # a 42 ceiling bids at every caution of the grid
    flat_table = flat.table(30, 30)
    assert flat_table[1 + 32] == sum(action_likelihood(30, 30, 33, c) for c in GRID) / len(GRID)
    assert flat.table(PASS, None) is None and flat.table(31, 30) is None   # no evidence
    flat.learn(31, 30, 5)
    assert flat.pi == [1 / len(GRID)] * len(GRID)
    assert len(table) == AUCTION_SLOTS and all(0 < x <= 1 for x in table)


def test_the_varied_bidder_draws_its_caution_from_the_deals_rng():
    hands = deal_from_seed(7)
    a = run_auction(hands, 0, (VariedBidder(),) * 4, random.Random(5))
    b = run_auction(hands, 0, (VariedBidder(),) * 4, random.Random(5))
    assert a == b
    seen = set()
    for seed in range(200):
        v = VariedBidder()
        hand = tuple(deal_from_seed(seed)[0])
        ctx = BidContext(hand=hand, seat=0, dealer=3, bids=(-1, -1, -1, -1), high_bid=0, high_seat=-1,
                         legal=legal_bids(0))
        bid = v.bid(ctx, random.Random(seed))
        seen.add(v.caution)
        best = best_trump(hand, 3)
        assert (bid != PASS) == (best is not None and min(42, best.bid_ceiling - v.caution) >= 30)
    assert seen == {-5, 0, 5}


class _Recorder(RandomPlay):
    remembers = True

    def __init__(self):
        super().__init__(0)
        self.keys: set = set()
        self.seen: dict = {}

    def choose(self, states, bid_values, marks=None, marks_to_win=7, games=None):
        assert games is not None and len(games) == len(states)
        self.keys |= set(games)
        return super().choose(states, bid_values, marks, marks_to_win)

    def observe(self, game, hand):
        self.seen.setdefault(game, []).append(hand.hand_idx)


def test_the_engine_tells_a_remembering_policy_its_games_and_hands():
    rec = _Recorder()
    games = run_paired(half=2, cfg=ArenaConfig(marks_to_win=2, base_seed=4), bid_a=HeuristicBidder(),
                       bid_b=HeuristicBidder(), play_a=rec, play_b=RandomPlay(1))
    assert rec.keys == {(i, t) for t in (0, 1) for i in range(2)}
    for g in games:
        assert rec.seen[(g.game_idx, g.a_team)] == [h.hand_idx for h in g.hands]


def test_a_regular_remembers_within_a_game_only():
    class _Walt:
        class walt:
            class config:
                reading = object()

        def choose(self, states, bid_values, marks=None, marks_to_win=7, auctions=None):
            self.auctions = auctions
            return [0] * len(states)

    r, r0 = RegularsPlay(_Walt(), memory=True), RegularsPlay(_Walt(), memory=False)
    assert r.remembers and not r0.remembers
    games = run_paired(half=1, cfg=ArenaConfig(marks_to_win=3, base_seed=9), bid_a=HeuristicBidder(),
                       bid_b=HeuristicBidder(), play_a=RandomPlay(0), play_b=RandomPlay(1))
    g = games[0]
    for hand in g.hands:
        r.observe((0, 0), hand)
    moved = [r.belief((0, 0), s).pi != r0.belief((0, 0), s).pi for s in range(4)]
    assert any(moved)
    assert r.belief((1, 0), 0).pi == [1 / len(GRID)] * len(GRID)       # another game starts flat
    assert r0.belief((0, 0), 0).pi == [1 / len(GRID)] * len(GRID)      # R0 never learns

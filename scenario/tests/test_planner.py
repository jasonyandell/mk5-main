"""The planner solves the represented problem: optimality against a fixed
field, the information inequality, scale invariance, determinism, legality."""
import random

import pytest

from forge.zeb.game import apply_action, current_player, is_terminal, legal_actions, new_game

from arena.engine import ArenaConfig, run_half
from arena.bidders import HeuristicBidder
from arena.play import RandomPlay
from arena.walt_play import WaltPlay, parse_walt_spec
from scenario.planner import Config, Rung, Walt, _Invocation, _bits
from scenario.record import record_from_state, remaining_hand
from scenario.rules import rules_for, tiles_of
from scenario.tests.test_rules_record import random_playing_states


def _mid_hand_states(n: int, seed: int, min_plays: int = 8):
    out = []
    for s in random_playing_states(60, seed=seed):
        if not is_terminal(s) and len(s.played) >= min_plays and len(legal_actions(s)) > 1:
            out.append(s)
        if len(out) == n:
            break
    return out


def _invocation(walt: Walt, state, bid: int) -> _Invocation:
    rec = record_from_state(state)
    viewer = rec.seat
    own = remaining_hand(state, viewer)
    return walt._invocation(walt.config.level, viewer, own, rec, rules_for(state.decl_id),
                            bid, state.bidder % 2)


def _brute_plan_value(inv: _Invocation, rec, r: int) -> float:
    """Independent best plan for the viewer in ONE scenario against its tape:
    plain minimax-free search over the viewer's own choices, the other seats
    fixed by the dice. The planner's cell recursion must agree exactly."""
    if inv.make:
        if rec.banked[inv.bidder_team] >= inv.bid:
            return 1.0
        if rec.banked[1 - inv.bidder_team] > 42 - inv.bid:
            return 0.0
    if rec.finished:
        return inv._score(rec)
    seat = rec.seat
    world = inv.worlds[r]
    hand = world[seat] & ~rec.played
    legal = inv.rules.legal(hand, rec.led)
    if seat == inv.viewer:
        vals = [_brute_plan_value(inv, rec.advance(t, inv.rules), r) for t in tiles_of(legal)]
        return max(vals) if inv.maximize else min(vals)
    return _brute_plan_value(inv, rec.advance(inv._dice(r, rec, legal), inv.rules), r)


def test_single_scenario_value_is_the_optimal_plan_against_the_tape():
    walt = Walt(Config((Rung(1, None),), "make", seed=3))
    for state in _mid_hand_states(25, seed=21, min_plays=12):
        inv = _invocation(walt, state, 30)
        assert len(inv.worlds) == 1
        root = inv.root
        assert inv.value(root, 1) == _brute_plan_value(inv, root, 0)


def test_information_inequality_holds_on_every_split():
    """V(mu1 + mu2) <= V(mu1) + V(mu2): a plan that may use the label of the
    split can only do better. Exact on the fixed sample tree."""
    walt = Walt(Config((Rung(8, None),), "make", seed=5))
    for state in _mid_hand_states(12, seed=31, min_plays=12):
        inv = _invocation(walt, state, 30)
        everyone = (1 << len(inv.worlds)) - 1
        whole = inv.value(inv.root, everyone)
        rng = random.Random(1)
        for _ in range(4):
            a = rng.randrange(1, everyone)
            b = everyone ^ a
            assert whole <= inv.value(inv.root, a) + inv.value(inv.root, b) + 1e-9
        # And the plan on the whole cell is at least as good as any single action
        # applied to the whole cell (it IS the max over them).
        q = inv.action_values()
        best = max(q.values()) if inv.maximize else min(q.values())
        assert abs(best * inv._mass(everyone) - whole) < 1e-9


def test_scale_invariance_of_the_decision():
    base = Walt(Config((Rung(6, 6),), "points", seed=8))
    scaled = Walt(Config((Rung(6, 6),), "points", seed=8),
                  belief=lambda sit, worlds: [3.0] * len(worlds))
    for state in _mid_hand_states(10, seed=41):
        t1, q1 = base.decide(state, 31)
        t2, q2 = scaled.decide(state, 31)
        assert t1 == t2
        assert q1.keys() == q2.keys()
        for t in q1:
            assert abs(q1[t] - q2[t]) < 1e-9


def test_decisions_are_deterministic_and_legal():
    walt = Walt(Config((Rung(4, 4), Rung(8, 8)), "make", seed=0))
    for state in _mid_hand_states(8, seed=51):
        t1, q1 = walt.decide(state, 30)
        t2, q2 = walt.decide(state, 30)
        assert t1 == t2 and q1 == q2
        seat = current_player(state)
        assert t1 in [state.hands[seat][a] for a in legal_actions(state)]
        assert set(q1) == {state.hands[seat][a] for a in legal_actions(state)}


def test_make_values_are_probabilities_and_defenders_minimize():
    walt = Walt(Config((Rung(5, None),), "make", seed=2))
    for state in _mid_hand_states(10, seed=61, min_plays=14):
        tile, q = walt.decide(state, 30)
        assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in q.values())
        on_bidding_side = current_player(state) % 2 == state.bidder % 2
        target = max(q.values()) if on_bidding_side else min(q.values())
        assert q[tile] == target


def test_forced_moves_skip_the_search():
    walt = Walt(Config((Rung(4, None),), "make"))
    for state in random_playing_states(5, seed=71):
        if is_terminal(state) or len(legal_actions(state)) != 1:
            continue
        tile, q = walt.decide(state, 30)
        assert walt.nodes == 0 and list(q) == [tile]
        return
    raise AssertionError("no forced move found in five random games")


def test_arena_half_completes_with_walt_in_both_seats():
    games = run_half(
        n_games=2, a_team=0,
        cfg=ArenaConfig(marks_to_win=1, base_seed=123),
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=WaltPlay(level=0, n_outer=3, horizon=3),
        play_b=RandomPlay(seed=2),
    )
    assert len(games) == 2 and all(g.marks[g.winner_team] >= 1 for g in games)


def test_spec_parsing():
    p = parse_walt_spec("k2,n12,m3,h10,g2,u=points")
    c = p.walt.config
    assert c.level == 2 and c.utility == "points"
    assert c.rungs == (Rung(3, 2), Rung(3, 2), Rung(12, 10))
    assert parse_walt_spec("hfull,gfull").walt.config.rungs == (Rung(4, None), Rung(16, None))


def test_tilt_dice_is_the_product_of_inverse_option_counts():
    """e^g under the dice field: replaying the hand with the scenario's hands,
    every discretionary play by a hidden seat with n options contributes 1/n;
    the viewer's own plays and forced plays contribute nothing."""
    from scenario.planner import Tilt, situation_from_state
    from scenario.record import empty_record
    walt = Walt(Config((Rung(12, None),), "make", seed=3, tilt=Tilt("dice")))
    for state in _mid_hand_states(10, seed=81, min_plays=8):
        sit = situation_from_state(state, 30)
        rules = rules_for(state.decl_id)
        inv = _invocation(walt, state, 30)
        w = walt.tilt_weights(sit, inv.worlds, rules)
        assert all(0 < x <= 1 for x in w)
        for r, hands in enumerate(inv.worlds):
            full = [hands[s] | sit.played_by(s) for s in range(4)]
            rec = empty_record(sit.plays[0][0])
            expect = 1.0
            for seat, tile in sit.plays:
                n = rules.legal(full[seat], rec.led).bit_count()
                if seat != sit.viewer and n > 1:
                    expect *= 1.0 / n
                full[seat] &= ~(1 << tile)
                rec = rec.advance(tile, rules)
            assert w[r] == expect
        # the truth is a scenario of positive weight whenever it is sampled
        truth = tuple(sum(1 << d for d in state.hands[s] if d not in state.played) for s in range(4))
        if truth in inv.worlds:
            assert w[inv.worlds.index(truth)] > 0


def test_tilt_field_at_eps_one_is_dice_and_the_field_rewards_its_own_plays():
    from scenario.planner import Tilt, situation_from_state
    dice = Walt(Config((Rung(3, 3), Rung(8, None)), "make", seed=4, tilt=Tilt("dice")))
    field1 = Walt(Config((Rung(3, 3), Rung(8, None)), "make", seed=4, tilt=Tilt("field", eps=1.0)))
    field = Walt(Config((Rung(3, 3), Rung(8, None)), "make", seed=4, tilt=Tilt("field", eps=0.1)))
    for state in _mid_hand_states(6, seed=91, min_plays=8):
        sit = situation_from_state(state, 30)
        rules = rules_for(state.decl_id)
        inv = _invocation(dice, state, 30)
        assert dice.tilt_weights(sit, inv.worlds, rules) == field1.tilt_weights(sit, inv.worlds, rules)
        w = field.tilt_weights(sit, inv.worlds, rules)
        assert all(x > 0 for x in w)
    # a one-rung player cannot model plays with rung 0
    from scenario.planner import Tilt as T
    with pytest.raises(ValueError):
        Config((Rung(8, None),), "make", tilt=T("field"))


def test_declaration_belief_favors_hands_that_would_have_called_the_trump():
    from arena.hand_metrics import best_trump
    from arena.walt_play import declaration_belief
    from scenario.planner import situation_from_state
    belief = declaration_belief(eps=0.1)
    walt = Walt(Config((Rung(16, None),), "make", seed=5))
    seen = False
    for state in _mid_hand_states(20, seed=101, min_plays=4):
        sit = situation_from_state(state, 30)
        inv = _invocation(walt, state, 30)
        w = belief(sit, inv.worlds)
        if sit.bidder == sit.viewer:
            assert w == [1.0] * len(inv.worlds)
            continue
        for x, hands in zip(w, inv.worlds):
            full = hands[sit.bidder] | sit.played_by(sit.bidder)
            e = best_trump(tuple(t for t in range(28) if (full >> t) & 1), 0)
            assert x == (0.1 / 7 + 0.9 if e.trump == state.decl_id else 0.1 / 7)
        seen = seen or len(set(w)) > 1
    assert seen, "some sample should mix consistent and inconsistent bidder hands"


def _belief(sit, worlds):
    """A deterministic non-uniform belief: a function of the worlds alone."""
    return [1.0 + ((w[(sit.viewer + 1) % 4] * 2654435761) % 97) / 50.0 for w in worlds]


def test_analysis_outcomes_reweigh_to_the_action_values():
    """Each root action's outcomes, weighed by the belief the planner used,
    are its action value: the trace routes every scenario to the leaf the
    recursion credits it with. And the analysis is the decision, opened up."""
    import math
    from scenario.planner import Tilt
    cases = [
        (Config((Rung(8, None),), "make", seed=1), None),
        (Config((Rung(6, 5),), "points", seed=2), None),
        (Config((Rung(3, 3), Rung(8, None)), "make", seed=3, tilt=Tilt("dice")), _belief),
        (Config((Rung(3, 3), Rung(8, 6)), "points", seed=4, tilt=Tilt("field", eps=0.2)), _belief),
    ]
    for cfg, belief in cases:
        walt = Walt(cfg, belief=belief)
        for state in _mid_hand_states(6, seed=111 + cfg.seed, min_plays=6):
            tile, q = walt.decide(state, 30)
            nodes = walt.nodes
            a = walt.analyze(state, 30)
            assert (a.tile, a.q, walt.nodes) == (tile, q, nodes)
            assert len(a.worlds) == len(a.weights) and a.outcomes.keys() == q.keys()
            total = math.fsum(a.weights)
            for t, leaf in a.outcomes.items():
                assert len(leaf) == len(a.worlds)
                assert abs(math.fsum(w * o for w, o in zip(a.weights, leaf)) / total - q[t]) < 1e-9
                if cfg.utility == "make":
                    assert set(leaf) <= {0.0, 1.0}


def test_forced_moves_analyze_to_no_scenarios():
    import math
    walt = Walt(Config((Rung(4, None),), "make"))
    for state in random_playing_states(5, seed=71):
        if is_terminal(state) or len(legal_actions(state)) != 1:
            continue
        a = walt.analyze(state, 30)
        assert walt.nodes == 0 and list(a.q) == [a.tile] and math.isnan(a.q[a.tile])
        assert a.worlds == [] and a.weights == [] and a.outcomes == {a.tile: ()}
        return
    raise AssertionError("no forced move found in five random games")


def test_a_floor_needs_a_rung_below_and_is_at_least_one():
    import scenario_core
    with pytest.raises(ValueError):
        Rung(8, None, 0)
    with pytest.raises(ValueError):
        Config((Rung(8, None, 2),))
    with pytest.raises(ValueError):
        Config((Rung(8, None, 2), Rung(8, None)))
    with pytest.raises(ValueError):
        scenario_core.Core([(8, None, 2), (8, None, 1)], "make", 0, "none", 0.1)
    Config((Rung(8, None), Rung(8, None, 9)))  # above the sample is allowed: the root still decides


def test_a_floor_above_the_sample_hands_every_interior_decision_down(monkeypatch):
    seen: list[tuple[int, bool]] = []
    floored = _Invocation._floored

    def spy(self, alive):
        out = floored(self, alive)
        seen.append((self.k, out))
        return out

    monkeypatch.setattr(_Invocation, "_floored", spy)
    walt = Walt(Config((Rung(3, 3), Rung(8, None, 9)), "points", seed=5))
    for state in _mid_hand_states(3, seed=41, min_plays=10):
        walt.decide(state, 30)
    live = [out for k, out in seen if k == 1]
    assert live and all(live)                              # the live seat never maximizes below the root
    assert all(not out for k, out in seen if k == 0)       # rung 0 keeps its floor of one


def test_spec_sets_the_live_floor():
    rungs = parse_walt_spec("k1,n8,m2,f=4,py").walt.config.rungs
    assert rungs[-1].floor == 4 and rungs[0].floor == 1
    assert parse_walt_spec("k1,n8,py").walt.config.rungs[-1].floor == 1
    with pytest.raises(ValueError):
        parse_walt_spec("k0,n8,f=2")

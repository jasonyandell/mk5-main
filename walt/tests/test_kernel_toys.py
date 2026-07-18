"""K2 toy gates for the net-free kernel (CONTRACTS.md).

On ≤2-trick endgames with ≤12 worlds, `br_solve` must equal an explicit
enumeration over ALL of hero's pure info-set strategies (walt T2 pattern),
with every reference number produced INDEPENDENTLY of the kernel: worlds
are replayed through the forge zeb engine; profile moves come from the
same rule/oracle the kernel was given, applied per state; path
probabilities multiply per world. Covered modes:

- deterministic rule σ (compile_rule_sigma → SigmaTable, fast + rewalk)
- the same rule exported as a prob-1 StochasticProfile keyed by path
  hashes (exercises the dict-provider/path-id machinery end to end)
- uniform-over-legal StochasticProfile (full-width expectimax)
- hidden-hero BR vs the uniform profile (per-hero-hand decomposition)
- payoff_make, and weight-split invariance

A small jud-σ batch (compile_sigma vs an oracle-driven reference walk)
ties the toys to the real field; the 46-fixture K1 sweep in bench_kernel
is the at-scale version.

Run: PYTHONPATH=. .venv python -m pytest walt/tests/test_kernel_toys.py
"""
from __future__ import annotations

import random
from itertools import product

import numpy as np
import pytest

from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)
from forge.zeb.types import BidState, GamePhase, ZebGameState

from walt.contracts import EndgameRoot
from walt.kernel import (
    StochasticProfile,
    br_solve,
    build_subgame,
    compile_rule_sigma,
    payoff_make,
    payoff_points,
)
from walt.kernel.subgame import paths_of
from walt.tables import hand_to_mask, mask_to_tiles
from walt.worlds import enumerate_worlds, seat_order

ATOL = 1e-9
MAX_STRATS = 4096
MAX_PATHS_PER_WORLD = 6000


# --------------------------------------------------------------------- #
#  Toy roots + per-world engine states (T2 pattern)                      #
# --------------------------------------------------------------------- #

def gen_root(seed: int, max_tiles: int = 2, max_worlds: int = 12):
    """Random playout to a root where the mover holds ≤ max_tiles (may be
    mid-trick). Returns (root, worlds) or None."""
    st = new_game(seed)
    rng = random.Random(seed * 7919 + 13)
    while not is_terminal(st):
        mover = current_player(st)
        rem = tuple(sorted(d for d in st.hands[mover] if d not in st.played))
        if len(st.play_history) >= 28 - 4 * max_tiles and \
                len(rem) <= max_tiles:
            root = EndgameRoot(
                decl_id=st.decl_id, bidder=st.bidder,
                bid_value=st.bid_state.high_bid, bids=st.bid_state.bids,
                dealer=st.dealer, me=mover, my_hand=rem,
                play_history=st.play_history, trick_leader=st.trick_leader,
                current_trick=st.current_trick, team_points=st.team_points)
            try:
                worlds = enumerate_worlds(root)
            except Exception:
                return None
            if not (1 <= worlds.shape[0] <= max_worlds):
                return None
            return root, worlds
        st = apply_action(st, rng.choice(legal_actions(st)))
    return None


def reconstruct_states(root, worlds) -> list:
    """One zeb state per world at the root node, hidden hands sorted."""
    me = int(root.me)
    seats = seat_order(root)
    me_orig = sorted(set(int(d) for d in root.my_hand)
                     | {int(d) for (s, d) in root.play_history if s == me})
    hidden_played = {s: [int(d) for (ss, d) in root.play_history if ss == s]
                     for s in seats}
    states = []
    for row in worlds:
        hands = [None, None, None, None]
        hands[me] = tuple(me_orig)
        for j, s in enumerate(seats):
            orig = sorted(set(mask_to_tiles(int(row[j])))
                          | set(hidden_played[s]))
            hands[s] = tuple(orig)
        assert all(len(h) == 7 for h in hands)
        assert len(set().union(*hands)) == 28
        base = ZebGameState(
            hands=tuple(hands), dealer=int(root.dealer),
            phase=GamePhase.PLAYING,
            bid_state=BidState(bids=tuple(int(b) for b in root.bids),
                               high_bidder=int(root.bidder),
                               high_bid=int(root.bid_value)),
            decl_id=int(root.decl_id), bidder=int(root.bidder),
            played=frozenset(), play_history=(), current_trick=(),
            trick_leader=int(root.bidder), team_points=(0, 0))
        st = base
        for (seat, dom) in root.play_history:
            assert current_player(st) == seat
            st = apply_action(st, st.hands[seat].index(int(dom)))
        states.append(st)
    return states


# --------------------------------------------------------------------- #
#  Independent reference: path enumeration + explicit strategy product   #
# --------------------------------------------------------------------- #

def _legal_ids(st) -> list[int]:
    mover = current_player(st)
    return sorted(st.hands[mover][sl] for sl in legal_actions(st))


def _rem_mask(st, seat) -> int:
    return int(hand_to_mask(d for d in st.hands[seat]
                            if d not in st.played))


def enum_world_paths(st0, hero, sigma_move, bid_team):
    """All hero-full-width paths of ONE world: [(prob, decl_pts, req)]
    where req = ((infoset_key, hero_move), ...) and sigma_move(st) is
    either a move id (deterministic) or None (uniform-over-legal)."""
    out = []

    def rec(st, prob, req):
        if len(out) > MAX_PATHS_PER_WORLD:
            raise OverflowError("toy too wide")
        if is_terminal(st):
            out.append((prob, int(st.team_points[bid_team]), req))
            return
        mover = current_player(st)
        legal = _legal_ids(st)
        if mover == hero:
            key = (_rem_mask(st, hero), st.play_history)
            for a in legal:
                rec(apply_action(st, st.hands[mover].index(a)),
                    prob, req + ((key, a),))
        else:
            mv = sigma_move(st)
            if mv is None:                      # uniform over legal
                k = len(legal)
                for a in legal:
                    rec(apply_action(st, st.hands[mover].index(a)),
                        prob / k, req)
            else:
                assert mv in legal
                rec(apply_action(st, st.hands[mover].index(mv)), prob, req)

    rec(st0, 1.0, ())
    return out


def ref_br_value(states, weights, hero, sigma_move, payoff43, bid_team,
                 sign):
    """max over ALL hero pure info-set strategies of the weighted expected
    payoff (unnormalized: caller divides by total weight). Returns
    (best, n_strats), or None if the product exceeds MAX_STRATS."""
    infosets: dict = {}
    grouped = []                    # per world: dict req -> Σ prob*payoff
    for st in states:
        paths = enum_world_paths(st, hero, sigma_move, bid_team)
        g: dict = {}
        for prob, pts, req in paths:
            for key, a in req:
                infosets.setdefault(key, set()).add(a)
            g[req] = g.get(req, 0.0) + prob * float(payoff43[pts])
        grouped.append(g)

    keys = sorted(infosets, key=repr)
    choice_lists = [sorted(infosets[k]) for k in keys]
    n = 1
    for c in choice_lists:
        n *= len(c)
        if n > MAX_STRATS:
            return None
    best = None
    for combo in product(*choice_lists):
        strat = dict(zip(keys, combo))
        tot = 0.0
        for w, g in zip(weights, grouped):
            for req, v in g.items():
                if all(strat[k] == a for k, a in req):
                    tot += float(w) * v
        if best is None or sign * tot > sign * best:
            best = tot
    return best, n


def ref_hidden_hero(root, worlds, weights, hero, sigma_move, payoff43):
    """Hidden-hero BR reference: exact decomposition by hero's root hand
    (info sets are disjoint across groups), per-group strategy product.
    Returns (total, max group n_strats) or None."""
    seats = seat_order(root)
    col = seats.index(hero)
    bid_team = int(root.bidder) % 2
    sign = 1.0 if hero % 2 == bid_team else -1.0
    total = 0.0
    n_max = 1
    hands = worlds[:, col]
    for hv in np.unique(hands):
        m = hands == hv
        states = reconstruct_states(root, worlds[m])
        got = ref_br_value(states, weights[m], hero, sigma_move, payoff43,
                           bid_team, sign)
        if got is None:
            return None
        total += got[0]
        n_max = max(n_max, got[1])
    return total, n_max


# --------------------------------------------------------------------- #
#  Rules (deterministic toy fields, deliberately jud-unlike)             #
# --------------------------------------------------------------------- #

def rule_low(seat, hand_mask, legal_mask, decl_id):
    """Lowest legal domino id."""
    return (legal_mask & -legal_mask).bit_length() - 1


def rule_high(seat, hand_mask, legal_mask, decl_id):
    """Highest legal domino id."""
    return legal_mask.bit_length() - 1


def _rule_sigma_move(rule, decl_id):
    def sigma_move(st):
        mover = current_player(st)
        legal = _legal_ids(st)
        lm = int(hand_to_mask(legal))
        return int(rule(mover, _rem_mask(st, mover), lm, decl_id))
    return sigma_move


def _iter_toys(seed0, max_tiles=2, max_worlds=12, span=30000):
    seed = seed0
    while seed < seed0 + span:
        seed += 1
        g = gen_root(seed, max_tiles, max_worlds)
        if g is not None:
            yield g


def _collect_toys(seed0, n, max_tiles=2, max_worlds=12):
    toys = []
    for g in _iter_toys(seed0, max_tiles, max_worlds):
        toys.append(g)
        if len(toys) == n:
            return toys
    raise AssertionError(f"only {len(toys)} toys from seed {seed0}")


# --------------------------------------------------------------------- #
#  Gates                                                                 #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("rule", [rule_low, rule_high])
def test_deterministic_rule_sigma(rule):
    """br_solve(SigmaTable) == strategy enumeration on 3-tile toys (real
    strategy products, not forced tails); fast == rewalk; the same rule
    exported as a prob-1 StochasticProfile agrees too."""
    pay = payoff_points()
    done = 0
    nontrivial = 0
    for root, worlds in _iter_toys(3000, max_tiles=3, max_worlds=12):
        if done >= 18 and nontrivial >= 8:
            break
        bid_team = int(root.bidder) % 2
        sign = 1.0 if int(root.me) % 2 == bid_team else -1.0
        weights = np.linspace(1.0, 2.0, worlds.shape[0])
        states = reconstruct_states(root, worlds)
        got = ref_br_value(states, weights, int(root.me),
                           _rule_sigma_move(rule, root.decl_id), pay,
                           bid_team, sign)
        if got is None:
            continue
        ref, n_strats = got
        nontrivial += n_strats >= 6
        ref /= float(weights.sum())

        tab = compile_rule_sigma(root, worlds, rule)
        sub = build_subgame(root, worlds, weights)
        br = br_solve(sub, tab, pay)
        assert abs(br.value - ref) <= ATOL, \
            f"seed toy: br {br.value} vs ref {ref}"
        br2 = br_solve(sub, tab, pay, rewalk=True)
        assert br2.best_move == br.best_move
        assert abs(br2.value - br.value) <= 1e-12
        assert br.root_values[br.best_move] == br.value

        # export the table as a prob-1 StochasticProfile (path-hash keys)
        prof = StochasticProfile()
        for w, dec in enumerate(tab.dec):
            if dec is None:
                continue
            nodes, hds, seats, mvs = dec
            paths = paths_of(tab.waves, w, nodes.astype(np.int64)) \
                if w > 0 else np.zeros((len(nodes), 0), dtype=np.int64)
            for i in range(len(nodes)):
                prof.set(int(seats[i]), int(hds[i]),
                         tuple(paths[i].tolist()), [int(mvs[i])], [1.0])
        br3 = br_solve(sub, prof, pay)
        assert br3.best_move == br.best_move
        assert abs(br3.value - br.value) <= ATOL
        done += 1
    assert done >= 18, f"only {done} toys actually checked"
    assert nontrivial >= 8, f"only {nontrivial} nontrivial strategy products"


def test_uniform_stochastic():
    """br_solve vs the uniform-over-legal profile == expectimax reference.

    2-tile toys: full-width 3-tile toys blow the reference's path/strategy
    caps by construction (that IS the measured x10^2-10^4 stochastic
    blowup). Hero-side info-set exactness at real strategy products is
    carried by the deterministic 3-tile gates — hero expansion is the same
    machinery for every profile kind; this gate targets the probability-
    weighted world flow, which 2-tile toys exercise fully (non-uniform
    weights, mid-trick roots included)."""
    pay = payoff_points()
    prof = StochasticProfile(uniform_fallback=True)
    done = 0
    for root, worlds in _iter_toys(9000, max_tiles=2, max_worlds=8):
        if done >= 15:
            break
        bid_team = int(root.bidder) % 2
        sign = 1.0 if int(root.me) % 2 == bid_team else -1.0
        weights = np.linspace(1.0, 1.5, worlds.shape[0])
        states = reconstruct_states(root, worlds)
        try:
            got = ref_br_value(states, weights, int(root.me),
                               lambda st: None, pay, bid_team, sign)
        except OverflowError:
            continue
        if got is None:
            continue
        ref = got[0] / float(weights.sum())
        sub = build_subgame(root, worlds, weights)
        br = br_solve(sub, prof, pay)
        assert abs(br.value - ref) <= ATOL, f"br {br.value} vs ref {ref}"
        done += 1
    assert done >= 15, f"only {done} toys actually checked"


def test_hidden_hero_uniform():
    """BR of a hidden seat vs uniform: per-hand decomposition, both sides
    (2-tile toys — the target here is the group split + per-group hero
    tracking, see test_uniform_stochastic's note on tile counts)."""
    pay = payoff_points()
    prof = StochasticProfile(uniform_fallback=True)
    done = 0
    for root, worlds in _iter_toys(15000, max_tiles=2, max_worlds=6):
        if done >= 8:
            break
        weights = np.linspace(1.0, 1.4, worlds.shape[0])
        hero = next(s for s in range(4) if s != int(root.me))
        try:
            got = ref_hidden_hero(root, worlds, weights, hero,
                                  lambda st: None, pay)
        except OverflowError:
            continue
        if got is None:
            continue
        ref = got[0] / float(weights.sum())
        sub = build_subgame(root, worlds, weights)
        br = br_solve(sub, prof, pay, hero=hero)
        assert br.best_move is None and br.root_values == {}
        assert abs(br.value - ref) <= ATOL, f"br {br.value} vs ref {ref}"
        done += 1
    assert done >= 8, f"only {done} toys actually checked"


def test_make_payoff_and_weight_split():
    """payoff_make agrees with the reference; splitting a world's weight
    across duplicate rows is a no-op (T4 pattern)."""
    done = 0
    for root, worlds in _iter_toys(21000, max_tiles=3):
        if done >= 8:
            break
        pay = payoff_make(root.bid_value)
        bid_team = int(root.bidder) % 2
        sign = 1.0 if int(root.me) % 2 == bid_team else -1.0
        weights = np.ones(worlds.shape[0])
        states = reconstruct_states(root, worlds)
        got = ref_br_value(states, weights, int(root.me),
                           _rule_sigma_move(rule_low, root.decl_id), pay,
                           bid_team, sign)
        if got is None:
            continue
        ref = got[0] / float(weights.sum())
        tab = compile_rule_sigma(root, worlds, rule_low)
        sub = build_subgame(root, worlds, weights)
        br = br_solve(sub, tab, pay)
        assert abs(br.value - ref) <= ATOL

        # weight-split invariance (same worlds => same digest => same table)
        if worlds.shape[0] >= 2:
            w2 = weights.copy()
            w2[0] = 0.37
            worlds2 = np.vstack([worlds, worlds[0:1]])
            w2 = np.concatenate([w2, [0.63]])
            tab2 = compile_rule_sigma(root, worlds2, rule_low)
            sub2 = build_subgame(root, worlds2, w2)
            br2 = br_solve(sub2, tab2, pay)
            assert abs(br2.value - br.value) <= 1e-7
        done += 1
    assert done >= 8, f"only {done} toys actually checked"


def test_profile_value_and_chunked_mode():
    """profile_value == all-profile expectation reference; a starved slot
    budget must flip br_solve into per-root-move chunking with identical
    value/best_move/root_values."""
    from walt.kernel import profile_value

    pay = payoff_points()
    prof = StochasticProfile(uniform_fallback=True)
    done = 0
    for root, worlds in _collect_toys(33000, 10, max_worlds=6):
        bid_team = int(root.bidder) % 2
        weights = np.linspace(1.0, 1.3, worlds.shape[0])
        states = reconstruct_states(root, worlds)
        # all-profile reference: hero = no seat (-2 never matches a mover)
        try:
            ref = 0.0
            for w, st in zip(weights, states):
                for prob, pts, _req in enum_world_paths(
                        st, -2, lambda s: None, bid_team):
                    ref += float(w) * prob * float(pay[pts])
        except OverflowError:
            continue
        ref /= float(weights.sum())
        sub = build_subgame(root, worlds, weights)
        pv = profile_value(sub, prof, pay)
        assert abs(pv - ref) <= ATOL, f"pv {pv} vs ref {ref}"

        # chunked fallback parity (slot budget too small for one tree,
        # big enough for per-root-move trees)
        from walt.kernel import KernelMemoryError

        br = br_solve(sub, prof, pay)
        for budget in (2000, 700, 300):
            try:
                br_c = br_solve(sub, prof, pay, slot_budget=budget)
            except KernelMemoryError:
                continue        # even single-move trees blew this budget
            if br_c.meta["mode"] == "stochastic-chunked":
                assert br_c.best_move == br.best_move
                assert abs(br_c.value - br.value) <= 1e-12
                assert br_c.root_values == pytest.approx(br.root_values,
                                                         abs=1e-12)
                break
        done += 1
    assert done >= 6, f"only {done} toys actually checked"


def test_full_width_walk():
    """expand_full_width structural invariants (the CFR lane's walk): each
    world's leaf-slot count equals its number of consistent full-width
    paths (a world survives a public action iff the acting seat holds the
    tile), and every wave keeps its slot partition + path ids."""
    from walt.kernel import expand_full_width

    done = 0
    for root, worlds in _iter_toys(39000, max_tiles=2, max_worlds=6):
        if done >= 6:
            break
        bid_team = int(root.bidder) % 2
        states = reconstruct_states(root, worlds)
        try:
            ref_counts = [
                len(enum_world_paths(st, -2, lambda s: None, bid_team))
                for st in states]
        except OverflowError:
            continue
        weights = np.ones(worlds.shape[0])
        sub = build_subgame(root, worlds, weights)
        res = expand_full_width(sub)
        leaf = res["leaf"]
        got = np.bincount(leaf["sworld"], minlength=worlds.shape[0])
        assert got.tolist() == ref_counts
        for w in res["waves"][:-1]:
            assert "snode" in w and "ph1" in w
        done += 1
    assert done >= 6, f"only {done} toys actually checked"


def test_jud_sigma_toys():
    """compile_sigma (the real net) on toys vs an oracle-driven reference
    walk — ties K2 to the actual field; K1 at scale lives in bench_kernel."""
    import torch

    torch.set_num_threads(1)
    from walt.field import FieldOracle, PubState

    oracle = FieldOracle(device="cpu")
    from walt.kernel import compile_sigma

    pay = payoff_points()

    def jud_move(st):
        mover = current_player(st)
        pub = PubState.from_state(st)
        return int(oracle.decisions(
            [(mover, _rem_mask(st, mover), pub)])[0])

    done = 0
    for root, worlds in _collect_toys(27000, 8, max_tiles=3):
        bid_team = int(root.bidder) % 2
        sign = 1.0 if int(root.me) % 2 == bid_team else -1.0
        weights = np.ones(worlds.shape[0])
        states = reconstruct_states(root, worlds)
        got = ref_br_value(states, weights, int(root.me), jud_move, pay,
                           bid_team, sign)
        if got is None:
            continue
        ref = got[0] / float(weights.sum())
        tab = compile_sigma(root, worlds, oracle)
        sub = build_subgame(root, worlds, weights)
        br = br_solve(sub, tab, pay)
        assert abs(br.value - ref) <= ATOL, f"br {br.value} vs ref {ref}"
        done += 1
    assert done >= 5, f"only {done} toys actually checked"

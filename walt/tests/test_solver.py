#!/usr/bin/env python
"""walt/tests/test_solver.py — correctness gates for walt.solver.solve.

Plain executable script (no pytest): prints one PASS/FAIL line per gate and
exits nonzero on any FAIL. The references here are INDEPENDENT of walt.solver:
they drive the forge zeb engine (forge.zeb.game.apply_action) and the real
arena.jud_play.JudPlay / the field oracle directly, never walt's own recursion.

Gates (DESIGN.md §Correctness gates):
  T1 known-world degeneracy   — |worlds|=1 ⇒ solver == engine+JudPlay brute rollout.
  T2 exactness                — solver == max over ALL enumerated pure info-set
                                strategies, each evaluated by simulating every
                                world through the engine + real JudPlay.
  T3 dominance                — solver value ≥ value of (a) random pure me and
                                (b) jud me, on the same worlds/field.
  T4 weight-split invariance  — duplicating a world with split weight is a no-op.
  T6 claim-vs-cash closure    — replaying the solver's own re-solved strategy
                                across all alive worlds cashes to the root value.
"""
from __future__ import annotations

import itertools
import random
import sys
import time

import numpy as np


from forge.zeb.game import (  # noqa: E402
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)
from forge.zeb.types import BidState, GamePhase, ZebGameState  # noqa: E402
from champion.jud_net import load_jud_net  # noqa: E402
from arena.jud_play import JudPlay  # noqa: E402

from walt.contracts import EndgameRoot  # noqa: E402
from walt.field import FieldOracle, PubState  # noqa: E402
from walt.solver import solve  # noqa: E402
from walt.tables import hand_to_mask, mask_to_tiles  # noqa: E402
from walt.worlds import enumerate_worlds, seat_order  # noqa: E402

ATOL = 1e-9

# Shared field + reference policy (both wrap champion/jud_net.pt).
_MODEL = load_jud_net()
ORACLE = FieldOracle()
JP = JudPlay(_MODEL)


# --------------------------------------------------------------------------- #
#  Root generation + per-world engine-state reconstruction                    #
# --------------------------------------------------------------------------- #

def gen_root(seed: int, tricks: int):
    """A trick-start endgame root with `tricks` tiles per seat (me = leader).

    Returns (root, true_state, worlds) or None if enumeration is degenerate.
    """
    st = new_game(seed)
    rng = random.Random((seed << 4) ^ tricks)
    target = 7 - tricks
    while not (len(st.play_history) == target * 4 and not st.current_trick):
        st = apply_action(st, rng.choice(legal_actions(st)))
    me = current_player(st)
    rem = tuple(sorted(d for d in st.hands[me] if d not in st.played))
    root = EndgameRoot(
        decl_id=st.decl_id, bidder=st.bidder, bid_value=st.bid_state.high_bid,
        bids=st.bid_state.bids, dealer=st.dealer, me=me, my_hand=rem,
        play_history=st.play_history, trick_leader=st.trick_leader,
        current_trick=st.current_trick, team_points=st.team_points,
    )
    try:
        worlds = enumerate_worlds(root)
    except Exception:
        return None
    if worlds.shape[0] == 0:
        return None
    return root, st, worlds


def reconstruct_states(root, worlds) -> list:
    """One engine ZebGameState per world, positioned at the root node, with the
    hidden hands SORTED ascending so JudPlay's lowest-slot tie-break coincides
    with the field's lowest-domino tie-break. Column j == seat_order(root)[j]."""
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
            orig = sorted(set(mask_to_tiles(int(row[j]))) | set(hidden_played[s]))
            hands[s] = tuple(orig)
        assert all(len(h) == 7 for h in hands), "each seat needs 7 tiles"
        assert len(set().union(*hands)) == 28, "world must be a full 28-tile deal"
        base = ZebGameState(
            hands=tuple(hands), dealer=int(root.dealer), phase=GamePhase.PLAYING,
            bid_state=BidState(bids=tuple(int(b) for b in root.bids),
                               high_bidder=int(root.bidder),
                               high_bid=int(root.bid_value)),
            decl_id=int(root.decl_id), bidder=int(root.bidder),
            played=frozenset(), play_history=(), current_trick=(),
            trick_leader=int(root.bidder), team_points=(0, 0),
        )
        st = base
        for (seat, dom) in root.play_history:
            assert current_player(st) == seat
            st = apply_action(st, st.hands[seat].index(int(dom)))
        states.append(st)
    return states


def _remaining_mask(st, seat) -> int:
    return int(hand_to_mask(d for d in st.hands[seat] if d not in st.played))


def _declaring_points(st, root) -> int:
    return int(st.team_points[int(root.bidder) % 2])


def _sign(root) -> float:
    return 1.0 if int(root.me) % 2 == int(root.bidder) % 2 else -1.0


# --------------------------------------------------------------------------- #
#  T1 — known-world degeneracy                                                 #
# --------------------------------------------------------------------------- #

def gate_T1(n_roots: int = 20) -> bool:
    fails = 0
    done = 0
    seed = 0
    while done < n_roots and seed < 4000:
        seed += 1
        for tricks in (2, 3, 4):
            g = gen_root(seed, tricks)
            if g is None:
                continue
            root, true_st, worlds = g
            me = int(root.me)
            sign = _sign(root)
            # single world = the TRUE hidden hands, rebuilt sorted for tie-parity
            seats = seat_order(root)
            true_row = np.array(
                [[_remaining_mask(true_st, s) for s in seats]], dtype=np.uint32)
            state = reconstruct_states(root, true_row)[0]

            # independent reference: engine + real JudPlay, only me branches
            def ref(s):
                if is_terminal(s):
                    return _declaring_points(s, root)
                mover = current_player(s)
                if mover == me:
                    best = None
                    for slot in legal_actions(s):
                        v = ref(apply_action(s, slot))
                        if best is None or sign * v > sign * best:
                            best = v
                    return best
                slot = JP.choose([s], [root.bid_value])[0]
                return ref(apply_action(s, slot))

            ref_val = ref(state)
            res = solve(root, true_row, np.array([1.0]), ORACLE, "points")
            if abs(res.value - ref_val) > ATOL:
                fails += 1
                if fails <= 5:
                    print(f"  T1 miss seed={seed} tricks={tricks} "
                          f"solve={res.value} ref={ref_val}")
            done += 1
            if done >= n_roots:
                break
    ok = fails == 0 and done >= n_roots
    print(f"{'PASS' if ok else 'FAIL'} T1 known-world degeneracy "
          f"({done} single-world roots, {fails} mismatches)")
    return ok


# --------------------------------------------------------------------------- #
#  T2 — exactness vs explicit enumeration of pure info-set strategies          #
# --------------------------------------------------------------------------- #

def _reference_infoset_max(root, worlds, weights):
    """Max over ALL of me's pure info-set strategies of the world-weighted
    expected declaring points, each strategy simulated through engine+JudPlay."""
    states = reconstruct_states(root, worlds)
    me = int(root.me)
    sign = _sign(root)
    bidteam = int(root.bidder) % 2
    total_w = float(weights.sum())

    # discover reachable me-info-sets: (my_mask, play_history) -> legal domino ids
    infosets: dict = {}

    def disc(group):  # group: list[(world_idx, state)] sharing this history
        st0 = group[0][1]
        if is_terminal(st0):
            return
        mover = current_player(st0)
        if mover == me:
            key = (_remaining_mask(st0, me), st0.play_history)
            legal_dom = tuple(sorted(st0.hands[me][sl] for sl in legal_actions(st0)))
            infosets[key] = legal_dom
            for dom in legal_dom:
                ng = [(i, apply_action(s, s.hands[me].index(dom))) for (i, s) in group]
                disc(ng)
        else:
            slots = JP.choose([s for _, s in group], [root.bid_value] * len(group))
            buckets: dict = {}
            for (i, s), sl in zip(group, slots):
                dom = s.hands[mover][sl]
                buckets.setdefault(dom, []).append((i, apply_action(s, sl)))
            for ng in buckets.values():
                disc(ng)

    disc(list(enumerate(states)))

    keys = list(infosets)
    choice_lists = [infosets[k] for k in keys]
    n_strats = int(np.prod([len(c) for c in choice_lists])) if choice_lists else 1

    best = None
    for combo in itertools.product(*choice_lists):
        strat = dict(zip(keys, combo))
        tot = 0.0
        for idx, st0 in enumerate(states):
            s = st0
            while not is_terminal(s):
                mover = current_player(s)
                if mover == me:
                    dom = strat[(_remaining_mask(s, me), s.play_history)]
                    sl = s.hands[me].index(dom)
                else:
                    sl = JP.choose([s], [root.bid_value])[0]
                s = apply_action(s, sl)
            tot += float(weights[idx]) * s.team_points[bidteam]
        avg = tot / total_w
        if best is None or sign * avg > sign * best:
            best = avg
    return best, n_strats


def gate_T2(n_roots: int = 50) -> bool:
    fails = 0
    done = 0
    seed = 1000
    while done < n_roots and seed < 12000:
        seed += 1
        g = gen_root(seed, 2)
        if g is None:
            continue
        root, _st, worlds = g
        N = worlds.shape[0]
        if not (1 <= N <= 24):
            continue
        weights = np.ones(N)
        ref_val, n_strats = _reference_infoset_max(root, worlds, weights)
        if n_strats > 4096:
            continue
        res = solve(root, worlds, weights, ORACLE, "points")
        if abs(res.value - ref_val) > ATOL:
            fails += 1
            if fails <= 5:
                print(f"  T2 miss seed={seed} N={N} strat={n_strats} "
                      f"solve={res.value} ref={ref_val}")
        done += 1
    ok = fails == 0 and done >= n_roots
    print(f"{'PASS' if ok else 'FAIL'} T2 exactness "
          f"({done} 2-trick roots vs full strategy enumeration, {fails} mismatches)")
    return ok


# --------------------------------------------------------------------------- #
#  T3 — dominance over fixed pure policies                                     #
# --------------------------------------------------------------------------- #

def _oracle_move(st, seat) -> int:
    pub = PubState.from_state(st)
    return ORACLE.decisions([(int(seat), _remaining_mask(st, seat), pub)])[0]


def _policy_value(root, worlds, weights, me_chooser) -> float:
    """Weighted mean declaring points when me follows `me_chooser` (a pure
    info-set policy) and every other seat follows the field (oracle)."""
    states = reconstruct_states(root, worlds)
    me = int(root.me)
    bidteam = int(root.bidder) % 2
    tot = 0.0
    for idx, st0 in enumerate(states):
        s = st0
        while not is_terminal(s):
            mover = current_player(s)
            if mover == me:
                dom = me_chooser(s)
            else:
                dom = _oracle_move(s, mover)
            s = apply_action(s, s.hands[mover].index(dom))
        tot += float(weights[idx]) * s.team_points[bidteam]
    return tot / float(weights.sum())


def gate_T3(n_roots: int = 30) -> bool:
    fails = 0
    done = 0
    seed = 20000
    while done < n_roots and seed < 40000:
        seed += 1
        tricks = 3 + (seed % 2)  # 3 or 4
        g = gen_root(seed, tricks)
        if g is None:
            continue
        root, _st, worlds = g
        N = worlds.shape[0]
        if not (2 <= N <= 60):
            continue
        weights = np.ones(N)
        sign = _sign(root)
        res = solve(root, worlds, weights, ORACLE, "points")

        me = int(root.me)

        def rand_me(s, _seed=seed):
            legal = [s.hands[me][sl] for sl in legal_actions(s)]
            key = (_remaining_mask(s, me), s.play_history)
            r = random.Random(hash((_seed, key)) & 0xFFFFFFFF)
            return r.choice(sorted(legal))

        def jud_me(s):
            return _oracle_move(s, me)

        v_rand = _policy_value(root, worlds, weights, rand_me)
        v_jud = _policy_value(root, worlds, weights, jud_me)
        # optimal ⇒ sign-oriented value dominates any pure policy
        if sign * res.value + ATOL < sign * v_rand or \
           sign * res.value + ATOL < sign * v_jud:
            fails += 1
            if fails <= 5:
                print(f"  T3 miss seed={seed} tricks={tricks} sign={sign} "
                      f"solve={res.value} rand={v_rand} jud={v_jud}")
        done += 1
    ok = fails == 0 and done >= n_roots
    print(f"{'PASS' if ok else 'FAIL'} T3 dominance "
          f"({done} 3-4 trick roots ≥ random & jud policies, {fails} violations)")
    return ok


# --------------------------------------------------------------------------- #
#  T4 — weight-split invariance                                                #
# --------------------------------------------------------------------------- #

def gate_T4(n_roots: int = 20) -> bool:
    fails = 0
    done = 0
    seed = 50000
    while done < n_roots and seed < 60000:
        seed += 1
        tricks = 2 + (seed % 3)  # 2,3,4
        g = gen_root(seed, tricks)
        if g is None:
            continue
        root, _st, worlds = g
        N = worlds.shape[0]
        if not (2 <= N <= 50):
            continue
        weights = np.ones(N)
        base = solve(root, worlds, weights, ORACLE, "points").value
        # duplicate world 0, split its weight 1.0 -> 0.37 + 0.63
        worlds2 = np.vstack([worlds, worlds[0:1]])
        w2 = np.concatenate([weights.copy(), [0.63]])
        w2[0] = 0.37
        split = solve(root, worlds2, w2, ORACLE, "points").value
        if abs(base - split) > 1e-7:
            fails += 1
            if fails <= 5:
                print(f"  T4 miss seed={seed} base={base} split={split}")
        done += 1
    ok = fails == 0 and done >= n_roots
    print(f"{'PASS' if ok else 'FAIL'} T4 weight-split invariance "
          f"({done} roots, {fails} mismatches)")
    return ok


# --------------------------------------------------------------------------- #
#  T6 — claim-vs-cash closure                                                  #
# --------------------------------------------------------------------------- #

def gate_T6(n_roots: int = 30, payoff: str = "points") -> bool:
    fails = 0
    done = 0
    seed = 70000
    seats_cache = None
    while done < n_roots and seed < 90000:
        seed += 1
        tricks = 3 + (seed % 2)  # 3 or 4
        g = gen_root(seed, tricks)
        if g is None:
            continue
        root, _st, worlds = g
        N = worlds.shape[0]
        if not (2 <= N <= 40):
            continue
        weights = np.ones(N)
        root_res = solve(root, worlds, weights, ORACLE, payoff)

        me = int(root.me)
        seats = seat_order(root)
        col_of = {s: i for i, s in enumerate(seats)}
        states = reconstruct_states(root, worlds)
        move_cache: dict = {}          # history-key -> me best_move (cand ⟂ world)
        total = 0.0
        for wi, st0 in enumerate(states):
            st = st0
            cand_idx = np.arange(N)
            cand_rem = worlds.copy()
            while not is_terminal(st):
                mover = current_player(st)
                hist = st.play_history
                if mover == me:
                    key = (hist, _remaining_mask(st, me))
                    dom = move_cache.get(key)
                    if dom is None:
                        sub = EndgameRoot(
                            decl_id=root.decl_id, bidder=root.bidder,
                            bid_value=root.bid_value, bids=root.bids,
                            dealer=root.dealer, me=me,
                            my_hand=tuple(sorted(d for d in st.hands[me]
                                                 if d not in st.played)),
                            play_history=st.play_history,
                            trick_leader=st.trick_leader,
                            current_trick=st.current_trick,
                            team_points=st.team_points,
                        )
                        r = solve(sub, cand_rem, weights[cand_idx], ORACLE, payoff)
                        dom = r.best_move
                        move_cache[key] = dom
                    sl = st.hands[me].index(dom)
                else:
                    col = col_of[mover]
                    pub = PubState.from_state(st)
                    q = [(mover, int(cand_rem[i, col]), pub)
                         for i in range(len(cand_idx))]
                    moves = np.asarray(ORACLE.decisions(q), dtype=np.int64)
                    pos = int(np.nonzero(cand_idx == wi)[0][0])
                    dom = int(moves[pos])
                    keep = moves == dom
                    cand_idx = cand_idx[keep]
                    cand_rem = cand_rem[keep].copy()
                    cand_rem[:, col] &= np.uint32((~(1 << dom)) & 0xFFFFFFFF)
                    sl = st.hands[mover].index(dom)
                st = apply_action(st, sl)
            decl_pts = _declaring_points(st, root)
            realized = float(decl_pts) if payoff == "points" else \
                (1.0 if decl_pts >= int(root.bid_value) else 0.0)
            total += float(weights[wi]) * realized
        realized_mean = total / float(weights.sum())
        if abs(realized_mean - root_res.value) > ATOL:
            fails += 1
            if fails <= 5:
                print(f"  T6 miss seed={seed} tricks={tricks} "
                      f"root={root_res.value} cashed={realized_mean}")
        done += 1
    ok = fails == 0 and done >= n_roots
    print(f"{'PASS' if ok else 'FAIL'} T6 claim-vs-cash closure/{payoff} "
          f"({done} roots, {fails} mismatches)")
    return ok


# --------------------------------------------------------------------------- #
#  Timing report                                                              #
# --------------------------------------------------------------------------- #

def timing_report():
    for tricks in (3, 4):
        times = []
        Ns = []
        seed = 100000
        while len(times) < 15 and seed < 130000:
            seed += 1
            g = gen_root(seed, tricks)
            if g is None:
                continue
            root, _st, worlds = g
            N = worlds.shape[0]
            if not (2 <= N <= 60):
                continue
            weights = np.ones(N)
            t0 = time.perf_counter()
            solve(root, worlds, weights, ORACLE, "points")
            times.append((time.perf_counter() - t0) * 1e3)
            Ns.append(N)
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
        print(f"  timing {tricks}-trick: n={len(times)} "
              f"medianN={int(np.median(Ns))} p50={p50:.1f}ms p95={p95:.1f}ms")


# --------------------------------------------------------------------------- #

def main() -> int:
    results = []
    results.append(gate_T1(20))
    results.append(gate_T2(50))
    results.append(gate_T3(30))
    results.append(gate_T4(20))
    results.append(gate_T6(30, "points"))
    results.append(gate_T6(30, "make"))
    timing_report()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

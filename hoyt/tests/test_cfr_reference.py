#!/usr/bin/env python
"""walt/tests/test_cfr_reference.py — gates for the pure-python kernel mirror.

Plain executable script (house style): one PASS/FAIL line per gate, exit
nonzero on any FAIL. Run from the repo root:

    /Users/jason/code/mk5-main/.venv/bin/python -u walt/tests/test_cfr_reference.py

The references here are INDEPENDENT of hoyt/reference.py's recursion:
R1 replays every toy world through the real zeb engine; R2/R3 enumerate ALL
pure info-set strategies of the best responder and simulate each one.

Gates:
  R1 engine parity      — profile_value under a deterministic joint sigma ==
                          weighted mean of per-world zeb-engine playouts.
  R2 hero BR exactness  — br_solve(hero=me) == max over ALL enumerated hero
                          pure info-set strategies (every registry toy).
  R3 hidden BR exactness— br_solve(hero=hidden seat) == enumerated max.
  R4 root consistency   — root_values complete, tie rule reproduces best_move,
                          BR dominates following the profile.
  R5 weight-split       — duplicating a world with split weight is a no-op.
"""
from __future__ import annotations

import sys

import numpy as np

from forge.zeb.game import apply_action, current_player, is_terminal  # noqa: E402
from forge.zeb.types import BidState, GamePhase, ZebGameState  # noqa: E402

from hoyt import reference as ref  # noqa: E402
from hoyt import toys as T  # noqa: E402
from walt.tables import hand_to_mask, mask_to_tiles  # noqa: E402
from walt.worlds import seat_order  # noqa: E402

ATOL = 1e-9
PAY = T.payoff_points()
_failures: list[str] = []


def _gate(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _reconstruct_state(root, world_row):
    """One zeb engine state per world, positioned at the root (test_solver
    pattern): rebuild original 7-tile hands, replay the full history."""
    me = int(root.me)
    seats = seat_order(root)
    me_orig = sorted(set(int(d) for d in root.my_hand)
                     | {int(d) for (s, d) in root.play_history if s == me})
    hands = [None, None, None, None]
    hands[me] = tuple(me_orig)
    for j, s in enumerate(seats):
        played = [int(d) for (ss, d) in root.play_history if ss == s]
        hands[s] = tuple(sorted(set(mask_to_tiles(int(world_row[j]))) | set(played)))
    assert all(len(h) == 7 for h in hands)
    assert len(set().union(*hands)) == 28
    st = ZebGameState(
        hands=tuple(hands), dealer=int(root.dealer), phase=GamePhase.PLAYING,
        bid_state=BidState(bids=tuple(int(b) for b in root.bids),
                           high_bidder=int(root.bidder),
                           high_bid=int(root.bid_value)),
        decl_id=int(root.decl_id), bidder=int(root.bidder),
        played=frozenset(), play_history=(), current_trick=(),
        trick_leader=int(root.bidder), team_points=(0, 0),
    )
    for (seat, dom) in root.play_history:
        assert current_player(st) == seat
        st = apply_action(st, st.hands[seat].index(int(dom)))
    return st


def gate_R1() -> None:
    """Engine parity of profile_value: full deterministic sigma, per-world
    zeb playout, weighted mean must match to 1e-9."""
    bad = []
    for name in ("t1_decl_w3", "t2_decl_w3", "t2_def_w12", "t2_decl_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        sigma = T.lowest_legal_sigma(sub, range(4))     # every seat pinned
        val = ref.profile_value(sub, sigma, PAY)

        tot, wsum = 0.0, 0.0
        for row, w in zip(toy.worlds, toy.weights):
            st = _reconstruct_state(toy.root, row)
            node = ()
            while not is_terminal(st):
                mover = current_player(st)
                hand = int(hand_to_mask(
                    d for d in st.hands[mover] if d not in st.played))
                dom = sigma.move(mover, hand, node)
                st = apply_action(st, st.hands[mover].index(dom))
                node = node + (dom,)
            tot += float(w) * st.team_points[int(toy.root.bidder) % 2]
            wsum += float(w)
        engine_val = tot / wsum
        if abs(val - engine_val) > ATOL:
            bad.append(f"{name}: ref={val} engine={engine_val}")
    _gate(not bad, "R1 engine parity (profile_value vs zeb playouts)",
          "; ".join(bad) if bad else "4 toys, all worlds, 1e-9")


def _enum_best(sub, hero, pinned, sgn):
    isets = T.discover_free_infosets(sub, {hero}, pinned)
    best = None
    n = 0
    for p in T.pure_profiles(isets, hero):
        prof = dict(pinned)
        prof[hero] = p
        v = ref.profile_value(sub, prof, PAY)
        n += 1
        if best is None or sgn * v > sgn * best:
            best = v
    return best, n


def gate_R2() -> None:
    bad = []
    total_pures = 0
    for toy in T.all_toys():
        sub = toy.build()
        others = [s for s in range(4) if s != sub.me]
        sigma = T.lowest_legal_sigma(sub, others)
        pinned = {s: sigma for s in others}
        br = ref.br_solve(sub, sigma, PAY, hero=sub.me)
        enum, n = _enum_best(sub, sub.me, pinned, ref.sign_of(sub.me, sub.bid_team))
        total_pures += n
        if abs(br.value - enum) > ATOL:
            bad.append(f"{toy.name}: br={br.value} enum={enum}")
    _gate(not bad, "R2 hero BR == pure-strategy enumeration max",
          "; ".join(bad) if bad else f"8 toys, {total_pures} pure strategies")


def gate_R3() -> None:
    bad = []
    total_pures = 0
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        for hero in (s for s in range(4) if s != sub.me):
            others = [s for s in range(4) if s != hero]
            sigma = T.lowest_legal_sigma(sub, others)
            pinned = {s: sigma for s in others}
            br = ref.br_solve(sub, sigma, PAY, hero=hero)
            enum, n = _enum_best(sub, hero, pinned, ref.sign_of(hero, sub.bid_team))
            total_pures += n
            if abs(br.value - enum) > ATOL:
                bad.append(f"{name}/hero{hero}: br={br.value} enum={enum}")
    _gate(not bad, "R3 hidden-hero BR == pure-strategy enumeration max",
          "; ".join(bad) if bad else f"4 toys x 3 heroes, {total_pures} pures")


def gate_R4() -> None:
    bad = []
    for toy in T.all_toys():
        sub = toy.build()
        others = [s for s in range(4) if s != sub.me]
        # others' sigma must cover hero deviations (free-hero support tree);
        # the all-four sigma is only defined on its own path (follow value).
        sigma_others = T.lowest_legal_sigma(sub, others)
        sigma_all = T.lowest_legal_sigma(sub, range(4))
        br = ref.br_solve(sub, sigma_others, PAY, hero=sub.me)
        sgn = ref.sign_of(sub.me, sub.bid_team)
        legal = ref.legal_tiles(sub.my_mask0, sub.pub0.led_tile, sub.decl_id)
        if set(br.root_values) != set(legal):
            bad.append(f"{toy.name}: root_values keys {sorted(br.root_values)} "
                       f"!= legal {legal}")
            continue
        # tie rule: ascending, first strict improvement
        best_v, best_m = None, None
        for m in legal:
            v = br.root_values[m]
            if best_v is None or sgn * v > sgn * best_v:
                best_v, best_m = v, m
        if best_m != br.best_move or abs(best_v - br.value) > ATOL:
            bad.append(f"{toy.name}: tie-rule replay ({best_m},{best_v}) vs "
                       f"({br.best_move},{br.value})")
        follow = ref.profile_value(sub, sigma_all, PAY)
        if sgn * br.value < sgn * follow - ATOL:
            bad.append(f"{toy.name}: BR {br.value} worse than following {follow}")
    _gate(not bad, "R4 root_values complete + tie rule + dominance",
          "; ".join(bad) if bad else "8 toys")


def gate_R5() -> None:
    bad = []
    for name in ("t2_decl_w3", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        others = [s for s in range(4) if s != sub.me]
        sigma = T.lowest_legal_sigma(sub, others)
        v0 = ref.br_solve(sub, sigma, PAY).value
        worlds2 = np.concatenate([toy.worlds, toy.worlds[:1]], axis=0)
        w2 = np.concatenate([toy.weights, [toy.weights[0] * 0.5]])
        w2[0] *= 0.5
        sub2 = ref.build_subgame(toy.root, worlds2, w2)
        v1 = ref.br_solve(sub2, sigma, PAY).value
        if abs(v0 - v1) > 1e-12:
            bad.append(f"{name}: {v0} vs {v1}")
    _gate(not bad, "R5 weight-split invariance",
          "; ".join(bad) if bad else "2 toys, 1e-12")


def main() -> int:
    gate_R1()
    gate_R2()
    gate_R3()
    gate_R4()
    gate_R5()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall reference gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

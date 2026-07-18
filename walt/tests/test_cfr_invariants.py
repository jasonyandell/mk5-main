#!/usr/bin/env python
"""walt/tests/test_cfr_invariants.py — CFR+ structural gates (V3, V4, V5).

Plain executable script (house style): one PASS/FAIL line per gate, exit
nonzero on any FAIL. Run from the repo root:

    /Users/jason/code/mk5-main/.venv/bin/python -u walt/tests/test_cfr_invariants.py

Gates:
  V3 RM+ invariants   — regrets non-negative after clamp; the average profile
                        is a valid distribution on EVERY info set; each info
                        set's move support equals the legality mask recomputed
                        independently from (hand, public node) via walt.tables.
  V4 seat symmetry    — moving the bid to the other team flips every seat's
                        sign and maps the game value v -> 42 - v (exactly for
                        the enumerated LP values, within CFR tolerance for the
                        solved profiles).
  V5 no impossible-good — exact BR against the CFR average profile can never
                        beat the impossibility bounds: at least the game value
                        in 2p-izable toys, at least the profile's own
                        self-play value everywhere (catches sign bugs).
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

from walt.kernel import reference as ref  # noqa: E402
from walt.kernel import toys as T  # noqa: E402
from walt.kernel.cfr import cfr_solve  # noqa: E402
from walt.tables import get_luts  # noqa: E402

PAY = T.payoff_points()
_failures: list[str] = []


def _gate(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _led_tile_at(sub, node) -> int | None:
    """Independently replay a node key through the public-state algebra to
    recover the led tile there."""
    luts = get_luts(sub.decl_id)
    pub = sub.pub0
    for tile in node:
        actor = (pub.leader + pub.n_in_trick) % 4
        pub = ref._pub_step(pub, actor, int(tile), luts, sub.bid_team)
    return pub.led_tile


def gate_V3() -> None:
    bad = []
    checked_isets = 0
    for name in ("t2_def_w3", "t2_decl_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        res = cfr_solve(sub, PAY, iters=60, br_every=20, impl=ref, debug=True)
        t = res.debug["tree"]
        reg = res.debug["reg"]
        asig = res.debug["avg_sig"]
        if (reg < 0).any():
            bad.append(f"{name}: negative regret after RM+ clamp")
        for i in range(res.debug["n_isets"]):
            lo = int(t.off[i])
            moves = t.i_moves[i]
            probs = asig[lo:lo + len(moves)]
            checked_isets += 1
            if abs(float(probs.sum()) - 1.0) > 1e-9 or (probs < 0).any():
                bad.append(f"{name}: iset {i} not a distribution {probs}")
                break
            if any(moves[k] >= moves[k + 1] for k in range(len(moves) - 1)):
                bad.append(f"{name}: iset {i} moves not ascending")
                break
            led = _led_tile_at(sub, t.i_node[i])
            legal = ref.legal_tiles(t.i_hand[i], led, sub.decl_id)
            if tuple(moves) != legal:
                bad.append(f"{name}: iset {i} moves {moves} != legality {legal}")
                break
        # the exported profile must be byte-for-byte the debug average
        for i in range(res.debug["n_isets"]):
            key = (t.i_seat[i], t.i_hand[i], t.i_node[i])
            moves, probs = res.profile.dist(*key)
            lo = int(t.off[i])
            if tuple(moves) != tuple(t.i_moves[i]) or \
                    not np.array_equal(probs, asig[lo:lo + len(moves)]):
                bad.append(f"{name}: exported profile drifts from average at {key}")
                break
    _gate(not bad, "V3 RM+ invariants (regret clamp, distributions, legality)",
          "; ".join(bad) if bad else f"2 toys, {checked_isets} info sets")


def _mirror_root(root):
    """Move the bid to the other team (bidder -> next seat). Play dynamics are
    unchanged; only the payoff orientation flips. bids are rotated to keep the
    tuple plausible (nothing net-free reads them)."""
    b2 = (int(root.bidder) + 1) % 4
    bids = list(root.bids)
    bids[b2], bids[int(root.bidder)] = bids[int(root.bidder)], bids[b2]
    return dataclasses.replace(root, bidder=b2, bids=tuple(bids))


def gate_V4() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w3", "t2_def_w12"):
        toy = T.get_toy(name)
        root2 = _mirror_root(toy.root)
        sub1 = toy.build()
        sub2 = ref.build_subgame(root2, toy.worlds, toy.weights)
        for u in range(4):
            if ref.sign_of(u, sub1.bid_team) != -ref.sign_of(u, sub2.bid_team):
                bad.append(f"{name}: sign of seat {u} did not flip")

        vals = []
        for sub in (sub1, sub2):
            u, v = sub.me, (sub.me + 1) % 4
            others = [s for s in range(4) if s not in (u, v)]
            sigma = T.lowest_legal_sigma(sub, others)
            pinned = {s: sigma for s in others}
            isets = T.discover_free_infosets(sub, {u, v}, pinned)
            pu = T.pure_profiles(isets, u)
            pv = T.pure_profiles(isets, v)
            M = T.joint_value_matrix(sub, PAY, u, pu, v, pv, pinned)
            lp = T.zero_sum_value(M, ref.sign_of(u, sub.bid_team))
            res = cfr_solve(sub, PAY, iters=200, br_every=25, target_gap=1e-3,
                            impl=ref, pinned=pinned)
            vals.append((lp, res.value))
        (lp1, c1), (lp2, c2) = vals
        lines.append(f"{name}: LP {lp1:.4f}+{lp2:.4f}, CFR {c1:.4f}+{c2:.4f}")
        if abs(lp1 + lp2 - 42.0) > 1e-9:
            bad.append(f"{name}: LP values {lp1}+{lp2} != 42")
        if abs(c1 + c2 - 42.0) > 2e-2:
            bad.append(f"{name}: CFR values {c1}+{c2} != 42")
    _gate(not bad, "V4 team relabel flips signs, maps value v -> 42 - v",
          "; ".join(bad) if bad else " ".join(lines))


def gate_V5() -> None:
    bad = []
    # (a) 2p-izable: BR against the CFR average can never sit below the game
    # value in the deviator's orientation (maxmin bound; catches sign bugs).
    for name in ("t2_decl_w3", "t2_def_w3", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        u, v = sub.me, (sub.me + 1) % 4
        others = [s for s in range(4) if s not in (u, v)]
        sigma = T.lowest_legal_sigma(sub, others)
        pinned = {s: sigma for s in others}
        isets = T.discover_free_infosets(sub, {u, v}, pinned)
        pu = T.pure_profiles(isets, u)
        pv = T.pure_profiles(isets, v)
        M = T.joint_value_matrix(sub, PAY, u, pu, v, pv, pinned)
        lp = T.zero_sum_value(M, ref.sign_of(u, sub.bid_team))
        res = cfr_solve(sub, PAY, iters=150, br_every=25, impl=ref,
                        pinned=pinned)
        for hero in (u, v):
            sgn = ref.sign_of(hero, sub.bid_team)
            brv = ref.br_solve(sub, res.profile, PAY, hero=hero).value
            if sgn * brv < sgn * lp - 1e-6:
                bad.append(f"{name}: seat {hero} BR {brv} beats the game-value "
                           f"bound {lp} (impossible)")
    # (b) everywhere: BR of any seat against the average profile is at least
    # the profile's own self-play value in that seat's orientation.
    for name in ("t2_decl_w12", "t2_def_w3"):
        toy = T.get_toy(name)
        sub = toy.build()
        res = cfr_solve(sub, PAY, iters=60, br_every=20, impl=ref)
        for hero in range(4):
            sgn = ref.sign_of(hero, sub.bid_team)
            brv = ref.br_solve(sub, res.profile, PAY, hero=hero).value
            if sgn * brv < sgn * res.value - 1e-9:
                bad.append(f"{name}: seat {hero} BR {brv} below self-play "
                           f"value {res.value} (impossible)")
    _gate(not bad, "V5 exact BR never impossibly good vs CFR average",
          "; ".join(bad) if bad else "3 pinned toys x2 seats + 2 full toys x4 seats")


def main() -> int:
    gate_V3()
    gate_V4()
    gate_V5()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall invariant gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

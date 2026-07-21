#!/usr/bin/env python
"""Exact DP world-sampler gates, including the Metal path."""
from __future__ import annotations

import sys

import numpy as np

from hoyt import toys as T
from hoyt.worldsample import WorldSampler
from walt.contracts import EndgameRoot
from walt.worlds import enumerate_worlds

_failures = []


def _gate(ok, name, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}"
          f"{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _rows(a):
    return {tuple(map(int, row)) for row in np.asarray(a)}


def gate_W1():
    bad = []
    for toy in T.all_toys():
        exact = enumerate_worlds(toy.root)
        sampler = WorldSampler.from_root(toy.root)
        if sampler.n_worlds != len(exact):
            bad.append(f"{toy.name}:{sampler.n_worlds}!={len(exact)}")
    _gate(not bad, "W1 DP count equals exact enumeration on all toys",
          "; ".join(bad) if bad else "8 roots")


def gate_W2():
    root = T.get_toy("t2_decl_w12").root
    exact = _rows(enumerate_worlds(root))
    sampler = WorldSampler.from_root(root)
    host = sampler.sample(4000, seed=123)
    metal = sampler.sample_metal(4000, seed=456)
    valid = all(row in exact for row in _rows(host)) \
        and all(row in exact for row in _rows(metal))
    complete = _rows(host) == exact and _rows(metal) == exact
    _gate(valid and complete, "W2 host and Metal samples are physical",
          f"population {len(exact)}, host/metal support complete")


def gate_W3():
    root = T.get_toy("t2_decl_w12").root
    sampler = WorldSampler.from_root(root)
    exact = sorted(_rows(enumerate_worlds(root)))
    draw = sampler.sample_metal(24000, seed=789)
    idx = {w: i for i, w in enumerate(exact)}
    count = np.zeros(len(exact), dtype=int)
    for row in draw:
        count[idx[tuple(map(int, row))]] += 1
    expected = len(draw) / len(exact)
    max_rel = float(np.max(np.abs(count - expected)) / expected)
    _gate(max_rel < 0.15, "W3 Metal sampler empirical uniformity",
          f"max relative cell deviation {max_rel:.3f}")


def gate_W4():
    common = dict(decl_id=7, bidder=3, bid_value=38,
                  bids=(0, 0, 0, 38), dealer=0, current_trick=())
    h5 = EndgameRoot(
        **common, me=3, my_hand=(1, 11, 13, 19, 25),
        play_history=((3, 26), (0, 24), (1, 14), (2, 23),
                      (1, 3), (2, 12), (3, 5), (0, 4)),
        trick_leader=3, team_points=(0, 2))
    h6 = EndgameRoot(
        **common, me=1, my_hand=(2, 3, 7, 8, 15, 17),
        play_history=((3, 26), (0, 24), (1, 14), (2, 23)),
        trick_leader=1, team_points=(0, 1))
    got = (WorldSampler.from_root(h5).n_worlds,
           WorldSampler.from_root(h6).n_worlds)
    _gate(got == (324_324, 17_153_136),
          "W4 registered H5/H6 populations fit the DP",
          f"{got[0]:,} / {got[1]:,} worlds")


def main():
    gate_W1()
    gate_W2()
    gate_W3()
    gate_W4()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall world-sampler gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

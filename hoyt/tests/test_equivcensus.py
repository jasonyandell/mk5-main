"""Certification gates for the equivalence-census instrument.

E1 (index relabeling): canonical keys are invariant under relabeling a
   signature's tile indices — certifies refinement + backtracking.
E2 (planted symmetry): a hand-built structure with a known swap symmetry
   yields exactly |Aut| = 2.
E3 (value-tie receipt): census-found interchangeable my-tile pairs are
   value-tied through br_solve vs the uniform field (bar 1e-9; measured
   0.0 on all 29 evalset pairs, 2026-07-20).
E4 (teeth): a root with no symmetry reports no pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hoyt.equivcensus import (
    Signature,
    _root_of,
    automorphisms,
    canonical_key,
    interchangeable_pairs,
    value_tie_receipt,
)

EVALSET = Path(__file__).parent.parent / "evalset_h4_v1.jsonl"


def _recs():
    return {json.loads(x)["seed"]: json.loads(x) for x in open(EVALSET)}


class _Toy:
    pass


def _permuted(sig, pi):
    out = _Toy()
    idx = np.array(pi)
    out.n = sig.n
    out.colors0 = [sig.colors0[p] for p in pi]
    out.C = sig.C[np.ix_(idx, idx)]
    out.Q = sig.Q[np.ix_(idx, idx)]
    return out


def test_e1_index_relabel_invariance():
    recs = _recs()
    rng = np.random.default_rng(7)
    for seed in (555001, 555009, 555132):
        root = _root_of(recs[seed]["root"])
        sig = Signature(root)
        k0 = canonical_key(sig, sig.header_P)
        pi = list(rng.permutation(sig.n))
        assert canonical_key(_permuted(sig, pi), sig.header_P) == k0


def test_e2_planted_symmetry():
    toy = _Toy()
    toy.n = 4
    toy.colors0 = [(1, 0, (0, 0, 0))] * 4
    toy.C = np.array([[1, 1, 0, 0], [1, 1, 0, 0],
                      [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int8)
    toy.Q = np.array([[1, 2, 0, 0], [1, 2, 0, 0],
                      [0, 0, 1, 2], [0, 0, 1, 2]], dtype=np.int8)
    auts = automorphisms(toy)
    assert sorted(auts) == [(0, 1, 2, 3), (2, 3, 0, 1)]


def test_e3_value_tie_receipt():
    recs = _recs()
    root = _root_of(recs[555009]["root"])
    pairs = interchangeable_pairs(root)
    assert pairs == {(5, 14)}  # 2-2 ~ 4-4 under doubles-trump
    ties = value_tie_receipt(root, cap=16)
    assert ties and all(dv <= 1e-9 for _a, _b, dv in ties)


def test_e4_no_symmetry_no_pairs():
    recs = _recs()
    root = _root_of(recs[555001]["root"])
    assert interchangeable_pairs(root) == set()

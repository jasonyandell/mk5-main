"""Gate R6 — the game algebra's symmetry, frozen.

Measured 2026-07-22 (wiki topics/forty-two-native-object): of the 5,039
nontrivial pip relabelings, exactly ONE is a game isomorphism — swapping
pips 2 and 3, which transports the twos-game onto the threes-game (and
survives only because the 2-3 tile is trump in both, hiding its
higher-end lead orientation). Count marks + the higher-end lead rule
kill everything else, including 2<->3 in every other declaration.

This is a mechanical property of the rules and must never drift: the
whole structural-unit program leans on "identity is contract-entangled;
symmetry pays nothing" being a fact, not an impression.
"""
from itertools import permutations

import numpy as np

from forge.oracle.tables import DOMINOES, DOMINO_COUNT_POINTS
from walt.tables import get_luts

_TID = {frozenset(d): i for i, d in enumerate(DOMINOES)}
_PIP_DECLS = tuple(range(7))
_DT, _NT = 7, 9
_ALL_DECLS = _PIP_DECLS + (_DT, _NT)


def _perm_tiles(sig):
    return np.array([_TID[frozenset({sig[h], sig[l]})]
                     for h, l in DOMINOES])


def _is_iso(sig, d) -> bool:
    """sig transports the decl-d game onto its image decl: counts, led
    suits, follow sets, and every within-context rank ORDER preserved."""
    pi = _perm_tiles(sig)
    d2 = sig[d] if d in _PIP_DECLS else d
    A, B = get_luts(d), get_luts(d2)
    cnt = np.array(DOMINO_COUNT_POINTS)
    if not np.array_equal(cnt[pi], cnt):
        return False
    ls_map = [sig[p] for p in range(7)] + [7]
    for t in range(28):
        if ls_map[A.led_suit[t]] != B.led_suit[pi[t]]:
            return False
    for ls in range(8):
        ls2 = ls_map[ls]
        if not np.array_equal(A.can_follow[ls], B.can_follow[ls2][pi]):
            return False
        ra = A.rank[ls].astype(int)
        rb = B.rank[ls2].astype(int)[pi]
        oa = np.sign(ra[:, None] - ra[None, :])
        ob = np.sign(rb[:, None] - rb[None, :])
        if not np.array_equal(oa, ob):
            return False
    return True


def test_r6_symmetry_group_is_one_arrow():
    ident = tuple(range(7))
    swap23 = (0, 1, 3, 2, 4, 5, 6)
    cnt = np.array(DOMINO_COUNT_POINTS)
    # count preservation is the cheap filter: it must kill every sigma
    # except identity and 2<->3
    survivors = [sig for sig in permutations(range(7))
                 if np.array_equal(cnt[_perm_tiles(sig)], cnt)]
    assert set(survivors) == {ident, swap23}
    # identity transports every decl; 2<->3 transports ONLY decls 2 and 3
    assert all(_is_iso(ident, d) for d in _ALL_DECLS)
    assert {d for d in _ALL_DECLS if _is_iso(swap23, d)} == {2, 3}

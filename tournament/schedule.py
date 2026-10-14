"""Which faceoff next: the one that teaches the rating table the most.

One faceoff between i and j adds about w = games · p(1 − p) of Fisher
information along x = e_i − e_j, where p is the current P(i beats j). The
rating covariance V then shrinks by a rank-one update, and its trace (the
total squared uncertainty in the table) falls by

    Δ(i, j) = w · xᵀV²x / (1 + w · xᵀVx).

The scheduler plays the pair with the largest Δ (A-optimal design, one step
ahead). It is the formal version of "the pair whose intervals overlap the
most": Δ is large when the two are close (p near ½, so a game is informative)
and their difference is uncertain (xᵀVx large), and it also counts how much
that pair's result would sharpen everyone else's rating through the
covariance. V is centred (θ minus its mean), so the choice does not depend on
which player anchors the scale.

A pair never played sits at the prior, so new participants are scheduled
first, against whoever their result would tell the most about.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .rating import Fit


@dataclass(frozen=True)
class Choice:
    a: str
    b: str
    gain: float


def gains(fit: Fit, games_per_faceoff: int, allowed: Iterable[str]) -> list[Choice]:
    """Δ for every pair of allowed players, best first (ties by name)."""
    allowed = sorted(set(allowed) & set(fit.players))
    n = len(fit.players)
    m = np.eye(n) - np.ones((n, n)) / n
    v = m @ fit.cov @ m
    v2 = v @ v
    out = []
    for i, j in combinations(allowed, 2):
        a, b = fit.players.index(i), fit.players.index(j)
        x = np.zeros(n)
        x[a], x[b] = 1.0, -1.0
        p = fit.p_win(i, j)
        w = games_per_faceoff * p * (1.0 - p)
        out.append(Choice(i, j, float(w * (x @ v2 @ x) / (1.0 + w * (x @ v @ x)))))
    out.sort(key=lambda c: (-round(c.gain, 12), c.a, c.b))
    return out


def next_pair(fit: Fit, games_per_faceoff: int, allowed: Iterable[str],
              rng: random.Random | None = None, within: float = 0.8) -> Choice | None:
    """The best pair, or, given an rng, a uniform pick among the pairs whose
    gain is at least `within` of the best. Workers read the same records, so
    without the pick they would all play the same pair at once; with it they
    spread over the near-best ones, and each still teaches nearly the most."""
    ranked = gains(fit, games_per_faceoff, allowed)
    if not ranked or rng is None:
        return ranked[0] if ranked else None
    near = [c for c in ranked if c.gain >= within * ranked[0].gain]
    return rng.choice(near)

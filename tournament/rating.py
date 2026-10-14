"""Ratings, derived: a Bradley–Terry fit over every game in the faceoff records.

Nothing here is stored. `fit` is a pure function of the games, so the table is
recomputed from the immutable records every time and two readers of the same
records get the same numbers in any order.

The model. Player i has strength θ_i; a game between i and j goes to i with

    P(i beats j) = σ(θ_i − θ_j),   σ(x) = 1 / (1 + e^{−x}).

θ is fitted by maximum a posteriori over every game, with a weak Gaussian prior
θ_i ~ N(0, PRIOR_SD²) that keeps a player who has won every game so far at a
finite rating (the MLE would be +∞) and is otherwise swamped by data. The
posterior is strictly concave, so Newton's method finds the one maximum.

Why not online Elo. Online Elo depends on the order the games arrive in, and
two workers append in an order nobody chose; the fit over all games does not.
Elo's update is a stochastic-gradient step on this same likelihood, so the
scale is the familiar one: rating = 1500 + (400 / ln 10)·(θ_i − θ_anchor), and a
400-point gap is 10:1 odds.

The intervals. A faceoff plays each deal twice with the seats swapped (a
duplicate pair), so the two games of a pair share their cards and are not
independent. The point estimate is the same either way (a pair's two games
contribute exactly what two separate games would to the likelihood); the
variance is not. It is the cluster-robust sandwich, one cluster per duplicate
pair:

    V = H⁻¹ (B + P) H⁻¹,   B = Σ_pairs g_c g_cᵀ,

with H the posterior's negative Hessian, g_c a pair's score (gradient of its
games' log-likelihood) and P the prior precision. When the games really are
independent B → the Fisher information and V → H⁻¹; when a pair's games
cancel (each side wins one of the two), the pair carries less information
than two games and V says so. Intervals are for θ_i − θ_anchor, the only
identified quantity: ±1.96·√(V_ii + V_aa − 2V_ia).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

PRIOR_SD = 2.0  # natural units: 2.0 is 347 Elo, a prior that a few games overrule
ELO_PER_NAT = 400.0 / math.log(10.0)
ANCHOR_ELO = 1500.0
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Game:
    """One game's result. `cluster` names the duplicate pair it belongs to."""

    a: str
    b: str
    a_won: bool
    cluster: str


@dataclass(frozen=True)
class Rating:
    player: str
    elo: float
    lo: float  # 95% interval on elo (the anchor's is a point)
    hi: float
    games: int
    wins: int


@dataclass(frozen=True)
class Fit:
    players: tuple[str, ...]
    anchor: str
    theta: np.ndarray  # natural units, posterior mode
    cov: np.ndarray  # sandwich covariance of theta
    ratings: tuple[Rating, ...]  # sorted best first

    def diff_var(self, i: str, j: str) -> float:
        a, b = self.players.index(i), self.players.index(j)
        return float(self.cov[a, a] + self.cov[b, b] - 2.0 * self.cov[a, b])

    def p_win(self, i: str, j: str) -> float:
        a, b = self.players.index(i), self.players.index(j)
        return float(_sigmoid(self.theta[a] - self.theta[b]))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit(
    games: Iterable[Game],
    players: Sequence[str],
    anchor: str,
    *,
    prior_sd: float = PRIOR_SD,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Fit:
    """The posterior mode and its sandwich covariance over `players`.

    Every player is rated, with or without games (one without games sits at
    the prior, with the prior's width). `anchor` is pinned to ANCHOR_ELO.
    """
    players = tuple(sorted(set(players)))
    if anchor not in players:
        raise ValueError(f"anchor {anchor!r} is not a player")
    index = {p: k for k, p in enumerate(players)}
    games = sorted(games, key=lambda g: (g.cluster, g.a, g.b, g.a_won))
    for g in games:
        if g.a not in index or g.b not in index:
            raise ValueError(f"game between unknown players {g.a!r}, {g.b!r}")
        if g.a == g.b:
            raise ValueError(f"a player cannot face itself: {g.a!r}")

    n = len(players)
    ia = np.array([index[g.a] for g in games], dtype=np.int64)
    ib = np.array([index[g.b] for g in games], dtype=np.int64)
    y = np.array([1.0 if g.a_won else 0.0 for g in games])
    prior_prec = np.eye(n) / prior_sd**2

    def grad_hess(theta):
        p = _sigmoid(theta[ia] - theta[ib]) if len(games) else np.zeros(0)
        r = y - p  # d loglik / d (θ_a − θ_b), per game
        g = -prior_prec @ theta
        np.add.at(g, ia, r)
        np.subtract.at(g, ib, r)
        w = p * (1.0 - p)
        h = prior_prec.copy()
        np.add.at(h, (ia, ia), w)
        np.add.at(h, (ib, ib), w)
        np.subtract.at(h, (ia, ib), w)
        np.subtract.at(h, (ib, ia), w)
        return g, h, r

    theta = np.zeros(n)
    for _ in range(max_iter):
        g, h, _ = grad_hess(theta)
        step = np.linalg.solve(h, g)
        theta = theta + step
        if float(np.max(np.abs(step), initial=0.0)) < tol:
            break
    _, h, r = grad_hess(theta)

    # Meat: per-cluster score outer products, plus the prior's precision.
    clusters: dict[str, np.ndarray] = {}
    for k, gm in enumerate(games):
        s = clusters.setdefault(gm.cluster, np.zeros(n))
        s[ia[k]] += r[k]
        s[ib[k]] -= r[k]
    meat = prior_prec.copy()
    for s in clusters.values():
        meat += np.outer(s, s)
    h_inv = np.linalg.inv(h)
    cov = h_inv @ meat @ h_inv

    a = index[anchor]
    counts = {p: [0, 0] for p in players}
    for gm in games:
        counts[gm.a][0] += 1
        counts[gm.b][0] += 1
        counts[gm.a if gm.a_won else gm.b][1] += 1
    ratings = []
    for p, k in index.items():
        d = theta[k] - theta[a]
        sd = math.sqrt(max(0.0, cov[k, k] + cov[a, a] - 2.0 * cov[k, a]))
        elo = ANCHOR_ELO + ELO_PER_NAT * d
        half = ELO_PER_NAT * Z95 * sd
        ratings.append(Rating(p, elo, elo - half, elo + half, counts[p][0], counts[p][1]))
    ratings.sort(key=lambda r: (-r.elo, r.player))
    return Fit(players, anchor, theta, cov, tuple(ratings))


def paired_margin(diffs: Sequence[float]) -> tuple[float, float, float]:
    """Mean of per-pair values with a 95% t-free normal interval (the house
    style: marks/game, A minus B, one value per duplicate pair)."""
    n = len(diffs)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    m = sum(diffs) / n
    if n == 1:
        return (m, math.nan, math.nan)
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    half = Z95 * sd / math.sqrt(n)
    return (m, m - half, m + half)

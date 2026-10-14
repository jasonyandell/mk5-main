"""Match equity for the game to seven marks, and the bid it prices.

The match equity V(a, b, d) is the probability that "we" win the game from
a marks to b, with d = 1 when our side shakes first this hand: the
fixed point of one hand's law, solved exactly by dynamic programming over
the score grid (every hand moves at least one mark, so the recursion is a
finite DAG, no iteration). This is `champion/utility.py`'s `race_wp` with
the hand's law measured instead of a coin, and the shaker carried (the
dealer alternates teams, and a pass-out reshake moves it once more).

The bid that follows from it (wiki/topics/match-equity.md):

  * Every contract from 30 to 42 pays one mark, made or set. If every
    branch of a decision moves exactly one mark, E[V] = V(a, b+1) + π·Δ with
    Δ = V(a+1, b) − V(a, b+1) > 0 and π the chance our side takes the mark:
    the score multiplies the answer and never changes it. Choosing among
    one-mark calls is a hand question at every score.
  * The score enters only where the marks at stake differ: 84 and up. So
    the match-aware bidder is the hand-only bidder plus one more option at
    its turn: a mark bid (84, or one mark over an 84), taken when its match
    equity beats the hand-only call's.

Information state only: the bidder's own forecast (its hand), the auction
so far, the score, the shaker. No hidden hand is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HandLaw:
    """One hand's law from the first shaker's side, as the field plays it:
    `win_flip[w][f]` = P(the first shaker's side takes the mark (w = 1) and
    the next hand's first shaker sits on the other side (f = 1)). Every
    hand under this law moves exactly one mark (no mark bids in the field).
    """

    win_flip: tuple[tuple[float, float], tuple[float, float]]

    @staticmethod
    def coin() -> "HandLaw":
        """The neutral race: a coin per hand, the shaker alternating."""
        return HandLaw(((0.0, 0.5), (0.0, 0.5)))


class MatchEquity:
    """V(a, b, d) under a hand law, to `target` marks."""

    def __init__(self, law: HandLaw, target: int = 7):
        self.law = law
        self.target = target
        self._v = lru_cache(maxsize=None)(self._solve)

    def __call__(self, a: int, b: int, d: int) -> float:
        """Our side at a, theirs at b, d = 1 when we shake first."""
        return self._v(min(a, self.target), min(b, self.target), d)

    def _solve(self, a: int, b: int, d: int) -> float:
        if a >= self.target:
            return 1.0
        if b >= self.target:
            return 0.0
        total = 0.0
        for w in (0, 1):
            for f in (0, 1):
                p = self.law.win_flip[w][f]
                if p == 0.0:
                    continue
                ours = w == d  # the shaker's side took it, and the shaker is us (or neither)
                nd = 1 - d if f else d
                total += p * (self(a + 1, b, nd) if ours else self(a, b + 1, nd))
        return total


@dataclass(frozen=True)
class Odds:
    """What happens to a one-mark call at our turn, as the field bids it,
    by the auction's shape at our turn: `holder` (who holds the high bid:
    "none", "partner", "opp") and `after` (seats still to speak, 0..3).

    `keep[(holder, after, band)]`: P(our point bid below 42 stands to the end
    of the auction), band "low" for 30..34 and "high" for 35..41.
    `over_set`: P(an overcaller over our point bid is set).
    `pass_ours[(holder, after)]`: P(our side takes the hand's mark when we
    pass here; a pass-out's reshake included, its mark is the next deal's).
    A 42 is never overcalled by a field that never bids mark bids."""

    keep: dict[tuple[str, int, str], float]
    over_set: float
    pass_ours: dict[tuple[str, int], float]

    def take(self, call: int, p_make: float, key: tuple[str, int]) -> float:
        """pi: the chance our side takes the hand's one mark after `call`
        (0 = pass) with this hand's make probability for it."""
        if call == 0:
            return self.pass_ours[key]
        keep = 1.0 if call >= 42 else self.keep[(*key, "low" if call < 35 else "high")]
        return keep * p_make + (1.0 - keep) * self.over_set


def one_mark_value(v: MatchEquity, a: int, b: int, nd: int, pi: float) -> float:
    """E[V] after a hand that moves one mark, ours with probability π."""
    return pi * v(a + 1, b, nd) + (1.0 - pi) * v(a, b + 1, nd)


def mark_bid_value(v: MatchEquity, a: int, b: int, nd: int, marks: int, p_make: float) -> float:
    """E[V] after we hold a `marks`-mark contract made with probability p."""
    return p_make * v(a + marks, b, nd) + (1.0 - p_make) * v(a, b + marks, nd)


class Linear:
    """The score-blind yardstick: the expected mark margin (a − b), the
    'marks per hand' a bidder maximizes when it ignores the score."""

    target = 7

    def __call__(self, a: int, b: int, d: int) -> float:
        return float(a - b)

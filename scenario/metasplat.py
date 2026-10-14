"""The metasplat: how Zeb's splat differs from the book's, finer than a mean.

A position's splat is each legal move's distribution of the bidder team's
final margin over the mover's worlds ([[splat-visualizer]]). The book study
priced a book rule by one scalar per position, the book's best pmake less
Zeb's. This module compares the whole distributions instead: at every
position where a book rule fires, Zeb's pick against one contrast move, world
by world (the pour's outcomes are paired: every move is played out in the
same worlds), and names the shape of the difference from a small fixed
vocabulary.

**The pair.** Z is Zeb's pick, the tile the pour played. If Z is outside the
book's allowed set (a *disagreement*), X is the book's best tile; if Z is
inside it (an *agreement*), X is the best tile the book rules out, and a rule
that allows every legal tile has no contrast. "Best" orders by the mover's
pmake, then the mover's mean margin, then the lowest tile id: the book's
strongest case, and the strongest alternative the book forbids.

**Units.** Everything is the mover's: the margin signed toward the mover's
side (m = margin for the bidders, −margin for the setters), success the
contract made for the bidders and set for the setters. Δp is Z's success
probability less X's; Δμ is Z's mean margin less X's, in points.

**The noise bar** is the pour's own sampling: twice the paired standard error
over the position's worlds (`BAR_SE`). Worlds are uniform draws from the
fiber, and both moves are played in the same worlds, so the paired error is
the error of the difference. A cell with fewer than `SAMPLED` worlds is the
whole fiber enumerated (`scenario.fiber`): it has no sampling error and its
bar is zero. The bar does not include the plan's own error (another seed, a
stronger rung), which the book study measured at 0.005 to 0.016 pmake: a
difference outside this bar can still be inside that one.

**The distribution test.** At every margin threshold t (the even margins
−42 … 40), F_X(t) − F_Z(t) is Z's advantage in the mass above t; it is
paired over the worlds like Δp and carries its own bar. Z is *ahead at t*
when the advantage exceeds its bar, *behind at t* when it is below minus its
bar. Δp is this advantage at the contract's threshold.

**The shapes** (fixed 2026-10-13, before any position was classified; the
first that applies):

1. ``identical``: Z and X end at the same margin in every world.
2. ``noise-tie``: Z is neither ahead nor behind at any threshold, and Δμ is
   inside its bar. A tie on noise: no disagreement.
3. ``behind-on-make``: Δp below minus its bar (Zeb's pick makes less).
4. ``dominant``: Δp above its bar, and Z is behind at no threshold: more
   make, nothing given up anywhere (first-order dominance within the bar).
5. ``make-for-margin``: Δp above its bar, Z behind somewhere, and Δμ below
   minus its bar: the safer tail at a lower mean.
6. ``make-for-tail``: Δp above its bar, Z behind somewhere, Δμ not below
   its bar: more make, a tail given up elsewhere, the mean held.
7. ``even-make-mean-up``: Δp inside its bar, Δμ above its bar.
8. ``even-make-mean-down``: Δp inside its bar, Δμ below minus its bar.
9. ``even-make-spread``: Δp and Δμ inside their bars, but Z ahead or behind
   somewhere: the same make and mean, a different spread.

Under the `make` utility the plan values only the make, so shapes 7–9 are
moves the plan is indifferent between (up to its sample), and past a decided
contract the margins are the tape's, not the plan's (issue #107): a margin
difference at an even make is a fact about the continuation, never a
preference of Zeb's.

**The population** is the book study's: every position of the 250,000-deal
pour where a book-faithful rule fires (`book/book-reprice-3`), less the rows
decided by the banked score. **The sample** for the page and the reasons
(also fixed before counting): from the first chunk, for each shape, three
disagreement rows drawn at random (`SAMPLE_SEED`) from three different book
rules where the shape has them, and one agreement row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario.planner import HAND_TOTAL

BAR_SE = 2.0            # the noise bar: this many paired standard errors over the position's worlds
SAMPLED = 512           # a cell this large is a sample of its fiber; a smaller one is the fiber, enumerated
N_THRESHOLDS = HAND_TOTAL  # thresholds t = 2i − 42, i = 0 … 41 (F(42) = 1 for every move)
N_BINS = HAND_TOTAL + 1    # margins are even: bin b holds margin 2b − 42
SAMPLE_SEED = 20261013

SHAPES = (
    "identical", "noise-tie", "behind-on-make", "dominant", "make-for-margin", "make-for-tail",
    "even-make-mean-up", "even-make-mean-down", "even-make-spread",
)
SHAPE = {name: i for i, name in enumerate(SHAPES)}
NO_CONTRAST = -1        # an agreement where the rule allows every legal tile
MAKE_EDGE = (SHAPE["dominant"], SHAPE["make-for-margin"], SHAPE["make-for-tail"])
EVEN_MAKE = (SHAPE["even-make-mean-up"], SHAPE["even-make-mean-down"], SHAPE["even-make-spread"])
TIES = (SHAPE["identical"], SHAPE["noise-tie"])


def success_index(sign: int | np.ndarray, bid: int) -> int | np.ndarray:
    """The threshold index i at which the mover's success is "mover's margin
    > 2i − 42": the bidders make at margin ≥ 2·bid − 42 (> T − 2), the
    setters set at margin < T (mover's margin > −T)."""
    t = 2 * bid - HAND_TOTAL
    t_star = np.where(np.asarray(sign) > 0, t - 2, -t)
    return (t_star + HAND_TOTAL) // 2


@dataclass(frozen=True)
class Contrast:
    """Zeb's pick Z against a contrast X at one position, in the mover's units."""

    dp: float                # Z's success probability less X's
    bar_p: float
    dmu: float               # Z's mean margin less X's, points
    bar_mu: float
    ahead: bool              # Z's mass above some threshold exceeds X's by more than its bar
    behind: bool             # X's exceeds Z's by more than its bar somewhere
    shape: str


def classify(identical: bool, dp: float, bar_p: float, dmu: float, bar_mu: float,
             ahead: bool, behind: bool) -> str:
    """The shape of one difference, by the module's fixed rules (first match)."""
    if identical:
        return "identical"
    if not ahead and not behind and abs(dmu) <= bar_mu:
        return "noise-tie"
    if dp < -bar_p:
        return "behind-on-make"
    if dp > bar_p:
        if not behind:
            return "dominant"
        return "make-for-margin" if dmu < -bar_mu else "make-for-tail"
    if dmu > bar_mu:
        return "even-make-mean-up"
    if dmu < -bar_mu:
        return "even-make-mean-down"
    return "even-make-spread"


def _bar(values_sum: float, squares_sum: float, n: int, exact: bool) -> float:
    if exact or n < 2:
        return 0.0
    mean = values_sum / n
    var = max(squares_sum - n * mean * mean, 0.0) / (n - 1)
    return BAR_SE * float(np.sqrt(var / n))


def contrast(z: np.ndarray, x: np.ndarray, sign: int, bid: int, exact: bool) -> Contrast:
    """The reference reading of one pair: `z` and `x` are the bidder team's
    final margins of Zeb's pick and the contrast in the same worlds (uniform
    weights), `sign` +1 when the mover bids, `exact` when the worlds are the
    whole fiber. Written threshold by threshold; `contrasts` is its batch."""
    mz, mx = sign * np.asarray(z, np.int64), sign * np.asarray(x, np.int64)
    n = len(mz)
    d = mz - mx
    ahead = behind = False
    advantage = {}
    for i in range(N_THRESHOLDS):
        t = 2 * i - HAND_TOTAL
        c = (mx <= t).astype(np.int64) - (mz <= t).astype(np.int64)
        a, bar = c.sum() / n, _bar(c.sum(), (c * c).sum(), n, exact)
        advantage[i] = (a, bar)
        ahead |= a > bar
        behind |= a < -bar
    dp, bar_p = advantage[int(success_index(sign, bid))]
    dmu, bar_mu = d.sum() / n, _bar(d.sum(), (d * d).sum(), n, exact)
    shape = classify(not d.any(), dp, bar_p, dmu, bar_mu, ahead, behind)
    return Contrast(float(dp), float(bar_p), float(dmu), float(bar_mu), bool(ahead), bool(behind), shape)


def contrasts(z: np.ndarray, x: np.ndarray, n: np.ndarray, sign: np.ndarray, bid: int,
              exact: np.ndarray) -> dict[str, np.ndarray]:
    """`contrast` for a batch: `z`, `x` [B, W] bidder-team margins (columns
    past a row's `n` are ignored), `n`, `sign` and `exact` [B]. Each
    threshold's advantage is counted from the worlds where the pair differs:
    Z ahead in a world adds one at every threshold from X's margin up to Z's,
    so the counts are two cumulative histograms."""
    B, W = z.shape
    n = np.asarray(n, np.int64)
    sign = np.asarray(sign, np.int64)
    valid = np.arange(W)[None, :] < n[:, None]
    mz = np.where(valid, sign[:, None] * z.astype(np.int64), 0)
    mx = np.where(valid, sign[:, None] * x.astype(np.int64), 0)
    d = mz - mx
    bz, bx = (mz + HAND_TOTAL) // 2, (mx + HAND_TOTAL) // 2
    rows = np.broadcast_to(np.arange(B)[:, None], (B, W))
    width = N_BINS + 1

    def spans(where: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Per row and threshold i: the worlds in `where` with lo ≤ i < hi."""
        r = rows[where] * width
        diff = (np.bincount(r + lo[where], minlength=B * width)
                - np.bincount(r + hi[where], minlength=B * width))
        return np.cumsum(diff.reshape(B, width), axis=1)[:, :N_THRESHOLDS]

    up, down = spans(d > 0, bx, bz), spans(d < 0, bz, bx)
    nf = n.astype(np.float64)[:, None]
    adv = (up - down) / nf
    var = np.maximum((up + down) - nf * adv * adv, 0.0) / np.maximum(nf - 1, 1)
    sampled = (~np.asarray(exact, bool) & (n >= 2)).astype(np.float64)[:, None]
    bars = BAR_SE * np.sqrt(var / nf) * sampled
    ahead, behind = (adv > bars).any(axis=1), (adv < -bars).any(axis=1)
    i_star = success_index(sign, bid)
    dp, bar_p = adv[np.arange(B), i_star], bars[np.arange(B), i_star]
    s1, s2 = d.sum(axis=1).astype(np.float64), (d * d).sum(axis=1).astype(np.float64)
    dmu = s1 / nf[:, 0]
    var_mu = np.maximum(s2 - nf[:, 0] * dmu * dmu, 0.0) / np.maximum(nf[:, 0] - 1, 1)
    bar_mu = BAR_SE * np.sqrt(var_mu / nf[:, 0]) * sampled[:, 0]
    identical = ~(d != 0).any(axis=1)

    shape = np.full(B, SHAPE["even-make-spread"], np.int8)
    even = (np.abs(dp) <= bar_p)
    shape[even & (dmu < -bar_mu)] = SHAPE["even-make-mean-down"]
    shape[even & (dmu > bar_mu)] = SHAPE["even-make-mean-up"]
    edge = dp > bar_p
    shape[edge & behind & ~(dmu < -bar_mu)] = SHAPE["make-for-tail"]
    shape[edge & behind & (dmu < -bar_mu)] = SHAPE["make-for-margin"]
    shape[edge & ~behind] = SHAPE["dominant"]
    shape[dp < -bar_p] = SHAPE["behind-on-make"]
    shape[~ahead & ~behind & (np.abs(dmu) <= bar_mu)] = SHAPE["noise-tie"]
    shape[identical] = SHAPE["identical"]
    return dict(dp=dp, bar_p=bar_p, dmu=dmu, bar_mu=bar_mu, ahead=ahead, behind=behind, shape=shape)


def best_action(success: np.ndarray, margin_sum: np.ndarray, tiles: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Per row, the column of the best allowed action: the most mover's
    successes, then the largest mover's margin sum, then the lowest tile id;
    −1 where nothing is allowed. All arguments [R, A] (integers; `allowed`
    bool, False on padding)."""
    key = (success.astype(np.int64) * (2 * HAND_TOTAL * SAMPLED + 1)
           + (margin_sum.astype(np.int64) + HAND_TOTAL * SAMPLED)) * 32 + (31 - tiles.astype(np.int64))
    key = np.where(allowed, key, -1)
    best = key.argmax(axis=1)
    return np.where(allowed.any(axis=1), best, -1)


def fiber_log2(pool: np.ndarray, sizes: np.ndarray, forbid: np.ndarray) -> np.ndarray:
    """log2 of the number of the mover's worlds, |fiber| (`scenario.fiber`),
    for a batch: `pool` [B] the tiles hidden from the mover (masks), `sizes`
    [B, 3] the hidden seats' hand sizes, `forbid` [B, 3] the tiles each
    provably lacks. Every hidden tile goes to one seat that may hold it: the
    count is the coefficient of x0^c0 x1^c1 x2^c2 in the product, over the
    hidden tiles, of the sum of x_s over the seats s allowed it."""
    pool = np.asarray(pool, np.int64)
    sizes = np.asarray(sizes, np.int64)
    forbid = np.asarray(forbid, np.int64)
    B = len(pool)
    cap = 8
    poly = np.zeros((B, cap, cap, cap))
    poly[:, 0, 0, 0] = 1.0
    for t in range(28):
        here = ((pool >> t) & 1).astype(bool)
        if not here.any():
            continue
        new = np.zeros_like(poly)
        for s in range(3):
            ok = (here & ~((forbid[:, s] >> t) & 1).astype(bool)).astype(np.float64)
            shifted = np.zeros_like(poly)
            if s == 0:
                shifted[:, 1:] = poly[:, :-1]
            elif s == 1:
                shifted[:, :, 1:] = poly[:, :, :-1]
            else:
                shifted[:, :, :, 1:] = poly[:, :, :, :-1]
            new += ok[:, None, None, None] * shifted
        poly = np.where(here[:, None, None, None], new, poly)
    count = poly[np.arange(B), sizes[:, 0], sizes[:, 1], sizes[:, 2]]
    if np.any(count < 1):
        raise AssertionError("an empty fiber: the record contradicts itself")
    return np.log2(count)

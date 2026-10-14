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
the error of the difference. A cell with fewer worlds than the pour samples
(its spec's n, 512 here) is the whole fiber enumerated (`scenario.fiber`): it
has no sampling error and its bar is zero. The bar does not include the
plan's own error (another seed, a stronger rung), which the book study
measured at 0.005 to 0.016 pmake: a difference outside this bar can still be
inside that one.

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

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from scenario.planner import HAND_TOTAL
from scenario.record import empty_record
from scenario.rules import ALL_TILES, rules_for

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


# -- the pour, read against the book ---------------------------------------------------

ROLES = ("bidder", "left setter", "partner", "right setter")   # seat relative to the bidder: +0, +1, +2, +3
POUR_COLUMNS = ("hands", "decl", "plays", "banked", "deal", "n_plays", "n_actions", "n_worlds", "actions", "q",
                "outcomes")
BATCH = 50_000


def read_pour(pour_dir: str | Path) -> dict[str, np.ndarray]:
    """A pour's columns, without the worlds (5 GB a chunk, not needed to
    compare splats). The weights must be uniform: the shapes assume it."""
    with np.load(Path(pour_dir) / "pour.npz") as z:
        if not np.all(z["weights"] == 1):
            raise ValueError("a pour with unequal world weights: the shapes read uniform worlds")
        return {c: z[c] for c in POUR_COLUMNS}


def read_reprice(reprice_dir: str | Path, label: str) -> tuple[dict[str, np.ndarray], list[str], list[list[str]]]:
    """The book re-price's per-position output for one chunk, its parts in
    order: every book rule's and gate's fired rule and allowed set at every
    planned position (`book/book-reprice-3/out/{label}.p{part}`)."""
    parts, meta = [], None
    for f in sorted(Path(reprice_dir).glob(f"{label}.p*.npz")):
        m = json.loads(f.with_suffix(".json").read_text())
        if meta is not None and (m["policies"], m["rules"]) != (meta["policies"], meta["rules"]):
            raise AssertionError(f"{f.name}: the policies differ from the other parts'")
        meta = m
        with np.load(f) as z:
            parts.append({k: z[k] for k in z.files})
    if not parts or meta is None:
        raise FileNotFoundError(f"no {label}.p*.npz under {reprice_dir}")
    arr = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    return arr, meta["policies"], meta["rules"]


def decided(banked: np.ndarray, bid: int) -> np.ndarray:
    """Rows decided by the banked score (the book study drops them): the
    bidders have the bid, or the setters have set it."""
    return (banked[:, 0] >= bid) | (banked[:, 1] > HAND_TOTAL - bid)


def action_stats(p: dict[str, np.ndarray], bid: int) -> tuple[np.ndarray, np.ndarray]:
    """Per action of the pour: the worlds where the bidders make, and the
    sum of the bidder team's margins over its worlds. Checks that `q` is
    the make share (the pour planned under `make`)."""
    na, nw = p["n_actions"].astype(np.int64), p["n_worlds"].astype(np.int64)
    per_action = np.repeat(nw, na)
    starts = np.concatenate(([0], np.cumsum(per_action)[:-1]))
    o = p["outcomes"]
    make = np.add.reduceat(o >= 2 * bid - HAND_TOTAL, starts, dtype=np.int64)
    total = np.add.reduceat(o, starts, dtype=np.int64)
    if not np.allclose(make / per_action, p["q"], rtol=0, atol=1e-12):
        raise AssertionError("q is not its splat's make share: the pour's utility is not make")
    return make, total


def position_features(p: dict[str, np.ndarray], positions: np.ndarray, bidder: int) -> dict[str, np.ndarray]:
    """Public facts of each position (pour indices), from its record: the
    mover's seat and place in the trick, the count on the table, whether the
    partner is winning the open trick, the banked score, and log2 |fiber|
    (the cell the mover's belief is uniform over)."""
    keys = ("seat", "in_trick", "table", "partner_wins", "banked_bidders", "banked_setters")
    out = {k: np.zeros(len(positions), np.int16) for k in keys}
    pool = np.zeros(len(positions), np.int64)
    sizes = np.zeros((len(positions), 3), np.int64)
    forbid = np.zeros((len(positions), 3), np.int64)
    by_deal: dict[int, dict[int, int]] = {}
    for j, q in enumerate(positions):
        by_deal.setdefault(int(p["deal"][q]), {})[int(p["n_plays"][q])] = j
    bt = bidder % 2
    for d, want in by_deal.items():
        rules = rules_for(int(p["decl"][d]))
        hands = [int(h) for h in p["hands"][d]]
        rec = empty_record(bidder)
        for k in range(max(want) + 1):
            if k in want:
                j, seat = want[k], rec.seat
                out["seat"][j], out["in_trick"][j] = seat, len(rec.trick)
                out["table"][j] = sum(rules.count[t] for t in rec.trick)
                if rec.trick:
                    ranks = rules.rank[rules.led_suit[rec.trick[0]]]
                    best = max(range(len(rec.trick)), key=lambda x: (ranks[rec.trick[x]], -x))
                    out["partner_wins"][j] = (rec.leader + best) % 4 == (seat + 2) % 4
                out["banked_bidders"][j], out["banked_setters"][j] = rec.banked[bt], rec.banked[1 - bt]
                own = hands[seat] & ~rec.played
                others = [s for s in range(4) if s != seat]
                pool[j] = ALL_TILES & ~rec.played & ~own
                sizes[j] = [rec.hand_size(s) for s in others]
                forbid[j] = [rec.forbid[s] for s in others]
            rec = rec.advance(int(p["plays"][d, k, 1]), rules)
    step = 20_000
    out["log2_fiber"] = np.concatenate(
        [np.zeros(0)] + [fiber_log2(pool[s:s + step], sizes[s:s + step], forbid[s:s + step])
                         for s in range(0, len(positions), step)])
    return out


def chunk_rows(p: dict[str, np.ndarray], book: dict[str, np.ndarray], policies: Sequence[str],
               rules: Sequence[Sequence[str]], bid: int = 30, bidder: int = 0,
               sampled: int = SAMPLED) -> dict[str, np.ndarray]:
    """Every (book rule, position) row of one chunk where a book rule fires
    and the score has not decided the hand: the pair (Zeb's pick, the
    contrast), its shape and statistics, and the position's public facts.
    `book` is `read_reprice`'s rows; each is matched to its pour position
    and checked against it (legal tiles, values, the tile played)."""
    D = len(p["hands"])
    idx = np.full((D, 28), -1, np.int64)
    idx[p["deal"], p["n_plays"]] = np.arange(len(p["deal"]))
    pos = idx[book["deal"], book["k"]]
    if np.any(pos < 0):
        raise AssertionError("a re-price row has no planned position in the pour")
    na = p["n_actions"].astype(np.int64)
    nw = p["n_worlds"].astype(np.int64)
    a_off = np.concatenate(([0], np.cumsum(na)))
    o_off = np.concatenate(([0], np.cumsum(na * nw)))
    make, total = action_stats(p, bid)

    # Per position, its actions padded to 7: tiles, the mover's successes and margin sums.
    cols = np.arange(7)
    valid = cols[None, :] < na[pos][:, None]
    flat = np.where(valid, a_off[pos][:, None] + cols[None, :], 0)
    tiles = np.where(valid, p["actions"][flat].astype(np.int64), -1)
    if not np.array_equal(tiles, book["actions"].astype(np.int64)):
        raise AssertionError("the re-price's legal tiles are not the pour's")
    if not np.array_equal(np.where(valid, p["q"][flat], 0), np.where(valid, book["q"], 0)):
        raise AssertionError("the re-price's values are not the pour's")
    seat = book["seat"].astype(np.int64)
    sign = np.where(seat % 2 == bidder % 2, 1, -1)
    n = nw[pos]
    succ = np.where(valid, np.where(sign[:, None] > 0, make[flat], n[:, None] - make[flat]), -1)
    msum = np.where(valid, sign[:, None] * total[flat], 0)
    chosen = p["plays"][p["deal"][pos], p["n_plays"][pos], 1].astype(np.int64)
    if not np.array_equal(chosen, book["zeb"].astype(np.int64)):
        raise AssertionError("the re-price's tile played is not the pour's")
    zcol = (tiles == chosen[:, None]).argmax(axis=1)
    live = ~decided(book["banked"], bid)

    out: dict[str, list[np.ndarray]] = {}
    for j, name in enumerate(policies):
        if not name.startswith("book:"):
            continue
        r = np.flatnonzero((book["rule"][:, j] >= 0) & live)
        if not len(r):
            continue
        mask = book["allowed"][r, j].astype(np.int64)
        inbook = valid[r] & (((mask[:, None] >> np.where(valid[r], tiles[r], 0)) & 1) == 1)
        agree = inbook[np.arange(len(r)), zcol[r]]
        pool_x = np.where(agree[:, None], valid[r] & ~inbook, inbook)
        xcol = best_action(succ[r], msum[r], tiles[r], pool_x)
        has = xcol >= 0
        xc = np.maximum(xcol, 0)
        nz = n[r].astype(np.float64)
        legal_succ = np.where(valid[r], succ[r], -1)
        rows = dict(
            pos=pos[r], deal=book["deal"][r].astype(np.int32), k=book["k"][r].astype(np.int8),
            policy=np.full(len(r), j, np.int8), rule=book["rule"][r, j].astype(np.int8), allowed=mask.astype(np.uint32),
            agree=agree, z=chosen[r].astype(np.int8), x=np.where(has, tiles[r, xc], -1).astype(np.int8),
            n_legal=na[pos[r]].astype(np.int8), n_book=inbook.sum(axis=1).astype(np.int8),
            n_worlds=n[r].astype(np.int16), exact=n[r] < sampled, sign=sign[r].astype(np.int8), mover=seat[r].astype(np.int16),
            z_best=succ[r, zcol[r]] == legal_succ.max(axis=1),
            dead=legal_succ.max(axis=1) == np.where(valid[r], succ[r], 1 << 20).min(axis=1),
            pz=succ[r, zcol[r]] / nz, mz=msum[r, zcol[r]] / nz,
            px=np.where(has, succ[r, xc] / nz, np.nan), mx=np.where(has, msum[r, xc] / nz, np.nan),
        )
        stats = {k: np.full(len(r), np.nan) for k in ("dp", "bar_p", "dmu", "bar_mu")}
        stats.update(ahead=np.zeros(len(r), bool), behind=np.zeros(len(r), bool),
                     shape=np.full(len(r), NO_CONTRAST, np.int8))
        h = np.flatnonzero(has)
        w = np.arange(int(nw.max()))[None, :]
        for s in range(0, len(h), BATCH):
            b = h[s:s + BATCH]
            pr = pos[r[b]]
            nb = nw[pr][:, None]
            zi = o_off[pr][:, None] + zcol[r[b]][:, None] * nb + np.minimum(w, nb - 1)
            xi = o_off[pr][:, None] + xcol[b][:, None] * nb + np.minimum(w, nb - 1)
            c = contrasts(p["outcomes"][zi], p["outcomes"][xi], nw[pr], sign[r[b]], bid, nw[pr] < sampled)
            for key, v in c.items():
                stats[key][b] = v
        rows.update(stats)
        for key, v in rows.items():
            out.setdefault(key, []).append(v)
    res = {k: np.concatenate(v) for k, v in out.items()}
    upos, inv = np.unique(res["pos"], return_inverse=True)
    feats = position_features(p, upos, bidder)
    exact = nw[upos] < sampled
    if not np.allclose(feats["log2_fiber"][exact], np.log2(nw[upos][exact]), rtol=0, atol=1e-9):
        raise AssertionError("an enumerated cell is not its fiber")
    mover = res.pop("mover")
    for key, v in feats.items():
        res[key] = v[inv]
    if not np.array_equal(res["seat"], mover):
        raise AssertionError("the record's seat to play is not the re-price's")
    return res


def run(pour_dir: str | Path, reprice_dir: str | Path, label: str, out: str | Path,
        deals: int | None = None) -> dict[str, np.ndarray]:
    """One chunk: its rows, saved as `out` (npz) with a json sidecar naming
    the policies and their rules."""
    book, policies, rules = read_reprice(reprice_dir, label)
    if deals is not None:
        book = {k: v[book["deal"] < deals] for k, v in book.items()}
    p = read_pour(pour_dir)
    manifest = json.loads((Path(pour_dir) / "manifest.json").read_text())
    n = re.search(r"(?:^|,)n(\d+)", manifest["spec"])
    sampled = int(n[1]) if n else SAMPLED
    res = chunk_rows(p, book, policies, rules, manifest["bid"], manifest["bidder"], sampled)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    out.with_suffix(".json").write_text(json.dumps(dict(
        label=label, policies=list(policies), rules=[list(r) for r in rules], rows=int(len(res["pos"])),
        shapes=list(SHAPES), bar_se=BAR_SE, sampled=sampled, deals=deals,
        pour={k: manifest.get(k) for k in ("spec", "seed", "commit", "deals", "positions", "utility", "bid")},
    ), indent=1) + "\n")
    return res


# -- the metasplat: the rows, aggregated -------------------------------------------------

Z95 = 1.959964


def load_rows(paths: Sequence[str | Path]) -> tuple[dict[str, np.ndarray], list[str], list[list[str]]]:
    """Chunks' rows concatenated, each deal keyed by its chunk (`cluster`)."""
    parts, meta = [], None
    for i, f in enumerate(paths):
        f = Path(f)
        m = json.loads(f.with_suffix(".json").read_text())
        if meta is not None and (m["policies"], m["rules"], m["shapes"]) != (
                meta["policies"], meta["rules"], meta["shapes"]):
            raise AssertionError(f"{f.name} was read under other policies or shapes")
        meta = m
        with np.load(f) as z:
            part = {k: z[k] for k in z.files}
        part["cluster"] = i * 1_000_000 + part["deal"].astype(np.int64)
        part["chunk"] = np.full(len(part["deal"]), i, np.int8)
        parts.append(part)
    if meta is None:
        raise FileNotFoundError("no rows")
    if list(meta["shapes"]) != list(SHAPES):
        raise AssertionError("the rows were classified under other shapes")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, meta["policies"], meta["rules"]


def shares(shape: np.ndarray, cluster: np.ndarray) -> dict:
    """Each shape's count and share of the rows, with a 95% interval that
    treats a deal's rows as one cluster (the ratio estimator's variance)."""
    n = len(shape)
    out = {"n": int(n)}
    if not n:
        return out
    _, inv = np.unique(cluster, return_inverse=True)
    m = np.bincount(inv).astype(np.float64)
    for i, name in enumerate(SHAPES):
        y = np.bincount(inv, weights=(shape == i).astype(np.float64), minlength=len(m))
        p = y.sum() / n
        se = float(np.sqrt(((y - p * m) ** 2).sum())) / n
        out[name] = dict(count=int(y.sum()), share=float(p), lo=max(0.0, p - Z95 * se), hi=min(1.0, p + Z95 * se))
    return out


def _quartiles(v: np.ndarray) -> list[float] | None:
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [25, 50, 75])] if len(v) else None


def describe_rows(r: dict[str, np.ndarray], sel: np.ndarray) -> dict:
    """The shape shares of a selection of rows, and per shape the quartiles of
    Δp, Δμ and their bars (the size of what each shape trades)."""
    out = shares(r["shape"][sel], r["cluster"][sel])
    size = {}
    for i, name in enumerate(SHAPES):
        s = sel & (r["shape"] == i)
        if s.any():
            size[name] = {k: _quartiles(r[k][s]) for k in ("dp", "dmu", "bar_p", "bar_mu")}
    out["size"] = size
    return out


def _bucket(v: np.ndarray, edges: Sequence[float], names: Sequence[str]) -> np.ndarray:
    return np.asarray(names, dtype=object)[np.searchsorted(np.asarray(edges), v, side="left")]


def axes(r: dict[str, np.ndarray], bid: int = 30, bidder: int = 0) -> dict[str, np.ndarray]:
    """Where a row lives, as named buckets: the trick, the mover's role and
    place in it, the count at risk (on the table, plus the larger count of
    the two tiles compared), the points each side still needs, the belief's
    size, whether the partner is winning the trick, and the dead rows (every
    legal tile makes alike in Zeb's worlds)."""
    count = np.array(rules_for(0).count)
    risk = r["table"] + np.maximum(count[np.maximum(r["z"], 0)], np.where(r["x"] >= 0, count[np.maximum(r["x"], 0)], 0))
    need_b = bid - r["banked_bidders"]
    need_s = HAND_TOTAL - bid + 1 - r["banked_setters"]
    return {
        "trick": np.array([f"trick {t + 1}" for t in range(7)], dtype=object)[r["k"] // 4],
        "role": np.asarray(ROLES, dtype=object)[(r["seat"] - bidder) % 4],
        "place": np.array(["leads", "2nd", "3rd", "last"], dtype=object)[r["in_trick"]],
        "count at risk": _bucket(risk, [0, 5, 10, 15], ["0", "5", "10", "15", "20+"]),
        "bidders need": _bucket(need_b, [10, 20], ["1-10", "11-20", "21-30"]),
        "setters need": _bucket(need_s, [4, 8], ["1-4", "5-8", "9-13"]),
        "log2 worlds": _bucket(r["log2_fiber"], [0, 6, 9, 12, 16, 20],
                               ["0 (one world)", "(0,6]", "(6,9]", "(9,12]", "(12,16]", "(16,20]", ">20"]),
        "partner winning": np.where(r["in_trick"] == 0, "leads", np.where(r["partner_wins"] == 1, "yes", "no")),
        "dead": np.where(r["dead"], "dead", "live"),
    }


def summarize(r: dict[str, np.ndarray], policies: Sequence[str], rules: Sequence[Sequence[str]]) -> dict:
    """The metasplat: shape shares per book rule and per rule inside it, for
    disagreements and agreements apart; across rules by every axis; and what
    the agreements are made of."""
    disagree = r["agree"] == 0
    contrast = r["shape"] >= 0
    res = dict(rows=int(len(r["shape"])), positions=int(len(np.unique(r["cluster"] * 32 + r["k"]))),
               shapes=list(SHAPES), bar_se=BAR_SE,
               disagree=describe_rows(r, disagree), agree=describe_rows(r, ~disagree & contrast),
               no_contrast=int((~contrast).sum()), z_not_best=int((~r["z_best"]).sum()), policy={}, axes={})
    for j, name in enumerate(policies):
        sel = r["policy"] == j
        if not sel.any():
            continue
        entry = dict(n=int(sel.sum()), disagree=describe_rows(r, sel & disagree),
                     agree=describe_rows(r, sel & ~disagree & contrast),
                     no_contrast=int((sel & ~contrast).sum()), rule={})
        for k, rule in enumerate(rules[j]):
            s = sel & (r["rule"] == k)
            if s.any():
                entry["rule"][rule] = dict(n=int(s.sum()), disagree=shares(r["shape"][s & disagree], r["cluster"][s & disagree]),
                                           agree=shares(r["shape"][s & ~disagree & contrast],
                                                        r["cluster"][s & ~disagree & contrast]))
        res["policy"][name.removeprefix("book:")] = entry
    for axis, label in axes(r).items():
        res["axes"][axis] = {}
        for value in sorted(set(label.tolist())):
            s = label == value
            res["axes"][axis][value] = dict(disagree=shares(r["shape"][s & disagree], r["cluster"][s & disagree]),
                                            agree=shares(r["shape"][s & ~disagree & contrast],
                                                         r["cluster"][s & ~disagree & contrast]))
    return res


# -- the sample: positions to look at, with the reason report's facts ------------------------


def pick_sample(r: dict[str, np.ndarray], chunk: int = 0, seed: int = SAMPLE_SEED,
                per_shape: int = 3) -> list[int]:
    """Row indices of the sample (fixed before counting, module docstring):
    within one chunk, per shape, `per_shape` disagreement rows from different
    book rules and one agreement row, drawn at random."""
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    inside = r["chunk"] == chunk
    for code in range(len(SHAPES)):
        rows = np.flatnonzero(inside & (r["shape"] == code) & (r["agree"] == 0))
        seen: set[int] = set()
        for i in rng.permutation(rows):
            if int(r["policy"][i]) not in seen:
                seen.add(int(r["policy"][i]))
                picks.append(int(i))
                if len(seen) == per_shape:
                    break
        agree = np.flatnonzero(inside & (r["shape"] == code) & (r["agree"] == 1))
        if len(agree):
            picks.append(int(rng.choice(agree)))
    return picks


def split_of(pos, z: int, x: int) -> tuple[str, list[int], list[int]]:
    """The worlds that tell Zeb's pick from the contrast at a pour position:
    where one succeeds for the mover and the other fails, or, when they never
    part on the contract, where one ends at a better margin. Returns the
    basis ("make" or "margin") and the two sides (Z's, X's)."""
    oz, ox = pos.outcomes[pos.actions.index(z)], pos.outcomes[pos.actions.index(x)]
    t = 2 * pos.bid - HAND_TOTAL
    sz, sx = (oz >= t) == (pos.sign > 0), (ox >= t) == (pos.sign > 0)
    one, two = np.flatnonzero(sz & ~sx), np.flatnonzero(sx & ~sz)
    if len(one) or len(two):
        return "make", one.tolist(), two.tolist()
    d = pos.sign * (oz.astype(np.int64) - ox.astype(np.int64))
    return "margin", np.flatnonzero(d > 0).tolist(), np.flatnonzero(d < 0).tolist()


def reason_doc(pos, z: int, x: int, seed: int) -> dict:
    """The reason report's separating facts (`scenario.reason.explain`) for
    the split between Zeb's pick and the contrast, read on the pour's own
    worlds: its declared space, threshold and chance level, no re-plan. When
    one side is empty, the other is set against the worlds where the two
    agree, as the report does."""
    from scenario_core import SilasAuthority

    from scenario.planner import Situation
    from scenario.reason import Scenarios, explain, tile_name

    basis, one, two = split_of(pos, z, x)
    columns = ["Zeb's", "the contrast's"]
    if not one and not two:
        return dict(basis=basis, found="identical", columns=columns, sides=[0, 0])
    if not one or not two:
        side = set(one or two)
        rest = [i for i in range(len(pos.weights)) if i not in side]
        one, two, columns = (one, rest, ["Zeb's", "the rest"]) if one else (rest, two, ["the rest", "the contrast's"])
    sit = Situation(decl_id=pos.decl_id, viewer=pos.viewer, bid=pos.bid, bidder=pos.bidder,
                    record=pos.record, plays=pos.plays)
    cell = Scenarios([tuple(int(h) for h in w) for w in pos.worlds], [float(w) for w in pos.weights])
    found, facts, pairs, chance, _ = explain(sit, pos.own, cell, one, two, (z, x),
                                              SilasAuthority.sampled(len(cell.worlds), seed))

    def fact(f) -> dict:
        return dict(gloss=f.candidate.gloss, on=[str(f.on_a), str(f.on_b)], coverage=[str(c) for c in f.coverage],
                    prior=None if f.prior is None else str(f.prior), alike=list(f.alike[:3]))

    return dict(basis=basis, found=found, columns=columns, sides=[len(one), len(two)],
                chance=[round(c, 4) for c in chance], facts=[fact(f) for f in facts[:3]],
                pairs=[fact(f) for f in pairs[:2]], tiles=[tile_name(z), tile_name(x)])


def book_doc(r: dict[str, np.ndarray], i: int, policies: Sequence[str], rules: Sequence[Sequence[str]]) -> dict:
    """One row as the page's annotation of its position."""
    j = int(r["policy"][i])
    allowed = int(r["allowed"][i])
    return dict(
        rule=policies[j].removeprefix("book:"), which=rules[j][int(r["rule"][i])],
        allowed=[t for t in range(28) if allowed >> t & 1], agree=bool(r["agree"][i]),
        contrast=int(r["x"][i]), shape=SHAPES[int(r["shape"][i])],
        **{k: round(float(r[k][i]), 4) for k in ("dp", "bar_p", "dmu", "bar_mu", "pz", "px", "mz", "mx")},
        ahead=bool(r["ahead"][i]), behind=bool(r["behind"][i]), exact=bool(r["exact"][i]),
    )


def export_sample(pour_dir: str | Path, r: dict[str, np.ndarray], picks: Sequence[int], policies: Sequence[str],
                  rules: Sequence[Sequence[str]], reasons: bool = True) -> dict:
    """The page's document (`scenario.viz.export`) for the sampled rows'
    deals, each holding only its sampled positions, each annotated with its
    rows (`book`): the rule, its allowed tiles, the contrast, the shape and
    its statistics, and the reason report's facts for the split."""
    from scenario.atlas import load
    from scenario.viz.export import deal_doc, document, position_doc

    src = Path(pour_dir)
    manifest = json.loads((src / "manifest.json").read_text())
    p = load(src)
    seeds = manifest.get("deal_seeds") or [None] * len(p.hands)
    by_pos: dict[int, list[int]] = {}
    for i in picks:
        by_pos.setdefault(int(r["pos"][i]), []).append(i)
    deals: dict[int, dict] = {}
    for q in by_pos:                    # the sample's order: by shape, the deals as they come
        pos = p.position(q)
        if pos.deal not in deals:
            deals[pos.deal] = deal_doc(p, pos.deal, seeds[pos.deal])
        doc = position_doc(pos)
        doc["book"] = []
        for i in by_pos[q]:
            if (int(r["deal"][i]), int(r["k"][i])) != (pos.deal, len(pos.plays)) or int(r["z"][i]) != pos.chosen:
                raise AssertionError("a sampled row is not its pour position")
            note = book_doc(r, i, policies, rules)
            if reasons and note["shape"] != "identical":
                note["reason"] = reason_doc(pos, int(r["z"][i]), int(r["x"][i]), int(seeds[pos.deal] or 0))
            doc["book"].append(note)
        deals[pos.deal]["positions"].append(doc)
    for dl in deals.values():
        dl["positions"].sort(key=lambda x: x["play"])
    head = document(p, list(deals.values()), manifest)
    head["source"]["pour"] = src.name
    head["book"] = dict(shapes=list(SHAPES), bar_se=BAR_SE, seed=SAMPLE_SEED)
    return head


SHORT = ("same", "noise", "behind", "dom", "make>μ", "make>tail", "μ up", "μ down", "spread")


def _row(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"| {label} | 0 |" + " |" * len(SHAPES)
    return f"| {label} | {s['n']:,} | " + " | ".join(f"{100 * s[name]['share']:.0f}" for name in SHAPES) + " |"


def tables(res: dict) -> str:
    """The summary as Markdown: shares in percent per book rule, per rule
    inside it, and per axis, disagreements and agreements apart."""
    head = "| | n | " + " | ".join(SHORT) + " |\n|---|---|" + "---|" * len(SHAPES)
    out = []
    for side in ("disagree", "agree"):
        out.append(f"### {side}\n\n{head}")
        out.append(_row("all rules", res[side]))
        for name, e in sorted(res["policy"].items(), key=lambda kv: -kv[1][side].get("n", 0)):
            out.append(_row(name, e[side]))
        out.append("")
    out.append("### rules inside\n\n" + head)
    for name, e in res["policy"].items():
        for rule, x in e["rule"].items():
            out.append(_row(f"{name}: {rule} (disagree)", x["disagree"]))
            out.append(_row(f"{name}: {rule} (agree)", x["agree"]))
    out.append("")
    for axis, values in res["axes"].items():
        out.append(f"### {axis}\n\n{head}")
        for value, x in values.items():
            out.append(_row(f"{value} (disagree)", x["disagree"]))
            out.append(_row(f"{value} (agree)", x["agree"]))
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -u -m scenario.metasplat", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="one chunk's rows: every firing book rule, Zeb's pick against the contrast")
    a.add_argument("--pour", required=True, help="the chunk's pour directory")
    a.add_argument("--reprice", required=True, help="book-reprice-3's out/ directory")
    a.add_argument("--label", required=True, help="the chunk's label there (s100001, ...)")
    a.add_argument("--out", required=True, help="rows .npz (a .json sidecar is written beside it)")
    a.add_argument("--deals", type=int, help="only the first N deals (a smoke run)")
    s = sub.add_parser("summarize", help="the metasplat: results.json and tables.md from chunks' rows")
    s.add_argument("rows", nargs="+", help="rows .npz, in chunk order")
    s.add_argument("--out", required=True, help="a directory")
    e = sub.add_parser("sample", help="the sampled positions as the splat page's data, with the reasons")
    e.add_argument("rows", nargs="+", help="rows .npz, in chunk order (the sample is drawn from the first)")
    e.add_argument("--pour", required=True, help="the first chunk's pour directory")
    e.add_argument("--out", required=True, help="the page's JSON")
    e.add_argument("--no-reasons", action="store_true", help="skip the reason report's facts")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        res = run(args.pour, args.reprice, args.label, args.out, args.deals)
        print(f"{args.label}: {len(res['pos']):,} rows -> {args.out}")
    elif args.cmd == "summarize":
        r, policies, rules = load_rows(args.rows)
        res = summarize(r, policies, rules)
        res["chunks"] = [Path(f).stem for f in args.rows]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
        (out / "tables.md").write_text(tables(res))
        print(f"{res['rows']:,} rows over {res['positions']:,} positions -> {out}")
    else:
        r, policies, rules = load_rows(args.rows)
        picks = pick_sample(r)
        doc = export_sample(args.pour, r, picks, policies, rules, reasons=not args.no_reasons)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        print(f"{len(picks)} rows, {len(doc['deals'])} deals -> {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()

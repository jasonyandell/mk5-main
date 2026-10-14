"""The metasplat's shapes: each of the nine is what its definition says on a
hand-built pair; the batch reading is the reference reading on random pairs,
sampled and exact; the best tile is the most make, then the most margin, then
the lowest id; the fiber's size is `scenario.fiber`'s."""
import random

import numpy as np
import pytest

from scenario.atlas import deal
from scenario.fiber import Fiber
from scenario.metasplat import (
    SHAPE, SHAPES, best_action, classify, contrast, contrasts, fiber_log2, success_index,
)
from scenario.record import empty_record
from scenario.rules import ALL_TILES, rules_for, tiles_of

BID = 30


def worlds(*runs):
    """A margin column from (margin, count) runs."""
    return np.concatenate([np.full(c, m, np.int64) for m, c in runs])


def test_the_success_threshold_is_the_contract_for_either_side():
    # bidders make at margin >= 18: mover's margin > 16; setters set at margin < 18: mover's margin > -18
    assert 2 * success_index(1, BID) - 42 == 16
    assert 2 * success_index(-1, BID) - 42 == -18


@pytest.mark.parametrize("z, x, sign, exact, shape", [
    (worlds((20, 300), (-10, 212)), worlds((20, 300), (-10, 212)), 1, False, "identical"),
    # one world apart: inside every bar
    (worlds((20, 300), (-10, 212)), worlds((20, 299), (-10, 213)), 1, False, "noise-tie"),
    # the same one world in an exact cell is a real difference: the make, nothing given up
    (worlds((20, 6), (-10, 6)), worlds((20, 5), (-10, 7)), 1, True, "dominant"),
    # 60 more worlds made, each by the smallest step, and nothing worse
    (worlds((18, 260), (-10, 252)), worlds((18, 200), (16, 60), (-10, 252)), 1, False, "dominant"),
    # made more often, but the big wins are given up: the mean falls
    (worlds((18, 400), (-10, 112)), worlds((42, 300), (-10, 212)), 1, False, "make-for-margin"),
    # made more often, a tail given up, the mean held
    (worlds((24, 300), (-20, 212)), worlds((24, 240), (-14, 272)), 1, False, "make-for-tail"),
    # same make, Zeb's wins larger
    (worlds((40, 300), (-10, 212)), worlds((20, 300), (-10, 212)), 1, False, "even-make-mean-up"),
    # same make, the setters' view: Zeb's sets smaller (bidder margins closer to 18)
    (worlds((10, 300), (30, 212)), worlds((-20, 300), (30, 212)), -1, False, "even-make-mean-down"),
    # same make and mean, a wider spread
    (worlds((30, 256), (-30, 256)), worlds((40, 128), (20, 128), (-20, 128), (-40, 128)), 1, False,
     "even-make-spread"),
    # Zeb's pick makes less
    (worlds((-10, 300), (20, 212)), worlds((20, 300), (-10, 212)), 1, False, "behind-on-make"),
])
def test_each_shape_on_a_built_pair(z, x, sign, exact, shape):
    assert len(z) == len(x)
    c = contrast(z, x, sign, BID, exact)
    assert c.shape == shape, c


def test_the_bars_are_paired_standard_errors():
    z = worlds((20, 300), (-10, 212))
    x = worlds((20, 250), (-10, 262))     # 50 worlds made by Z only
    c = contrast(z, x, 1, BID, False)
    e = np.r_[np.zeros(250), np.ones(50), np.zeros(212)]
    assert c.dp == pytest.approx(50 / 512)
    assert c.bar_p == pytest.approx(2 * e.std(ddof=1) / np.sqrt(512))
    assert c.dmu == pytest.approx(50 * 30 / 512)
    assert contrast(z[:12], x[:12], 1, BID, True).bar_p == 0.0


def test_the_batch_is_the_reference():
    rng = np.random.default_rng(7)
    B, W = 400, 64
    z = rng.choice(np.arange(-42, 43, 2), size=(B, W)).astype(np.int8)
    x = z.copy()
    for b in range(B):   # the pair differs in a random share of worlds, by a random step
        k = rng.integers(0, W)
        idx = rng.choice(W, k, replace=False)
        x[b, idx] = np.clip(z[b, idx] + 2 * rng.integers(-8, 9, k), -42, 42)
    n = rng.integers(1, W + 1, B)
    sign = rng.choice([-1, 1], B)
    exact = rng.random(B) < 0.3
    got = contrasts(z, x, n, sign, BID, exact)
    seen = set()
    for b in range(B):
        ref = contrast(z[b, :n[b]], x[b, :n[b]], int(sign[b]), BID, bool(exact[b]))
        assert SHAPES[got["shape"][b]] == ref.shape
        for k in ("dp", "bar_p", "dmu", "bar_mu"):
            assert got[k][b] == pytest.approx(getattr(ref, k), abs=1e-12)
        assert (got["ahead"][b], got["behind"][b]) == (ref.ahead, ref.behind)
        seen.add(ref.shape)
    assert len(seen) >= 6


def test_classify_is_first_match():
    assert classify(True, 0.5, 0, 9, 0, True, True) == "identical"
    assert classify(False, 0.0, 0.01, 0.1, 0.2, False, False) == "noise-tie"
    assert classify(False, 0.0, 0.01, 0.3, 0.2, False, False) == "even-make-mean-up"
    assert classify(False, -0.05, 0.01, 3, 1, True, True) == "behind-on-make"
    assert len(SHAPES) == len(SHAPE) == 9


def test_the_best_action_is_make_then_margin_then_lowest_tile():
    success = np.array([[3, 5, 5, 5, 0]])
    msum = np.array([[90, 10, 20, 20, 0]])
    tiles = np.array([[2, 4, 9, 7, -1]])
    allowed = np.array([[True, True, True, True, False]])
    assert best_action(success, msum, tiles, allowed)[0] == 3            # 7 before 9 at equal make and margin
    assert best_action(success, msum, tiles, allowed & np.array([[1, 1, 0, 0, 0]], bool))[0] == 1
    assert best_action(success, msum, tiles, np.zeros_like(allowed))[0] == -1


def _positions(n_deals: int, seed: int):
    """Random-play positions: (viewer, own, played, sizes4, forbid4)."""
    rng = random.Random(seed)
    for s in range(n_deals):
        hands = list(deal(seed + s))
        decl = s % 7
        rules = rules_for(decl)
        rec = empty_record(0)
        for _ in range(28):
            seat = rec.seat
            own = hands[seat] & ~rec.played
            yield seat, own, rec.played, tuple(rec.hand_size(q) for q in range(4)), rec.forbid
            led = rec.trick[0] if rec.trick else None
            rec = rec.advance(rng.choice(tiles_of(rules.legal(own, led))), rules)


def test_the_fiber_size_is_the_fibers():
    rows = list(_positions(12, 31))
    pool, sizes, forbid, want = [], [], [], []
    for viewer, own, played, sizes4, forbid4 in rows:
        f = Fiber(viewer, own, played, sizes4, forbid4)
        pool.append(ALL_TILES & ~played & ~own)
        sizes.append(f.sizes)
        forbid.append(f.forbid)
        want.append(f.size)
    got = fiber_log2(np.array(pool), np.array(sizes), np.array(forbid))
    assert np.allclose(got, np.log2(want), atol=1e-12)
    assert max(want) > 10 ** 6 and min(want) == 1 and any(any(f) for f in forbid)

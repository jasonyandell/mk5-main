"""Lumps by a hidden tile's holder, on a real (small) pour: every row is the
plain reading of its position (lump by lump, move by move), the VOI is the
information inequality's gap and never negative, a tile whose seat is known
has nothing to sell, and the unpacked columns read as the npz."""
import numpy as np
import pytest

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import deal, pour, save
from scenario.lumps_voi import (
    BAR_SE, MIN_MASS, SAMPLED, deal_rows, lump_rows, open_pour, summarize, tables, top_rows, describe, unpack,
)
from scenario.planner import HAND_TOTAL
from scenario.rules import ALL_TILES, bit

BID = 30
SPEC = "k0,n16"


@pytest.fixture(scope="module")
def poured(tmp_path_factory):
    seeds = (21, 22, 23)
    p = pour([deal(s) for s in seeds], 0, BID, _declared_trump, parse_walt_spec(SPEC).walt)
    root = tmp_path_factory.mktemp("lumps")
    save(p, root / "pour", spec=SPEC, deal_seeds=list(seeds))
    return p, root


def reference(pos, tile: int, sampled: int) -> dict:
    """The row for one position and tile, read plainly."""
    o = pos.outcomes.astype(np.int64)
    succ = ((o >= 2 * pos.bid - HAND_TOTAL) == (pos.sign > 0)).astype(float)
    marg = (pos.sign * o).astype(float)
    W = o.shape[1]
    holder = [next(i for i in range(3) if (int(w[(pos.viewer + i + 1) % 4]) >> tile) & 1) for w in pos.worlds]
    lumps = [[r for r in range(W) if holder[r] == lu] for lu in range(3)]
    mass = [len(x) / W for x in lumps]
    voi = {}
    for name, v in (("p", succ), ("mu", marg)):
        voi[name] = sum(max(v[a, x].sum() for a in range(len(pos.actions))) for x in lumps if x) / W \
            - max(v[a].sum() for a in range(len(pos.actions))) / W
    nontrivial = [x for x, m in zip(lumps, mass) if m >= MIN_MASS]

    def best(x, first, second):
        return max(range(len(pos.actions)), key=lambda a: (first[a, x].sum(), second[a, x].sum(), -pos.actions[a]))

    def within(x, v, b, a) -> bool:
        d = v[b, x] - v[a, x]
        bar = 0.0 if W < sampled or len(x) < 2 else BAR_SE * d.std(ddof=1) / np.sqrt(len(x))
        return d.mean() <= bar + 1e-12

    robust = {
        "p": any(all(within(x, succ, best(x, succ, marg), a) for x in nontrivial) for a in range(len(pos.actions))),
        "mu": any(all(within(x, marg, best(x, marg, succ), a) for x in nontrivial) for a in range(len(pos.actions))),
    }
    strict = {"p": len({best(x, succ, marg) for x in nontrivial}) == 1,
              "mu": len({best(x, marg, succ) for x in nontrivial}) == 1}
    return dict(mass=mass, voi=voi, split=len(nontrivial) >= 2, robust=robust, strict=strict,
                best_p=[pos.actions[best(x, succ, marg)] if x else -1 for x in lumps])


def test_every_row_is_the_plain_reading_of_its_position(poured):
    p, root = poured
    pr = open_pour(root / "pour")
    r = deal_rows(pr, [0, 1, 2], BID, 0, sampled=16)
    by = {(q.deal, len(q.plays)): q for q in p.positions()}
    live = [q for q in by.values() if q.record.banked[0] < BID and q.record.banked[1] <= HAND_TOTAL - BID]
    hidden = sum((ALL_TILES & ~q.record.played & ~q.own).bit_count() for q in live)
    assert len(r["tile"]) == hidden > 100
    seen = set()
    for i in range(len(r["tile"])):
        q = by[(int(r["deal"][i]), int(r["k"][i]))]
        t = int(r["tile"][i])
        assert not (q.own | q.record.played) & bit(t)
        ref = reference(q, t, 16)
        assert np.allclose(r["mass"][i], ref["mass"], atol=1e-6)
        assert r["voi_p"][i] == pytest.approx(ref["voi"]["p"], abs=1e-12)
        assert r["voi_mu"][i] == pytest.approx(ref["voi"]["mu"], abs=1e-9)
        assert bool(r["split"][i]) == ref["split"]
        for v in ("p", "mu"):
            if ref["split"]:
                assert bool(r[f"robust_{v}"][i]) == ref["robust"][v]
                assert bool(r[f"strict_{v}"][i]) == ref["strict"][v]
        assert list(r["best_p"][i]) == ref["best_p"]
        assert r["voi_p"][i] >= 0 and r["voi_mu"][i] >= 0 and r["null_p"][i] >= 0
        assert int(r["chosen"][i]) == q.chosen
        seen.add((bool(r["split"][i]), bool(r["robust_p"][i]), bool(r["voi_p"][i] > 0)))
    assert len(seen) >= 3   # the fixture exercises split and settled, robust and hedge, VOI zero and not


def test_a_tile_whose_seat_is_known_is_worth_nothing():
    # two worlds, two moves; tile 0 is the left's in both, tile 1 moves between left and partner
    worlds = np.array([[0, 0b01, 0b10, 0], [0, 0b11, 0, 0]], np.uint32)
    outcomes = np.array([[40, -40], [-40, 40]], np.int8)      # move a makes in world 0, move b in world 1
    r = lump_rows(outcomes, worlds, [5, 6], 5, 0, 0b11, 1, BID, True, np.random.default_rng(0))
    known, moving = (0, 1) if r["tile"][0] == 0 else (1, 0)
    assert not r["split"][known] and r["voi_p"][known] == 0
    assert r["split"][moving] and r["voi_p"][moving] == pytest.approx(0.5)   # knowing it: make always, else half
    assert not r["robust_p"][moving] and not r["strict_p"][moving]           # an exact cell: no bar


def test_unpacked_columns_read_as_the_npz_and_the_summary_counts_every_row(poured, tmp_path):
    _, root = poured
    unpack(root / "pour", tmp_path / "unpacked")
    a = deal_rows(open_pour(root / "pour"), [0, 1, 2], BID, 0, sampled=16)
    b = deal_rows(open_pour(tmp_path / "unpacked"), [0, 1, 2], BID, 0, sampled=16)
    for k in a:
        assert np.array_equal(a[k], b[k])
    res = summarize(a)
    assert res["rows"] == len(a["tile"])
    assert sum(e["n"] for e in res["tile"].values()) == res["rows"]
    assert res["hedge"]["all"]["n"] == int(a["split"].sum())
    assert "VOI by tile" in tables(res)
    top = [describe(a, i) for i in top_rows(a, 3)]
    assert top[0]["voi_p"] == round(float(a["voi_p"].max()), 3)
    assert SAMPLED == 512

"""The metasplat on a real (small) pour, read against a stand-in book rule
("play your lowest legal tile"): every row is its position's pair, read by
the reference `contrast`; the contrast is the best tile on the other side of
the book; the position's facts are its record's; the summary counts every
row once; the sample is the page's own document, annotated, and its reasons
split the worlds where the pair parts."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import deal, pour, save
from scenario.fiber import Fiber
from scenario.metasplat import (
    NO_CONTRAST, SHAPES, contrast, export_sample, load_rows, pick_sample, run, split_of, summarize, tables,
)
from scenario.viz.export import position_doc

BID = 30
SPEC = "k0,n16"
POLICIES = ["book:lowest", "gate:never"]
RULES = [["the-lowest-tile"], ["never"]]


@pytest.fixture(scope="module")
def poured(tmp_path_factory):
    p = pour([deal(s) for s in (11, 12, 13, 14)], 0, BID, _declared_trump, parse_walt_spec(SPEC).walt)
    root = tmp_path_factory.mktemp("metasplat")
    save(p, root / "pour", spec=SPEC, deal_seeds=[11, 12, 13, 14])
    # the re-price's columns for a book of one rule: the lowest legal tile, everywhere
    rows = list(p.positions())
    n = len(rows)
    book = dict(deal=np.array([q.deal for q in rows], np.int32), k=np.array([len(q.plays) for q in rows], np.int8),
                seat=np.array([q.viewer for q in rows], np.int8), decl=np.array([q.decl_id for q in rows], np.int8),
                banked=np.array([q.record.banked for q in rows], np.int16), zeb=np.array([q.chosen for q in rows], np.int8),
                actions=np.full((n, 7), -1, np.int8), q=np.full((n, 7), np.nan), rule=np.full((n, 2), -1, np.int16),
                allowed=np.zeros((n, 2), np.uint32))
    for i, q in enumerate(rows):
        book["actions"][i, :len(q.actions)] = q.actions
        book["q"][i, :len(q.actions)] = q.q
        book["rule"][i, 0] = 0
        book["allowed"][i, 0] = 1 << min(q.actions)
    (root / "reprice").mkdir()
    np.savez_compressed(root / "reprice" / "t1.p0.npz", **book)
    (root / "reprice" / "t1.p0.json").write_text(json.dumps(dict(policies=POLICIES, rules=RULES)))
    res = run(root / "pour", root / "reprice", "t1", root / "rows" / "t1.npz")
    return p, root, res


def test_every_live_firing_is_a_row_and_its_pair_is_the_reference_reading(poured):
    p, _, res = poured
    by = {(q.deal, len(q.plays)): q for q in p.positions()}
    live = [q for q in by.values() if q.record.banked[0] < BID and q.record.banked[1] <= 42 - BID]
    assert len(res["pos"]) == len(live) > 20
    sampled = 0
    for i in range(len(res["pos"])):
        q = by[(int(res["deal"][i]), int(res["k"][i]))]
        z, low = q.chosen, min(q.actions)
        assert res["z"][i] == z and bool(res["agree"][i]) == (z == low)
        side = [t for t in q.actions if t != low] if z == low else [low]
        if not side:
            assert res["shape"][i] == NO_CONTRAST and res["x"][i] == -1
            continue

        def key(t):
            o = q.sign * q.outcomes[q.actions.index(t)].astype(np.int64)
            ok = (q.outcomes[q.actions.index(t)] >= 18) == (q.sign > 0)
            return (ok.sum(), o.sum(), -t)
        x = max(side, key=key)
        assert res["x"][i] == x
        exact = len(q.weights) < 16
        sampled += not exact
        ref = contrast(q.outcomes[q.actions.index(z)], q.outcomes[q.actions.index(x)], q.sign, BID, exact)
        assert SHAPES[res["shape"][i]] == ref.shape
        assert res["dp"][i] == pytest.approx(ref.dp) and res["dmu"][i] == pytest.approx(ref.dmu)
        assert res["bar_p"][i] == pytest.approx(ref.bar_p) and res["exact"][i] == exact
        assert res["pz"][i] - res["px"][i] == pytest.approx(ref.dp)
        assert res["z_best"][i]                     # the plan's pick has the most make
    assert sampled > 0


def test_the_facts_are_the_records(poured):
    p, _, res = poured
    by = {(q.deal, len(q.plays)): q for q in p.positions()}
    for i in range(len(res["pos"])):
        q = by[(int(res["deal"][i]), int(res["k"][i]))]
        rec = q.record
        f = Fiber(q.viewer, q.own, rec.played, tuple(rec.hand_size(s) for s in range(4)), rec.forbid)
        assert res["log2_fiber"][i] == pytest.approx(np.log2(f.size))
        assert res["seat"][i] == q.viewer and res["in_trick"][i] == len(rec.trick)
        assert (res["banked_bidders"][i], res["banked_setters"][i]) == rec.banked
        assert res["table"][i] == sum({20: 10, 25: 10, 15: 5, 11: 5, 8: 5}.get(t, 0) for t in rec.trick)


def test_the_summary_counts_every_row_once(poured):
    _, root, res = poured
    r, policies, rules = load_rows([root / "rows" / "t1.npz"])
    s = summarize(r, policies, rules)
    contrasted = res["shape"] >= 0
    assert s["disagree"]["n"] + s["agree"]["n"] + s["no_contrast"] == s["rows"] == len(res["pos"])
    for side in ("disagree", "agree"):
        assert sum(s[side][name]["count"] for name in SHAPES) == s[side]["n"]
        for axis in s["axes"].values():
            assert sum(v[side]["n"] for v in axis.values()) == s[side]["n"]
    assert s["policy"]["lowest"]["n"] == len(res["pos"]) and s["agree"]["n"] == int((res["agree"] & contrasted).sum())
    assert "| lowest |" in tables(s)


def test_the_sample_is_the_pages_document_annotated(poured):
    p, root, _ = poured
    r, policies, rules = load_rows([root / "rows" / "t1.npz"])
    picks = pick_sample(r)
    assert picks and len(set(picks)) == len(picks)
    doc = export_sample(root / "pour", r, picks, policies, rules)
    by = {(q.deal, len(q.plays)): q for q in p.positions()}
    notes = 0
    for dl in doc["deals"]:
        for x in dl["positions"]:
            q = by[(_deal_of(p, dl), x["play"])]
            want = position_doc(q)
            assert {k: v for k, v in x.items() if k != "book"} == json.loads(json.dumps(want))
            for b in x["book"]:
                notes += 1
                assert b["rule"] == "lowest" and b["allowed"] == [min(q.actions)]
                assert b["agree"] == (q.chosen == min(q.actions))
                if b["shape"] == "identical":
                    continue
                basis, one, two = split_of(q, q.chosen, b["contrast"])
                reason = b["reason"]
                assert reason["basis"] == basis
                if one and two:
                    assert reason["sides"] == [len(one), len(two)]
                for f in reason["facts"]:
                    assert Fraction(f["on"][0]) >= 0 and Fraction(f["on"][1]) <= 1
    assert notes == len(picks)


def test_the_checked_in_sample_reads_off_its_own_moves():
    """`scenario/viz/data/metasplat_sample.json`: every note's contrast and
    book tiles are legal there, its Δp and Δμ are its two moves' own pmake and
    mean, and its shape is the reference reading of their world-by-world
    outcomes."""
    doc = json.loads((Path(__file__).resolve().parents[1] / "viz" / "data" / "metasplat_sample.json").read_text())
    seen = set()
    for dl in doc["deals"]:
        for x in dl["positions"]:
            moves = {m["tile"]: m for m in x["moves"]}
            for b in x["book"]:
                z, c = moves[x["chosen"]], moves[b["contrast"]]
                assert set(b["allowed"]) <= set(moves) and b["agree"] == (x["chosen"] in b["allowed"])
                sign = x["sign"]
                assert b["dp"] == pytest.approx(sign * (z["pmake"] - c["pmake"]), abs=1e-4)
                assert b["dmu"] == pytest.approx(sign * (z["mean"] - c["mean"]), abs=1e-4)
                oz, oc = (2 * (np.frombuffer(m["outcomes"].encode(), np.uint8).astype(np.int64) - 48) - 42
                          for m in (z, c))
                assert contrast(oz, oc, sign, doc["bid"], b["exact"]).shape == b["shape"]
                seen.add(b["shape"])
    assert len(seen) == 8          # every shape the pour has (none is behind on the make)


def _deal_of(p, dl) -> int:
    """The pour's index of an exported deal (by its hands)."""
    for d in range(len(p.hands)):
        if [sorted(t for t in range(28) if int(p.hands[d][s]) >> t & 1) for s in range(4)] == dl["hands"]:
            return d
    raise AssertionError("an exported deal is not in the pour")

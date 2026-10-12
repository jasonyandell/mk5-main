"""The splat visualizer's exporter: a pour becomes the page's JSON without
losing or inventing a number. Every planned position is there with its
viewer, legal moves and plan; each move's bins hold exactly its splat (the
belief weight sums to the cell's, the tail at the contract is
`atlas.makes`'s mass); the deal replays to its recorded ending."""
import json

import numpy as np
import pytest

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import N_PLAYS, deal, in_plan_units, makes, pour, save
from scenario.planner import HAND_TOTAL
from scenario.viz.export import BINS, export, export_dir, main

BID = 30


@pytest.fixture(scope="module")
def poured():
    return pour([deal(s) for s in (5, 6)], 0, BID, _declared_trump, parse_walt_spec("k1,n8,m2").walt)


@pytest.fixture(scope="module")
def doc(poured):
    # through json, as the page reads it
    return json.loads(json.dumps(export(poured, {"deal_seeds": [5, 6], "spec": "k1,n8,m2"})))


def test_every_planned_position_is_exported_in_order(poured, doc):
    got = [(d, p["play"]) for d, dl in enumerate(doc["deals"]) for p in dl["positions"]]
    want = sorted((pos.deal, len(pos.plays)) for pos in poured.positions())
    assert got == want
    assert doc["threshold"] == 2 * BID - HAND_TOTAL == 18
    assert doc["utility"] == poured.utility and doc["source"]["spec"] == "k1,n8,m2"


def test_each_splat_is_its_moves_outcomes(poured, doc):
    by = {(dl_i, p["play"]): p for dl_i, dl in enumerate(doc["deals"]) for p in dl["positions"]}
    for pos in poured.positions():
        p = by[(pos.deal, len(pos.plays))]
        total = pos.weights.sum()
        assert p["viewer"] == pos.viewer and p["sign"] == pos.sign and p["chosen"] == pos.chosen
        assert p["worlds"] == len(pos.weights) and p["weight"] == pytest.approx(total)
        assert [m["tile"] for m in p["moves"]] == list(pos.actions)
        for a, m in enumerate(p["moves"]):
            bins = np.array(m["bins"], dtype=np.float64)
            assert len(bins) == BINS
            assert bins.sum() == pytest.approx(total)                       # nothing lost
            margins = 2 * np.arange(BINS) - HAND_TOTAL
            mass = pos.weights[makes(pos, m["tile"])].sum() / total
            assert m["pmake"] == pytest.approx(mass, abs=1e-12)               # atlas.makes's mass
            assert bins[margins >= doc["threshold"]].sum() / total == pytest.approx(mass, abs=1e-12)
            assert bins @ margins / total == pytest.approx(m["mean"])         # the mean is the bins'
            assert m["q"] == pos.q[a]
            assert in_plan_units(pos, pos.outcomes[a]) @ pos.weights / total == pytest.approx(m["q"])


def test_the_belief_is_on_the_real_holders(poured, doc):
    for pos in poured.positions():
        p = next(x for x in doc["deals"][pos.deal]["positions"] if x["play"] == len(pos.plays))
        hidden = {(s, t) for s in range(4) if s != pos.viewer
                  for t in range(28) if (pos.hands[s] & ~pos.record.played) >> t & 1}
        assert {(s, t) for s, t, _ in p["belief"]} == hidden
        assert all(0 <= b <= 1 for _, _, b in p["belief"])


def test_the_deal_replays_to_its_ending(poured, doc):
    for d, dl in enumerate(doc["deals"]):
        assert dl["seed"] == (5, 6)[d] and dl["decl"] == poured.decl[d]
        assert len(dl["plays"]) == N_PLAYS and len(dl["banked"]) == N_PLAYS + 1
        assert dl["banked"][0] == [0, 0] and tuple(dl["banked"][-1]) == tuple(poured.banked[d])
        assert sum(t["points"] for t in dl["tricks"]) == HAND_TOTAL and len(dl["tricks"]) == 7
        assert dl["margin"] == int(poured.banked[d, 0]) - int(poured.banked[d, 1])
        assert sorted(t for h in dl["hands"] for t in h) == list(range(28))


def test_the_command_line_writes_the_document(poured, tmp_path, capsys):
    save(poured, tmp_path / "pour", spec="k1,n8,m2", deal_seeds=[5, 6])
    out = tmp_path / "viz.json"
    main([str(tmp_path / "pour"), "--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(json.dumps(export_dir(tmp_path / "pour")))
    assert "planned positions" in capsys.readouterr().out

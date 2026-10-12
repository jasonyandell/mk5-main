"""The splat visualizer's exporter: a pour becomes the page's JSON without
losing or inventing a number. Every planned position is there with its
viewer, legal moves and plan; each move's bins hold exactly its splat (the
belief weight sums to the cell's, the tail at the contract is
`atlas.makes`'s mass); the deal replays to its recorded ending."""
import dataclasses
import json

import numpy as np
import pytest

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import N_PLAYS, deal, holding_predictors, in_plan_units, makes, pour, save
from scenario.planner import HAND_TOTAL
from scenario.viz.export import BINS, cell_doc, export, export_dir, main

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


def _cell(p):
    """A position's worlds as the page decodes them: holder seat per hidden
    tile per world, each move's margin per world, and the weights."""
    n, h = p["worlds"], len(p["hidden"])
    digits = np.frombuffer(p["holders"].encode(), dtype=np.uint8).reshape(n, h) - 48
    holder = np.array(p["seats"])[digits]
    margins = [2 * (np.frombuffer(m["outcomes"].encode(), dtype=np.uint8).astype(np.int64) - 48) - HAND_TOTAL
               for m in p["moves"]]
    w = np.array(p["weights"], dtype=np.float64) if "weights" in p else np.full(n, p["weight"] / n)
    return holder, margins, w


def test_the_worlds_are_the_cells(poured, doc):
    assert doc["uniform"] is all(np.all(pos.weights == pos.weights[0]) for pos in poured.positions())
    for pos in poured.positions():
        p = next(x for x in doc["deals"][pos.deal]["positions"] if x["play"] == len(pos.plays))
        holder, margins, w = _cell(p)
        assert p["seats"] == [s for s in range(4) if s != pos.viewer]
        assert np.array_equal(w, pos.weights)
        for a in range(len(pos.actions)):
            assert np.array_equal(margins[a], pos.outcomes[a])                 # every world's ending
        for s in p["seats"]:                                                  # every world's hidden hands
            mask = [sum(1 << t for t, x in zip(p["hidden"], row) if x == s) for row in holder]
            assert np.array_equal(np.array(mask, dtype=np.uint32), pos.worlds[:, s])


def test_what_must_be_true_reads_off_the_worlds(poured, doc):
    """The page's what-must-be-true is `holding_predictors` over a region of
    a move's splat: the make tail, its complement, any band of margins."""
    for pos in poured.positions():
        p = next(x for x in doc["deals"][pos.deal]["positions"] if x["play"] == len(pos.plays))
        holder, margins, w = _cell(p)
        for a, tile in enumerate(pos.actions):
            for region in (makes(pos, tile), ~makes(pos, tile), margins[a] <= 0):
                base, pred = holding_predictors(pos, np.flatnonzero(region))
                assert w[region].sum() / w.sum() == pytest.approx(base, abs=1e-12)
                got = {}
                for j, t in enumerate(p["hidden"]):
                    for s in p["seats"]:
                        holds = holder[:, j] == s
                        if w[holds].sum() > 0:
                            got[(s, t)] = (w[holds & region].sum() / w[holds].sum(), w[holds].sum() / w.sum())
                assert got.keys() == pred.keys()
                for k in pred:
                    assert got[k] == pytest.approx(pred[k], abs=1e-12)


def test_unequal_weights_are_written(poured):
    """The pour's worlds are equally weighted, so `weights` is omitted; a
    belief that weighs them unequally is written out and read back."""
    w = 1.0 + np.arange(len(poured.weights)) % 3
    for pos in dataclasses.replace(poured, weights=w).positions():
        p = json.loads(json.dumps({"worlds": len(pos.weights), "weight": 0, "moves": [], **cell_doc(pos)}))
        assert np.array_equal(_cell(p)[2], pos.weights)

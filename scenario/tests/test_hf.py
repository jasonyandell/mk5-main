# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""A pour, published. The three tables are the pour's columns cut at its
offsets: every row reads back to its position bit for bit, the advantages
are the atlas reader's, and the staged repo is a dataset card (CC0, one
config per table, one split per pour) over Parquet and the raw npz. No
network: upload is not exercised."""
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import _COLUMNS, N_PLAYS, advantages, deal, load, pour, save
from scenario.hf import _main, read_split, stage, tables
from scenario.rules import mask_of

SPEC = "k1,n8,m2,u=points"


def _poured(seeds):
    return pour([deal(s) for s in seeds], 0, 30, _declared_trump, parse_walt_spec(SPEC).walt)


@pytest.fixture(scope="module")
def pours(tmp_path_factory):
    out = {}
    for name, seeds in (("train", (1, 2, 3)), ("validation", (4, 5))):
        path = tmp_path_factory.mktemp(name)
        save(_poured(seeds), path, spec=SPEC, seed=seeds[0], fiber=False, deal_seeds=list(seeds), commit="c0ffee")
        out[name] = path
    return out


def test_every_row_is_its_position(pours):
    p = load(pours["train"])
    t = {k: v.to_pylist() for k, v in tables(p, {"deal_seeds": [1, 2, 3]}).items()}
    assert len(t["positions"]) == len(t["splats"]) == len(p) and len(t["deals"]) == 3
    for i, (row, splat) in enumerate(zip(t["positions"], t["splats"])):
        pos = p.position(i)
        assert row["position"] == splat["position"] == i and row["deal_seed"] == 1 + pos.deal
        assert [(h["seat"], h["tile"]) for h in row["history"]] == list(pos.plays)
        assert (row["seat"], row["sign"], row["decl"]) == (pos.viewer, pos.sign, pos.decl_id)
        assert mask_of(row["hand"]) == pos.own and row["chosen"] == pos.chosen
        assert tuple(row["actions"]) == tuple(splat["actions"]) == pos.actions
        assert row["q"] == pos.q.tolist() and row["n_worlds"] == len(pos.weights)
        assert np.allclose(row["advantage"], list(advantages(pos).values()), rtol=0, atol=1e-12)
        assert np.array_equal(np.array(splat["outcomes"], dtype=np.int8), pos.outcomes)
        assert np.array_equal(np.array(splat["worlds"], dtype=np.uint32), pos.worlds)
        assert splat["weights"] == pos.weights.tolist()
    for d, row in enumerate(t["deals"]):
        assert [mask_of(h) for h in row["hands"]] == [int(h) for h in p.hands[d]]
        assert [(x["seat"], x["tile"]) for x in row["plays"]] == [tuple(x) for x in p.plays[d].tolist()]
        assert row["banked"] == p.banked[d].tolist() and len(row["plays"]) == N_PLAYS
        assert row["made"] == (row["banked"][0] >= 30) and row["margin"] == row["banked"][0] - row["banked"][1]


def test_the_staged_repo_is_a_card_over_parquet_and_the_raw_pour(pours, tmp_path):
    out = stage([read_split(n, d) for n, d in pours.items()], tmp_path / "repo", "someone/walt-pour")
    text = (out / "README.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    assert front["license"] == "cc0-1.0" and "texas-42" in front["tags"]
    configs = {c["config_name"]: c for c in front["configs"]}
    assert set(configs) == {"positions", "splats", "deals"} and configs["positions"]["default"]
    for name, path in pours.items():
        p = load(path)
        for config, rows in (("positions", len(p)), ("splats", len(p)), ("deals", len(p.hands))):
            files = [f["path"] for f in configs[config]["data_files"] if f["split"] == name]
            assert files == [f"data/{config}/{name}-*.parquet"]
            assert sum(pq.read_metadata(f).num_rows for f in out.glob(files[0])) == rows
        back = load(out / "raw" / name)
        for c in _COLUMNS:
            assert np.array_equal(getattr(back, c), getattr(p, c)), c
        assert json.loads((out / "raw" / name / "manifest.json").read_text())["commit"] == "c0ffee"
        assert f"| `{name}` | {len(p.hands):,} |" in text
    assert "CC0 1.0 Universal" in (out / "LICENSE").read_text()
    sums = (out / "SHA256SUMS").read_text()
    assert "raw/train/pour.npz" in sums and "data/splats/validation-0000.parquet" in sums


def test_publish_refuses_what_would_mislabel(pours, tmp_path):
    with pytest.raises(SystemExit, match="share a deal"):
        stage([read_split("a", pours["train"]), read_split("b", pours["train"])], tmp_path, "x/y")
    bare = tmp_path / "bare"
    save(load(pours["validation"]), bare, spec=SPEC, deal_seeds=[4, 5])
    with pytest.raises(SystemExit, match="no commit"):
        read_split("v", bare)
    assert read_split("v", bare, commit="abc").manifest["commit"] == "abc"


def test_the_command_line_stages_without_uploading(pours, tmp_path, capsys):
    _main(["publish", "--repo", "someone/walt-pour", "--split", f"train={pours['train']}",
           "--stage", str(tmp_path / "s"), "--no-upload"])
    assert "staged train" in capsys.readouterr().out
    assert (tmp_path / "s" / "data" / "positions" / "train-0000.parquet").exists()


def test_rounds_of_an_expert_iteration_ship_their_nets(pours, tmp_path):
    """Splits may differ in the field net their spec names, and only there:
    every net (and one attached) lands in nets/ with its sidecar, the card
    names each split's net and its sha256, and the spec shows no local path."""
    import hashlib

    nets = {}
    for name in ("round1", "round2", "trained"):
        nets[name] = tmp_path / "local" / f"{name}.npz"
        nets[name].parent.mkdir(exist_ok=True)
        nets[name].write_bytes(name.encode())
    nets["round1"].with_suffix(".json").write_text('{"pour": "earlier"}')
    splits = []
    for (name, path), net in zip(pours.items(), ("round1", "round2")):
        p = tmp_path / name
        save(load(path), p, spec=f"k0,n8,F={nets[net]}", seed=1, deal_seeds=read_split(name, path).manifest["deal_seeds"],
             commit="c0ffee")
        splits.append(read_split(name, p))
    out = stage(splits, tmp_path / "repo", "someone/zeb-pour", title="Zeb: a pour", notes="Poured overnight.",
                nets=[nets["trained"]])
    assert sorted(f.name for f in (out / "nets").iterdir()) == ["round1.json", "round1.npz", "round2.npz", "trained.npz"]
    text = (out / "README.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    assert front["pretty_name"] == "Zeb: a pour" and "# Zeb: a pour" in text
    assert "## This release\n\nPoured overnight." in text and str(tmp_path / "local") not in text
    assert "`k0,n8,F=<net>`, each split's field net as in the table" in text
    assert "--spec k0,n8,F=nets/round1.npz" in text and "--net nets/trained.npz" in text
    for net, role in (("round1", "the field of `train`"), ("round2", "the field of `validation`"),
                      ("trained", "attached: trained on this pour")):
        assert f"| `{net}` | {role} | `{hashlib.sha256(net.encode()).hexdigest()}` |" in text
    assert "modeled by the field net `<net>`" in text and "cell-size floor of 16" in text
    assert "nets/trained.npz" in (out / "SHA256SUMS").read_text()
    bad = tmp_path / "k1"
    # a split under another planner is allowed (one contract, a planner per split, named in the card);
    # another contract is not
    save(load(pours["validation"]), bad, spec=f"k1,n8,F={nets['round2']}", seed=4, fiber=False, deal_seeds=[4, 5], commit="c0ffee")
    out2 = stage([splits[0], read_split("validation", bad)], tmp_path / "x", "x/y")
    assert "a planner per split" in (out2 / "README.md").read_text()
    m = json.loads((bad / "manifest.json").read_text()); m["bid"] = 31
    (bad / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="different contracts"):
        stage([splits[0], read_split("validation", bad)], tmp_path / "x2", "x/y")
    gone = tmp_path / "gone"
    save(load(pours["validation"]), gone, spec="k0,n8,F=/no/such.npz", deal_seeds=[4, 5], commit="c0ffee")
    with pytest.raises(SystemExit, match="no such field net"):
        stage([read_split("validation", gone)], tmp_path / "y", "x/y")


def test_splits_may_differ_in_planner_but_not_in_contract(pours, tmp_path):
    other = tmp_path / "other"
    save(_poured((7, 8)), other, spec="k1,n8,m8", seed=7, fiber=False, deal_seeds=[7, 8], commit="c0ffee",
         declare="declares by the book's menu, one pip deal in eight forced to doubles or no trump")
    out = stage([read_split("pip", pours["train"]), read_split("dtnt", other)], tmp_path / "s", "x/y")
    card = (out / "README.md").read_text()
    assert "a planner per split" in card and "k1,n8,m8" in card
    assert "one pip deal in eight" not in card           # the first split's manifest names the rule
    out2 = stage([read_split("dtnt", other), read_split("pip", pours["train"])], tmp_path / "s2", "x/y")
    assert "one pip deal in eight" in (out2 / "README.md").read_text()
    bad = tmp_path / "bad"
    save(_poured((9,)), bad, spec=SPEC, seed=9, fiber=False, deal_seeds=[9], commit="c0ffee")
    m = json.loads((bad / "manifest.json").read_text()); m["bid"] = 31
    (bad / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="different contracts"):
        stage([read_split("pip", pours["train"]), read_split("b", bad)], tmp_path / "s3", "x/y")


def test_a_list_column_past_two_to_the_31_takes_int64_offsets(monkeypatch):
    """A split's splat outcomes can hold more than 2^31 values (pour A's pip half does);
    the list column then takes int64 offsets instead of overflowing int32."""
    import pyarrow as pa
    from scenario import hf
    small = hf._lists(np.array([2, 1]), pa.array(np.arange(3, dtype=np.int8)))
    assert isinstance(small, pa.ListArray) and small.to_pylist() == [[0, 1], [2]]
    monkeypatch.setattr(hf, "INT32_MAX", 2)       # the same branch, without allocating 2^31 values
    large = hf._lists(np.array([2, 1]), pa.array(np.arange(3, dtype=np.int8)))
    assert isinstance(large, pa.LargeListArray) and large.to_pylist() == [[0, 1], [2]]


def test_a_manifest_that_names_its_reproduction_is_quoted_not_rebuilt():
    """A pour merged from chunks has no single seed; its manifest's `reproduce` is the card's command."""
    from scenario import hf
    assert hf._command({"reproduce": "for i in ...; done"}) == "for i in ...; done"
    assert "--seed 5" in hf._command({"seed": 5, "deals": 3, "spec": "k0,n8", "bid": 30, "bidder": 0})

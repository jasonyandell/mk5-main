# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Information boundaries, causal predictions, replay and checkpoint continuity."""
from argparse import Namespace
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from lem.observer.data import QUERIES, SEAT, encode, load_deals, partition, prepare
from lem.observer.model import Config, Observer
from lem.observer.train import evaluate, run
from lem.observer.inspect import build
from lem.observer.uncertainty import summarize
from lem.observer.tracking import metrics
from scenario.atlas import deal
from scenario.record import empty_record
from scenario.rules import rules_for, tiles_of


def recording(seed=42, decl=3):
    hands, plays = deal(seed), []
    rec, rules = empty_record(0), rules_for(decl)
    for _ in range(28):
        tile = tiles_of(rules.legal(hands[rec.seat] & ~rec.played, rec.led))[0]
        plays.append([rec.seat, tile])
        rec = rec.advance(tile, rules)
    return dict(seed=seed, hands=list(hands), decl=decl, plays=plays,
                coverage_forced=False, split=partition(hands))


def test_alignment_and_initial_hand_prefix():
    r = recording()
    data = prepare([r])
    assert data["x"].shape == (4, 68)
    for viewer in range(4):
        for i, (actor, target) in enumerate(r["plays"]):
            expected = np.append(encode(r["hands"][viewer], viewer, 0, r["decl"], r["plays"][:i]),
                                 SEAT + (actor - viewer) % 4)
            np.testing.assert_array_equal(data["x"][viewer, :QUERIES[i] + 1], expected)
            assert data["targets"][viewer, i] == target
            assert data["legal"][viewer, i, target] and data["possible"][viewer, i, target]
        assert data["own"][viewer].sum() == 7


def test_hidden_hands_and_future_never_reach_prefix():
    r = recording()
    hidden_changed = deepcopy(r)
    hidden_changed["hands"][1:3] = hidden_changed["hands"][2:0:-1]
    hidden_changed["plays"][10:] = list(reversed(hidden_changed["plays"][10:]))
    np.testing.assert_array_equal(encode(r["hands"][0], 0, 0, 3, r["plays"][:10]),
                                  encode(hidden_changed["hands"][0], 0, 0, 3, hidden_changed["plays"][:10]))
    d = prepare([r])
    actor = r["plays"][0][0]
    for viewer in range(4):
        assert d["possible"][viewer, 0].sum() == (7 if actor == viewer else 21)


def test_future_and_other_batch_members_cannot_change_prediction():
    torch.manual_seed(2)
    model = Observer(Config(width=16, layers=2, heads=2)).eval()
    x = torch.as_tensor(prepare([recording()])["x"])
    with torch.inference_mode():
        original, trace = model(x, inspect=True)
        changed = x.clone()
        changed[0, 24:] = (changed[0, 24:] + 1) % 46
        changed[1:] = 0
        torch.testing.assert_close(model(changed)[0, :24], original[0, :24], rtol=0, atol=0)
        torch.testing.assert_close(model(x[:1, :24])[0], original[0, :24], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model(x), original, rtol=0, atol=0)
        assert torch.count_nonzero(torch.triu(trace[0]["attention"], diagonal=1)) == 0
        assert trace[0]["pre"].shape == (4, 68, 64)


def test_split_groups_all_views_and_rejects_illegal_record():
    r = recording()
    d = prepare([r])
    assert len(set(d["split"])) == 1
    assert partition(r["hands"]) == partition(tuple(r["hands"]))
    r["plays"][1] = r["plays"][0]
    with pytest.raises(ValueError, match="illegal"):
        prepare([r])


def test_evaluation_does_not_consume_sampler_or_training_rng():
    model = Observer(Config(width=8, layers=1, heads=1))
    d = prepare([recording()])
    before = torch.get_rng_state()
    stats = evaluate(model, d, np.arange(4), "cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert stats["all"]["decisions"] == 112
    assert stats["own"]["decisions"] == 28


def test_training_resume_matches_uninterrupted_and_rejects_changed_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = [recording(seed) for seed in range(50)]
    np.savez(source / "pour.npz", hands=[r["hands"] for r in records],
             decl=[r["decl"] for r in records], plays=[r["plays"] for r in records])
    (source / "manifest.json").write_text(json.dumps(dict(bidder=0, bid=30, deal_seeds=list(range(50)))))
    common = dict(source=[str(source)], deals=50, steps=4, eval_every=2, batch=4,
                  width=8, layers=1, heads=1, lr=0.001, seed=42, device="cpu", resume=False,
                  init_from=None, checkpoint_every=2, wandb_project=None, wandb_entity=None, wandb_name=None)
    run(Namespace(**common, out=str(tmp_path / "continuous")))
    staged = Namespace(**common, out=str(tmp_path / "staged"))
    staged.steps = 2
    run(staged)
    fork = Namespace(**{**common, "steps": 4, "init_from": str(tmp_path / "staged")}, out=str(tmp_path / "fork"))
    # A fork at step 2 must continue the optimizer and sampler exactly, too.
    fork_report = run(fork)
    staged.steps, staged.resume = 4, True
    run(staged)
    a, b = [torch.load(tmp_path / folder / "latest.pt", weights_only=True) for folder in ("continuous", "staged")]
    assert torch.equal(a["sampler_state"], b["sampler_state"])
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    fork_saved = torch.load(tmp_path / "fork" / "latest.pt", weights_only=True)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], fork_saved["model"][key], rtol=0, atol=0)
    assert fork_report["parent_step"] == 2
    assert (tmp_path / "fork" / "initial.pt").read_bytes() == (tmp_path / "staged" / "initial.pt").read_bytes()
    intervals = summarize(tmp_path / "continuous", draws=100)
    report = json.loads((tmp_path / "continuous" / "report.json").read_text())
    for group in ("all", "own", "others", "discretionary"):
        assert intervals["groups"][group]["top1"]["estimate"] == report["test"][group]["top1"]
    page = tmp_path / "worked-hand.html"
    payload = build(tmp_path / "continuous", page)
    assert "__OBSERVER_DATA__" not in page.read_text()
    assert len(payload["views"]) == 4
    view = payload["views"][0]
    assert len(view["tokens"]) == 68 and len(view["trained"]) == 28
    assert len(view["units"][0][0]) == 32  # test model width 8
    assert sum(view["attention"][0][0][0][14:]) == 0
    staged.deals = 49
    with pytest.raises(ValueError, match="identical data"):
        run(staged)
    assert len(json.loads((tmp_path / "staged" / "deals.json").read_text())) == 50
    with pytest.raises(ValueError, match="overlapping"):
        load_deals([source, source], 50)


def test_tracking_calls_an_epoch_sampling_exposure_not_a_completed_pass():
    row = {"step": 2000, "train_batch_nll": 1.5,
           "validation": {"all": {"nll": 1.8, "top1": .4, "known_impossible_mass": .17}}}
    result = metrics(row, 64, 3988 * 4)
    assert result["epoch_equivalent"] == 2000 * 64 / (3988 * 4)
    assert result["validation_all_nll"] == 1.8
    assert not any(k.startswith("test") for k in result)

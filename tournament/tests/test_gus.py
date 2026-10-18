# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The fixed-contract Gus ladder preserves visibility, pins, and game identity."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from tournament import faceoff, seats, worker
from tournament.gus_seat import LADDER, gus_participant, make_seat, zeb_bid30
from lem.observer.model import Config, Observer
from lem.observer.data import VERSION
from lem.observer.play import ObserverPlay
from lem.tests.test_observer import recording
from lem.tests.test_observer_play import state_from_record


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Config(width=8, layers=1, heads=1)
    path = tmp_path / "gus.pt"
    torch.save({"config": config.__dict__, "model": Observer(config).state_dict(),
                "tokenizer": VERSION, "step": 0, "data_sha256": "fixture"}, path)
    p = gus_participant("g000", "a" * 40, hashlib.sha256(path.read_bytes()).hexdigest(), "b" * 40)
    monkeypatch.setattr(seats, "materialize", lambda _: tmp_path)
    return p, tmp_path


def test_contract_scoping_and_ladder_anchor(frozen):
    p, _ = frozen
    records = {p.id: p.to_record(), zeb_bid30("b"*40).id: zeb_bid30("b"*40).to_record()}
    caps = {"torch", "gus-observer", "node", "zeb-native"}
    assert worker.runnable(records, caps) == []
    assert len(worker.runnable(records, caps, ladder=LADDER)) == 2
    assert worker.runnable(records, {"torch"}, ladder=LADDER) == []
    assert worker.ladder_anchor(LADDER) == "zeb-0.4.1-contested-bid30"
    spec = faceoff.FaceoffSpec("marks7", p.id, p.id, 21, 1, "x")
    with pytest.raises(ValueError, match="contract"):
        faceoff.play_game(spec.__dict__, 0, 0, str(frozen[1] / "wrong.json"), records)


def test_pin_checks_include_cached_weights_and_changed_sources(frozen):
    p, path = frozen
    altered = replace(p, pin={**p.pin, "source_files": {"arena/bidders.py": "0"*64}})
    with pytest.raises(RuntimeError, match="source changed"):
        make_seat(altered, path)
    (path / "gus.pt").write_bytes(b"corrupt cache")
    with pytest.raises(RuntimeError, match="cached Gus"):
        make_seat(p, path)


def test_wrapper_matches_original_adapter_and_ignores_hidden_hands(frozen):
    p, path = frozen
    wrapped, original = seats.seat(p, "fixture"), ObserverPlay(path / "gus.pt", "cpu")
    state = state_from_record(recording())
    hands = list(state.hands)
    hands[1], hands[2] = hands[2], hands[1]
    other = replace(state, hands=tuple(hands))
    assert wrapped.choose([state, other], [30, 30]) == original.choose([state, state], [30, 30])
    with pytest.raises(ValueError, match="bid 30"):
        wrapped.choose([state], [31])


def test_same_policy_duplicate_games_repeat_exactly(frozen):
    p, path = frozen
    q = replace(p, id="gus-copy")
    records = {r.id: r.to_record() for r in (p, q)}
    spec = faceoff.FaceoffSpec(LADDER, p.id, q.id, 922, 1, "20261018T000000Z")
    games = [faceoff.play_game(spec.__dict__, 0, side, str(path / f"game-{i}.json"), records)
             for i, side in enumerate((0, 1, 0))]
    assert games[0]["hands"] == games[1]["hands"] == games[2]["hands"]
    assert all(h["bid_value"] == 30 and 0 <= h["decl_id"] <= 6 for h in games[0]["hands"])
    rec = faceoff.assemble(spec, records, games[:2], {}, "done")
    assert rec["summary"]["a_wins"] == rec["summary"]["b_wins"] == 1
    assert rec["rules"]["bidder"] == "Bid30Bidder"
    assert len(worker.games_of([json.loads(json.dumps(rec))])) == 2

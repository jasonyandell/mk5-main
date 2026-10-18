# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""A valid checkpoint/hash chain cannot conceal an altered learner input."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from forge.zeb.game import legal_actions
from lem.observer.audit import audit_generation, audit_panel
from lem.observer.data import VERSION
from lem.observer.model import Config, Observer
from lem.observer.population import atomic_json, collect, digest, make_specs
from lem.observer.reward import save_episodes, train_block


def test_audit_rejects_extra_input_even_after_checksums_are_updated(tmp_path, monkeypatch):
    class LegalFixture:
        def choose(self, states, bids):
            return [legal_actions(s)[0] for s in states]

    monkeypatch.setattr("lem.observer.reward.policies",lambda config,device:{
        name:LegalFixture() for name in ("g000","zeb")})
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=Observer(Config())
    initial=tmp_path/"initial.pt"
    torch.save({"config":asdict(model.config),"model":model.state_dict(),"tokenizer":VERSION,
                "data_sha256":"fixture","step":0},initial)
    field=tmp_path/"field";field.write_bytes(b"mock policy fixture")
    atomic_json(tmp_path/"specs.json",make_specs(16,"audit-fixture",["g000"],set()))
    atomic_json(tmp_path/"config.json",{
        "generation":"g001","blocks":1,"hands_per_block":16,"lr":3e-5,
        "pretraining_data_sha256":"fixture","pool":{"g000":{"path":str(initial),"sha256":digest(initial)}},
        "zeb_field":str(field),"zeb_field_sha256":digest(field),"specs_sha256":digest(tmp_path/"specs.json")})
    train_block(tmp_path,0,"cpu")
    atomic_json(tmp_path/"published.json",{"anonymous_model_and_replay_verified":True,"archive_revision":"fixture"})
    block=tmp_path/"block-00"
    generation={"id":"g001","layers":2,"sha256":digest(block/"checkpoint.pt")}
    result=audit_generation(tmp_path,generation,["g000"],set())
    assert result["hands"] == 16 and result["decisions"] == 112

    with np.load(block/"decisions.npz") as source:
        data={k:source[k] for k in source.files}
    data["x"][0,5]=(data["x"][0,5]+1)%28
    np.savez_compressed(block/"decisions.npz",**data)
    receipt=json.loads((block/"complete.json").read_text())
    receipt["decisions_sha256"]=digest(block/"decisions.npz")
    atomic_json(block/"complete.json",receipt)
    with pytest.raises(AssertionError):
        audit_generation(tmp_path,generation,["g000"],set())


def test_panel_rejects_a_frozen_player_changing_before_learner(tmp_path):
    class LegalFixture:
        def __init__(self,last=False): self.last=last
        def choose(self,states,bids):
            return [legal_actions(s)[-1 if self.last else 0] for s in states]

    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=Observer(Config(width=8,layers=1,heads=1)).eval()
    # Learner 1 is not the opening bidder, so the first choice is frozen.
    specs=[make_specs(4,"panel-fixture",["g000"],set())[1]]
    _,baseline=collect(model,{n:LegalFixture() for n in ("g000","zeb")},specs,"cpu",sample=False)
    _,changed=collect(model,{n:LegalFixture(True) for n in ("g000","zeb")},specs,"cpu",sample=False)
    baseline=json.loads(json.dumps(baseline))  # The audit reads archived JSON.
    folder=tmp_path/"g001/development";folder.mkdir(parents=True)
    atomic_json(tmp_path/"development-specs.json",specs)

    def write_panel(episodes):
        save_episodes(folder/"hands.jsonl.gz",episodes)
        wins=[(e["reward"]+1)/2 for e in episodes]
        delta=(episodes[0]["reward"]-baseline[0]["reward"])/2
        atomic_json(folder/"report.json",{"specs_sha256":digest(tmp_path/"development-specs.json"),
            "hands":1,"wins":wins,"team_win_rate":float(np.mean(wins)),
            "vs_reference":{"win_rate_difference":delta}})

    write_panel(baseline)
    assert audit_panel(tmp_path,"g001",baseline)["hands_with_different_play"] == 0
    write_panel(changed)
    # Both full games are legal, all scores and checksums are accurate, and
    # seat assignments match. The uncontrolled frozen-policy change is rejected.
    with pytest.raises(AssertionError):
        audit_panel(tmp_path,"g001",baseline)

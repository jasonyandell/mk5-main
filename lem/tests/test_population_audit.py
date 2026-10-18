# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""A valid checkpoint/hash chain cannot conceal an altered learner input."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from forge.zeb.game import legal_actions
from lem.observer.audit import audit_generation
from lem.observer.data import VERSION
from lem.observer.model import Config, Observer
from lem.observer.population import atomic_json, digest, make_specs
from lem.observer.reward import train_block


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

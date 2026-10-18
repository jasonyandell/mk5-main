# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Reward sign, lawful mixed routing, PPO gradients, growth and resume."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from forge.zeb.game import current_player, legal_actions
from lem.observer.data import VERSION
from lem.observer.campaign import plateau
from lem.observer.model import Config, Observer
from lem.observer.population import (active_pool, atomic_json, clipped_loss, collect,
                                    deepen, features, initial_state, load_model,
                                    log_distribution, make_specs, team_reward, update)
from lem.observer.reward import train_block
from scenario.rules import mask_of


@pytest.fixture
def tiny():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return Observer(Config(width=8, layers=1, heads=1)).eval()


def payload(model):
    return {"config": asdict(model.config), "model": model.state_dict(),
            "step": 0, "tokenizer": VERSION, "data_sha256": "test"}


class FirstLegal:
    def __init__(self):
        self.calls = []

    def choose(self, states, bids):
        self.calls.extend((tuple(s.hands), current_player(s)) for s in states)
        assert bids == [30]*len(states)
        return [min(legal_actions(s), key=lambda slot:s.hands[current_player(s)][slot]) for s in states]


def test_growth_is_identity_and_parameters_fit_target(tiny):
    saved = deepen(payload(tiny), 3, seed=7)
    larger = Observer(Config(**saved["config"])).eval()
    larger.load_state_dict(saved["model"])
    x = torch.randint(46, (3, 68))
    assert torch.equal(tiny(x), larger(x))
    # New output projections receive gradients, so identity insertion can learn.
    larger(x).square().mean().backward()
    assert larger.blocks[-1].projection.weight.grad.abs().sum() > 0
    with torch.device("meta"):
        final = Observer(Config(layers=15))
    assert sum(p.numel() for p in final.parameters()) == 757184
    with pytest.raises(ValueError):
        deepen(payload(tiny), 1, seed=7)


def test_reward_sign_for_bidder_partner_and_defenders():
    for bidder in range(4):
        for made in (False, True):
            points = [12,12]
            points[bidder%2] = 30 if made else 29
            points[1-bidder%2] = 42-points[bidder%2]
            rewards = [team_reward(points,bidder,s) for s in range(4)]
            assert rewards[bidder] == (1 if made else -1)
            assert rewards[(bidder+2)%4] == rewards[bidder]
            assert rewards[(bidder+1)%4] == -rewards[bidder]


def test_clipped_policy_gradient_direction_and_forced_move():
    for reward in (-1.,1.):
        z = torch.zeros((1,28), requires_grad=True)
        legal = torch.zeros_like(z, dtype=torch.bool); legal[:,:2] = True
        lp = z.masked_fill(~legal,-1e9).log_softmax(-1)
        loss,ratio,_ = clipped_loss(lp,torch.tensor([0]),torch.tensor([-np.log(2)],dtype=torch.float32),
                                   torch.tensor([reward]),entropy_weight=0.)
        assert torch.allclose(ratio,torch.ones(1))
        loss.backward()
        assert z.grad[0,0] * reward < 0
        assert z.grad[0,1] * reward > 0
    z = torch.randn((1,28),requires_grad=True)
    legal[:,:] = False; legal[:,0] = True
    lp = z.masked_fill(~legal,-1e9).log_softmax(-1)
    loss,_,entropy = clipped_loss(lp,torch.tensor([0]),torch.zeros(1),torch.ones(1))
    loss.backward()
    assert entropy.item() == 0
    assert torch.equal(z.grad,torch.zeros_like(z))


def test_specs_balance_roles_exclude_deals_and_retain_five():
    specs = make_specs(32,"test",["g0","g1"],set())
    assert specs == make_specs(32,"test",["g0","g1"],set())
    assert {(s["learner"],s["bidder"]) for s in specs} == {(a,b) for a in range(4) for b in range(4)}
    excluded = {tuple(mask_of(h) for h in specs[0]["hands"])}
    fresh = make_specs(32,"test",["g0","g1"],excluded)
    assert all(tuple(mask_of(h) for h in s["hands"]) not in excluded for s in fresh)
    generations = [f"g{i}" for i in range(8)]
    assert active_pool(generations) == generations[-5:]
    assert generations == [f"g{i}" for i in range(8)]
    assert active_pool(["g0"]) == ["g0"]


def test_plateau_requires_four_rounds_and_resets_on_material_gain():
    best, stale = .5, 0
    for score in [.50,.51,.515]:
        result=plateau(score,best,stale)
        assert not result["plateau"]
        best,stale=result["best"],result["stale"]
    assert plateau(.51,best,stale)["plateau"]
    result=plateau(.525,best,stale)
    assert result["improved"] and result["stale"] == 0 and not result["plateau"]
    with pytest.raises(ValueError):
        plateau(float("nan"),best,stale)


def test_only_visible_inputs_and_own_legality_reach_model():
    state = initial_state(make_specs(1,"visibility",["g0"],set())[0])
    seat = current_player(state)
    hidden = [i for i in range(4) if i != seat]
    hands = list(state.hands)
    a,b = hidden[:2]; hands[a],hands[b] = hands[b],hands[a]
    other = replace(state,hands=tuple(hands))
    for left,right in zip(features([state]),features([other])):
        np.testing.assert_array_equal(left,right)


def test_mixed_routing_sampling_probability_and_replay(tiny):
    specs = make_specs(16,"collect",["g0","g1"],set())
    policies = {name:FirstLegal() for name in ["g0","g1","zeb"]}
    data,episodes = collect(tiny,policies,specs,"cpu")
    assert len(data["action"]) == 16*7
    assert set(data["reward"]) <= {-1,1}
    for name,policy in policies.items():
        for hands,seat in policy.calls:
            spec = next(s for s in specs if tuple(map(tuple,s["hands"])) == hands)
            assert spec["seats"][seat] == name
        expected = 7*sum(s["seats"].count(name) for s in specs)
        assert len(policy.calls) == expected
    t = {k:torch.as_tensor(v) for k,v in data.items()}
    lp = log_distribution(tiny,t["x"],t["length"],t["legal"])
    chosen = lp.gather(1,t["action"][:,None]).squeeze(1)
    torch.testing.assert_close(chosen,t["old_logp"],atol=1e-6,rtol=1e-5)
    assert torch.allclose(t["old_probs"].sum(-1),torch.ones(len(chosen)),atol=1e-6)
    assert np.all(data["legal"][np.arange(len(chosen)),data["action"]])
    again,again_episodes = collect(tiny,{n:FirstLegal() for n in policies},specs,"cpu")
    assert episodes == again_episodes
    for key in data:
        np.testing.assert_array_equal(data[key],again[key])
    optimizer = torch.optim.AdamW(tiny.parameters(),lr=3e-5,weight_decay=0.,betas=(.9,.95))
    report = update(tiny,optimizer,data,0.,12,"cpu",passes=2)
    assert report["initial_logp_error"] < 1e-6
    assert report["passes_completed"] == 2


def test_block_resume_is_same_as_uninterrupted(tmp_path,tiny,monkeypatch):
    monkeypatch.setattr("lem.observer.reward.policies",lambda config,device:{n:FirstLegal() for n in ["g0","zeb"]})
    outputs = []
    for name in ["continuous","resumed"]:
        run = tmp_path/name;run.mkdir()
        torch.save(payload(tiny),run/"initial.pt")
        atomic_json(run/"config.json",{"blocks":2,"hands_per_block":8,"lr":3e-5,"generation":1,
                                       "pretraining_data_sha256":"test"})
        atomic_json(run/"specs.json",make_specs(16,"resume",["g0"],set()))
        first = train_block(run,0,"cpu")
        if name == "resumed":
            assert train_block(run,0,"cpu") == first
        train_block(run,1,"cpu")
        outputs.append(torch.load(run/"block-01/checkpoint.pt",weights_only=True))
    for key in outputs[0]["model"]:
        assert torch.equal(outputs[0]["model"][key],outputs[1]["model"][key])
    assert outputs[0]["prior_return"] == outputs[1]["prior_return"]
    assert outputs[0]["rl_updates"] == outputs[1]["rl_updates"]

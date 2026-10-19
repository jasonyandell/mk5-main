# SPDX-FileCopyrightText: 2026 Jason Yandell
# SPDX-License-Identifier: CC0-1.0
import json
import gzip
import signal

import numpy as np
import pytest
import torch

from lem.observer import outcomes
from lem.observer.value import ValueBaseline, actor_features, monte_carlo_advantages
from lem.observer.population import atomic_json
from lem.tests.test_outcomes import setup, Fixed
from lem.tests.test_rlvr import nested_equal


RECIPE = {'representation':'frozen_reference', 'estimator':'monte_carlo_batch_normalized',
          'lr':.001, 'steps':3, 'clip_norm':1.}


def test_monte_carlo_returns_and_prior_values_are_distinct_from_group_means():
    reward=np.array([-1,-1,1,1],np.float32);value=np.array([.2,.8,.4,.9],np.float32)
    returns,raw,adv,stats=monte_carlo_advantages(reward,value)
    np.testing.assert_array_equal(returns,[0,0,1,1])
    np.testing.assert_array_equal(raw,returns-value)
    expected=(raw.astype(np.float64)-raw.mean(dtype=np.float64))/raw.astype(np.float64).std()
    np.testing.assert_array_equal(adv,expected.astype(np.float32))
    assert adv[0]!=adv[1] and stats['zero_fraction']==0
    # Identical returns need not erase state-conditioned advantages. A constant
    # baseline with identical returns does have zero normalized signal.
    assert np.any(monte_carlo_advantages(np.ones(4),value)[2])
    assert not np.any(monte_carlo_advantages(np.ones(4),np.full(4,.5))[2])
    with pytest.raises(ValueError,match='verified reward'):
        monte_carlo_advantages(np.array([0]),np.array([.5]))


def test_frozen_causal_features_and_separate_critic_update(tmp_path):
    model,plan=setup(tmp_path);model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    data,_,_=outcomes.trajectories(model,{'g0':Fixed(),'zeb':Fixed()},plan['train'],3,'value-test','cpu')
    critic=ValueBaseline(model.config.width,RECIPE,'cpu')
    original_adv=data['advantage'].copy();stats=critic.attach(data,model,16)
    np.testing.assert_array_equal(data['old_value'],np.full(len(data['action']),.5))
    np.testing.assert_array_equal(data['group_advantage'],original_adv)
    x=data['x'].copy()
    for i,n in enumerate(data['length']):x[i,n:]=(x[i,n:]+19)%46
    np.testing.assert_array_equal(actor_features(model,x,data['length'],16,'cpu'),data['value_features'])
    old_adv=data['advantage'].copy();old_value=data['old_value'].copy()
    metrics,after=critic.update(data)
    assert metrics['optimizer_steps']==3 and metrics['parameters']==9 and np.isfinite(after).all()
    assert not np.array_equal(after,old_value)
    np.testing.assert_array_equal(data['advantage'],old_adv)
    np.testing.assert_array_equal(data['old_value'],old_value)
    nested_equal(model.state_dict(),before)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError,match='changed after advantage capture'):
        critic.update(data)


@pytest.mark.parametrize('failure', ['signal','crash'])
def test_critic_optimizer_interruption_recovers_actor_and_value(tmp_path,monkeypatch,failure):
    _,plan=setup(tmp_path);plan['recipe']['value']=RECIPE;atomic_json(tmp_path/'plan.json',plan)
    monkeypatch.setattr(outcomes,'policies',lambda *a:{'g0':Fixed(),'zeb':Fixed()})
    def args(name,resume=False):
        return outcomes.parser().parse_args(['--plan',str(tmp_path/'plan.json'),'--out',str(tmp_path/name),
            '--mode','ppo','--device','cpu']+(['--resume'] if resume else []))
    outcomes.run(args('whole'))
    original=torch.optim.Adam.step;calls=0
    def interrupt(self,*a,**kw):
        nonlocal calls
        result=original(self,*a,**kw)
        if type(self) is torch.optim.Adam:
            calls+=1
            if calls==1:
                if failure=='signal':signal.raise_signal(signal.SIGTERM)
                else:raise RuntimeError('critic interrupted mid-update')
        return result
    monkeypatch.setattr(torch.optim.Adam,'step',interrupt)
    if failure=='signal':assert outcomes.run(args('split'))['state']=='paused'
    else:
        with pytest.raises(RuntimeError,match='critic interrupted'):
            outcomes.run(args('split'))
    assert calls>0
    monkeypatch.setattr(torch.optim.Adam,'step',original)
    outcomes.run(args('split',True))
    for name in ('latest.pt','best.pt'):
        nested_equal(*[torch.load(tmp_path/folder/name,weights_only=True) for folder in ('whole','split')])
    saved=torch.load(tmp_path/'split/latest.pt',weights_only=True)
    assert saved['value']['steps']==6 and saved['settings']['schema']=='gus-full-hand-outcomes-v6'
    assert saved['value']['optimizer']['state']
    before={p.name:p.read_bytes() for p in (tmp_path/'split').iterdir() if p.suffix in ('.pt','.json')}
    outcomes.run(args('split',True))
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'split').iterdir() if p.suffix in ('.pt','.json')}
    plan['recipe']['value']=dict(RECIPE,steps=4);atomic_json(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='identity changed'):outcomes.run(args('split',True))


def test_declared_recipe_and_unmodified_advantages_are_required(tmp_path,monkeypatch):
    setup(tmp_path);monkeypatch.setattr(outcomes,'policies',lambda *a:{'g0':Fixed(),'zeb':Fixed()})
    args=outcomes.parser().parse_args(['--plan',str(tmp_path/'plan.json'),'--out',str(tmp_path/'out'),'--mode','ppo','--device','cpu'])
    with pytest.raises(ValueError,match='explicit value-baseline recipe'):outcomes.run(args)
    for patch in ({'steps':0},{'steps':1.5},{'lr':float('nan')},{'representation':'hidden_world'}):
        with pytest.raises(ValueError,match='value-baseline recipe'):
            ValueBaseline(8,dict(RECIPE,**patch),'cpu')


def test_ppo_actor_learns_a_different_update_from_matched_rehearsal(tmp_path,monkeypatch):
    _,plan=setup(tmp_path);plan['recipe']['value']=RECIPE;atomic_json(tmp_path/'plan.json',plan)
    monkeypatch.setattr(outcomes,'policies',lambda *a:{'g0':Fixed(),'zeb':Fixed()})
    for mode in ('ppo','rehearsal'):
        args=outcomes.parser().parse_args(['--plan',str(tmp_path/'plan.json'),'--out',str(tmp_path/mode),
            '--mode',mode,'--device','cpu'])
        outcomes.run(args)
    first=[]
    for mode in ('ppo','rehearsal'):
        with gzip.open(tmp_path/mode/'round-0001/hands.jsonl.gz','rt') as stream:
            first.append([json.loads(line) for line in stream])
    assert first[0]==first[1]
    initial=[torch.load(tmp_path/mode/'initial.pt',weights_only=True) for mode in ('ppo','rehearsal')]
    nested_equal(initial[0]['model'],initial[1]['model'])
    trained=[torch.load(tmp_path/mode/'latest.pt',weights_only=True) for mode in ('ppo','rehearsal')]
    assert any(not torch.equal(v,trained[1]['model'][k]) for k,v in trained[0]['model'].items() if k.startswith('backbone.'))
    first_data=dict(np.load(tmp_path/'ppo/round-0001/decisions.npz'))
    np.testing.assert_array_equal(first_data['old_value'],np.full(len(first_data['action']),.5))
    # The first fixture batch loses every hand; the second has one all-loss
    # root and one all-win root. Relative group advantages vanish in both.
    d=dict(np.load(tmp_path/'ppo/round-0002/decisions.npz'))
    assert np.any(d['advantage']) and not np.any(d['group_advantage'])
    np.testing.assert_array_equal(d['advantage'],monte_carlo_advantages(d['reward'],d['old_value'])[2])

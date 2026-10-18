# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Real signals and stop files preserve the next optimizer update exactly."""
from argparse import Namespace
import json
import os
import random
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.observer.splat.data import build
from lem.observer.splat.recovery import capture_rng, restore_rng, learning_rate
from lem.observer.splat.train import run
from lem.tests.test_splat_observer import make_source


def arguments(tmp_path, name, **kwargs):
    return Namespace(cache=[str(tmp_path/'cache')],out=str(tmp_path/name),steps=6,
        eval_every=2,batch=8,width=8,lr=.001,seed=42,device='cpu',resume=False,
        wandb_project=None,wandb_entity=None,wandb_name=None,**kwargs)


def equal_state(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal_state(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):equal_state(a,b)
    else:
        assert left==right


@pytest.mark.parametrize('stop_kind',[signal.SIGINT,signal.SIGTERM,'file'])
def test_interrupt_then_resume_exact_optimizer_rng_and_schedule(tmp_path,monkeypatch,stop_kind):
    make_source(tmp_path/'source');build(tmp_path/'source',tmp_path/'cache')
    original=torch.optim.AdamW.step
    counter=0;trigger=False
    stop=tmp_path/'STOP'
    def update(optimizer,*args,**kwargs):
        nonlocal counter
        result=original(optimizer,*args,**kwargs)
        # Exercise all saved RNGs: production currently only needs the sampler,
        # but an added stochastic operation must also reproduce after recovery.
        noise=(random.random()+float(np.random.random())+float(torch.rand(())))*1e-7
        with torch.no_grad():optimizer.param_groups[0]['params'][0].add_(noise)
        counter+=1
        if trigger and counter==3:
            if stop_kind=='file':stop.touch()
            else:os.kill(os.getpid(),stop_kind)
        return result
    monkeypatch.setattr(torch.optim.AdamW,'step',update)
    options=dict(schedule='cosine',min_lr=.0001,stop_file=str(stop))
    run(arguments(tmp_path,'full',**options))
    counter=0;trigger=True
    prior=signal.getsignal(signal.SIGINT)
    result=run(arguments(tmp_path,'split',**options))
    assert result['status']=='paused' and result['step']==3
    assert signal.getsignal(signal.SIGINT)==prior
    assert not (tmp_path/'split'/'report.json').exists()
    assert (tmp_path/'split'/'pause-000003.pt').exists()
    stop.unlink(missing_ok=True)
    resumed=arguments(tmp_path,'split',**options);resumed.resume=True
    run(resumed)
    a=torch.load(tmp_path/'full'/'latest.pt',weights_only=True)
    b=torch.load(tmp_path/'split'/'latest.pt',weights_only=True)
    for key in ('model','optimizer','sampler_state','rng_state','schedule','step','best_loss'):
        equal_state(a[key],b[key])
    for left,right in zip(a['history'],b['history']):
        equal_state({k:v for k,v in left.items() if k!='segment_seconds'},
                    {k:v for k,v in right.items() if k!='segment_seconds'})
    bad=arguments(tmp_path,'split',**options);bad.resume=True;bad.steps=8
    with pytest.raises(ValueError,match='horizon'):run(bad)


def test_rng_safe_load_and_schedule_endpoints(tmp_path):
    state=capture_rng('cpu');torch.save(state,tmp_path/'rng.pt')
    expected=(random.random(),float(np.random.random()),torch.rand(4))
    restore_rng(torch.load(tmp_path/'rng.pt',weights_only=True))
    actual=(random.random(),float(np.random.random()),torch.rand(4))
    equal_state(expected,actual)
    schedule=dict(kind='cosine',peak=.001,minimum=.0001,steps=11)
    assert learning_rate(schedule,1)==.001
    assert learning_rate(schedule,11)==.0001
    assert learning_rate(schedule,6)==pytest.approx(.00055)


def test_preexisting_stop_and_fork_inherit_full_state(tmp_path):
    make_source(tmp_path/'source');build(tmp_path/'source',tmp_path/'cache')
    run(arguments(tmp_path,'parent'))
    source=tmp_path/'parent'/'latest.pt';stop=tmp_path/'STOP';stop.touch()
    args=arguments(tmp_path,'fork',fork_from=str(source),stop_file=str(stop),schedule='cosine')
    result=run(args)
    assert result['step']==0
    parent=torch.load(source,weights_only=True);initial=torch.load(tmp_path/'fork'/'initial.pt',weights_only=True)
    for key in ('model','optimizer','sampler_state','rng_state'):equal_state(parent[key],initial[key])
    stop.unlink();args.resume=True
    report=run(args)
    assert report['parent']['step']==6
    assert (tmp_path/'fork'/'initial.json').exists()
    assert report['steps']==6


def test_wandb_identity_survives_interruption(tmp_path,monkeypatch):
    make_source(tmp_path/'source');build(tmp_path/'source',tmp_path/'cache')
    calls=[]
    class Tracker:
        id='fixed-id';url='https://example.invalid/fixed-id';entity='entity';project='project'
        summary={}
        def define_metric(self,*args,**kwargs):pass
        def log(self,*args,**kwargs):pass
        def finish(self,**kwargs):calls.append(('finish',kwargs))
    def initialize(**kwargs):calls.append(('init',kwargs));return Tracker()
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=initialize))
    stop=tmp_path/'STOP';stop.touch()
    args=arguments(tmp_path,'tracked',stop_file=str(stop))
    args.wandb_project='project';args.wandb_entity='entity';args.wandb_name='name'
    run(args);stop.unlink();args.resume=True;run(args)
    inits=[x for kind,x in calls if kind=='init']
    assert len(inits)==2 and inits[1]['id']=='fixed-id' and inits[1]['resume']=='allow'
    saved=torch.load(tmp_path/'tracked'/'latest.pt',weights_only=True)
    assert saved['run_identity']==json.loads((tmp_path/'tracked'/'wandb-run.json').read_text())

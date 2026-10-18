# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Weighted lawful targets, unchanged information, causal heads and loss masks."""
from argparse import Namespace
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from lem.observer.data import QUERIES, prepare, VERSION
from lem.observer.model import Config
from lem.observer.splat.data import histograms, build, Corpus
from lem.observer.splat.model import SplatObserver, joint_loss, TASK
from lem.tests.test_observer import recording
from scenario.rules import tiles_of


def test_weighted_histogram_and_reject_malformed():
    h = histograms(np.array([-42,18,42,0,18,42]),np.array([1.,2.,3.]),[2],[3])
    assert h.shape == (2,43)
    np.testing.assert_allclose(h.sum(-1),1)
    np.testing.assert_allclose(h[:,30:].sum(-1),5/6)
    assert h[0,0] == 1/6 and h[1,21] == 1/6
    with pytest.raises(ValueError,match="integral"):
        histograms(np.array([1]),np.ones(1),[1],[1])
    with pytest.raises(ValueError,match="zero"):
        histograms(np.array([0]),np.zeros(1),[1],[1])


def test_same_backbone_and_both_heads_causal():
    torch.manual_seed(42)
    m = SplatObserver(Config(width=8,layers=1,heads=1)).eval()
    d = prepare([recording()])
    x = torch.tensor(d['x'])
    tile,splat = m(x)
    torch.testing.assert_close(tile,m.backbone(x),atol=0,rtol=0)
    y = x.clone(); y[0,24:] = (y[0,24:]+1)%46; y[1:] = 0
    t,s = m(y)
    torch.testing.assert_close(tile[0,:24],t[0,:24],atol=0,rtol=0)
    torch.testing.assert_close(splat[0,:24],s[0,:24],atol=0,rtol=0)
    pos = torch.tensor(np.broadcast_to(QUERIES,d['own'].shape)[d['own']].reshape(4,7))
    _, selected = m(x,pos)
    torch.testing.assert_close(selected,splat[torch.arange(4)[:,None],pos])


def test_unlabelled_candidates_have_zero_gradient_and_position_weighting():
    t = torch.zeros(1,28,28,requires_grad=True)
    s = torch.zeros(1,2,28,43,requires_grad=True)
    h = torch.zeros(1,2,7,43); h[0,0,0,42]=1
    c = torch.zeros(1,2,7,dtype=torch.long)
    loss,tl,sl = joint_loss(t,torch.zeros(1,28,dtype=torch.long),s,h,c)
    assert sl.item() == pytest.approx(np.log(43))
    loss.backward()
    assert s.grad[0,0,0,42] < 0
    assert torch.count_nonzero(s.grad[0,1]) == 0
    assert torch.count_nonzero(s.grad[0,0,1:]) == 0


def make_source(path, n=30):
    path.mkdir()
    records = [recording(i+101) for i in range(n)]
    deals=[];ks=[];actions=[]
    for i,r in enumerate(records):
        p = prepare([r])
        for k in range(28):
            a = np.flatnonzero(p['legal'][0,k])
            if len(a)>1:
                deals.append(i);ks.append(k);actions.append(a)
    na = np.array(list(map(len,actions)))
    np.savez_compressed(path/'pour.npz',hands=np.array([r['hands'] for r in records]),
        decl=np.array([r['decl'] for r in records]),plays=np.array([r['plays'] for r in records]),
        deal=np.array(deals),n_plays=np.array(ks),n_actions=na,n_worlds=np.full(len(na),2),
        actions=np.concatenate(actions),q=np.full(sum(na),.75),weights=np.tile([1.,3.],len(na)),
        outcomes=np.tile(np.array([-42,42],np.int8),sum(na)))
    (path/'manifest.json').write_text(json.dumps(dict(bidder=0,bid=30,utility='make',deals=n,
        deal_seeds=[r['seed'] for r in records],commit='test',spec='test')))
    return records


def test_cache_alignment_same_inputs_and_no_worlds_required(tmp_path):
    records=make_source(tmp_path/'source')
    build(tmp_path/'source',tmp_path/'cache')
    corpus=Corpus([tmp_path/'cache'])
    p=prepare(records)
    b=corpus.batch(np.arange(4*len(records)))
    np.testing.assert_array_equal(b['x'],p['x'])
    np.testing.assert_array_equal(b['targets'],p['targets'])
    assert set(corpus.split)=={0,1,2}
    for row in range(len(b['x'])):
        for own,k in enumerate((b['own_positions'][row]-13)//2):
            mask=b['histograms'][row,own].sum(-1)>0
            expected=np.flatnonzero(p['legal'][row,k])
            if len(expected)>1:
                np.testing.assert_array_equal(b['candidates'][row,own,mask],expected)
                np.testing.assert_allclose(b['histograms'][row,own,mask,30:].sum(-1),.75)
            else:
                assert not mask.any()
    with pytest.raises(ValueError,match='overlap'):
        Corpus([tmp_path/'cache',tmp_path/'cache'])


def test_joint_training_resume_and_reject_changed_recipe(tmp_path):
    from lem.observer.splat.train import run
    make_source(tmp_path/'source')
    build(tmp_path/'source',tmp_path/'cache')
    def args(name,steps,resume=False):
        return Namespace(cache=[str(tmp_path/'cache')],out=str(tmp_path/name),steps=steps,
            eval_every=2,batch=8,width=8,lr=.001,seed=42,device='cpu',resume=resume,
            wandb_project=None,wandb_entity=None,wandb_name=None)
    run(args('full',4));run(args('split',2));run(args('split',4,True))
    a=torch.load(tmp_path/'full'/'latest.pt',weights_only=True)
    b=torch.load(tmp_path/'split'/'latest.pt',weights_only=True)
    for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
    torch.testing.assert_close(a['sampler_state'],b['sampler_state'],atol=0,rtol=0)
    bad=args('split',6,True);bad.lr=.002
    with pytest.raises(ValueError,match='settings'):run(bad)


def test_splat_policy_defender_sign_visibility_and_legality(tmp_path):
    from dataclasses import replace
    from lem.observer.splat.play import SplatPlay
    from lem.tests.test_observer_play import state_from_record
    from forge.zeb.game import apply_action,legal_actions,current_player
    m=SplatObserver(Config(width=8,layers=1,heads=1))
    path=tmp_path/'splat.pt'
    torch.save(dict(task=TASK,tokenizer=VERSION,config=asdict(m.config),model=m.state_dict(),
                    step=0,data_sha256='fixture'),path)
    player=SplatPlay(path)
    class Scores(torch.nn.Module):
        def forward(self,x,positions):
            logits=torch.zeros(len(x),1,28,43)
            logits[...,42]=torch.arange(28).float()/2
            return torch.zeros(len(x),x.shape[1],28),logits
    player.model=Scores()
    state=state_from_record(recording())
    for _ in range(28):
        actor=current_player(state);hand=state.hands[actor]
        legal=legal_actions(state)
        expected=(max if actor%2==state.bidder%2 else min)(hand[s] for s in legal)
        choice=player.choose([state],[30])[0]
        assert hand[choice]==expected
        hidden=list(state.hands);others=[s for s in range(4) if s!=actor]
        hidden[others[0]],hidden[others[1]]=hidden[others[1]],hidden[others[0]]
        assert player.choose([replace(state,hands=tuple(hidden))],[30],[(0,6)])==[choice]
        state=apply_action(state,legal[0])

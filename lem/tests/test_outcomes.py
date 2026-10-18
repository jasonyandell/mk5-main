# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
from dataclasses import asdict
import copy
import json
import signal

import numpy as np
import pytest
import torch

from forge.zeb.game import legal_actions
from lem.observer import outcomes
from lem.observer.data import VERSION
from lem.observer.model import Config
from lem.observer.population import atomic_json, atomic_save, digest, make_specs
from lem.observer.splat.model import SplatObserver, TASK
from lem.tests.test_rlvr import nested_equal
from lem.tests.test_rlvr_rehearsal import replay_fixture


class Fixed:
    def choose(self, states, bids):
        return [legal_actions(s)[0] for s in states]


def setup(tmp_path):
    torch.manual_seed(9); torch.set_num_threads(1)
    model = SplatObserver(Config(width=8, layers=1, heads=2)).eval()
    initial = tmp_path/'initial.pt'
    atomic_save(initial, {'model': model.state_dict(), 'task': TASK, 'tokenizer': VERSION,
        'config': asdict(model.config), 'step': 0, 'data_sha256': 'test'})
    replay_fixture(tmp_path/'replay')
    specs = make_specs(8, 'outcome-unit-test', ['g0'], [])
    for s in specs:
        s['seats'] = ['zeb']*4; s['seats'][s['learner']] = 'learner'; s['seats'][(s['learner']+2)%4] = 'g0'
    plan = {'initial': str(initial), 'initial_sha256': digest(initial),
        'reference': str(initial), 'reference_sha256': digest(initial), 'replay': str(tmp_path/'replay'),
        'replay_sha256': digest(tmp_path/'replay/rehearsal.npz'), 'seed': 12, 'namespace': 'outcome-unit', 'field': {},
        'train': specs[:4], 'development': specs[4:6], 'confirmation': specs[6:],
        'recipe': {'rounds': 2, 'roots': 2, 'group': 3, 'inner': 2, 'batch': 16, 'eval_every': 1,
            'replay_batch': 3, 'replay_weight': 1., 'lr': .001, 'beta': .02, 'clip': .2,
            'retention_nll': .02, 'retention_splat': .02, 'reward_horizon': 'full'}}
    atomic_json(tmp_path/'plan.json', plan)
    return model, plan


def distinct_start(tmp_path, plan):
    saved = torch.load(plan['reference'], weights_only=True)
    generator = torch.Generator().manual_seed(51)
    weights = saved['model']['backbone.token.weight']
    weights.add_(.002 * torch.randn(weights.shape, generator=generator))
    path = tmp_path/'warm-start.pt'; atomic_save(path, saved)
    plan.update(initial=str(path), initial_sha256=digest(path))
    atomic_json(tmp_path/'plan.json', plan)


def test_full_hand_grouping_audit_and_replay_bytes(tmp_path):
    model, plan = setup(tmp_path); field = {'g0': Fixed(), 'zeb': Fixed()}
    a, episodes, stats = outcomes.trajectories(model, field, plan['train'], 8, 'same', 'cpu')
    b, repeated, _ = outcomes.trajectories(model, field, plan['train'], 8, 'same', 'cpu')
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert episodes == repeated and stats['hands'] == 32
    assert len({e['action_seed'] for e in episodes}) == 32
    assert len({e['root_deal_seed'] for e in episodes}) == 4
    assert any(len({tuple(map(tuple, e['plays'])) for e in episodes[i:i+8]}) > 1 for i in range(0,32,8))
    for i in range(0, 32, 8):
        ids = np.isin(a['hand'], range(i,i+8))
        assert a['advantage'][ids].mean() == pytest.approx(0, abs=1e-6)
    outcomes.save_episodes(tmp_path/'a.gz', episodes); outcomes.save_episodes(tmp_path/'b.gz', repeated)
    assert (tmp_path/'a.gz').read_bytes() == (tmp_path/'b.gz').read_bytes()
    bad = {k:v.copy() for k,v in a.items()}; bad['x'][0, 0] = 45
    with pytest.raises(ValueError, match='input/output replay mismatch'):
        outcomes.audit(bad, episodes)
    bad = copy.deepcopy(episodes); bad[0]['reward'] *= -1
    with pytest.raises(ValueError, match='wrong team reward'):
        outcomes.audit(a, bad)


@pytest.mark.parametrize('mode,horizon,distinct', [('grpo','full',False), ('grpo','unsettled',False),
    ('rehearsal','full',False), ('grpo','full',True)])
def test_actual_sigterm_and_crash_recovery(tmp_path, monkeypatch, mode, horizon, distinct):
    _, plan = setup(tmp_path)
    if distinct:
        distinct_start(tmp_path, plan)
    plan['recipe']['reward_horizon'] = horizon; atomic_json(tmp_path/'plan.json', plan)
    monkeypatch.setattr(outcomes, 'policies', lambda *args: {'g0':Fixed(), 'zeb':Fixed()})
    def args(name, resume=False):
        return outcomes.parser().parse_args(['--plan', str(tmp_path/'plan.json'), '--out', str(tmp_path/name),
            '--mode', mode, '--device', 'cpu'] + (['--resume'] if resume else []))
    outcomes.run(args('whole'))
    original = torch.optim.AdamW.step; calls = 0
    def interrupt(self, *a, **kw):
        nonlocal calls
        result = original(self, *a, **kw); calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupt)
    assert outcomes.run(args('split'))['state'] == 'paused'
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    outcomes.run(args('split', True))
    status = json.loads((tmp_path/'split/status.json').read_text())
    report = json.loads((tmp_path/'split/report.json').read_text())
    assert status == report and status['state'] == 'complete'
    for name in ('latest.pt', 'best.pt'):
        a,b = [torch.load(tmp_path/p/name, weights_only=True) for p in ('whole','split')]
        nested_equal(a,b)
    # Simulate hard failure after weights changed but before a round committed.
    def fail(self, *a, **kw):
        original(self, *a, **kw)
        raise RuntimeError('simulated crash')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError, match='simulated crash'):
        outcomes.run(args('crashed'))
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    outcomes.run(args('crashed', True))
    nested_equal(torch.load(tmp_path/'whole/latest.pt', weights_only=True),
                 torch.load(tmp_path/'crashed/latest.pt', weights_only=True))
    assert outcomes.run(args('split', True))['state'] == 'complete'
    plan['recipe']['lr'] = .02; atomic_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='identity changed'):
        outcomes.run(args('split', True))


def test_zero_variance_and_split_rejection(tmp_path):
    _, plan = setup(tmp_path)
    adv = outcomes.advantages(torch.tensor([[1.,1.,1.],[0.,0.,0.],[1.,0.,0.]]))
    assert torch.equal(adv[:2], torch.zeros(2,3))
    assert float(adv[2].mean()) == pytest.approx(0, abs=1e-7)
    replay = outcomes.Rehearsal(plan['replay'], 'cpu')
    outcomes.validate_plan(plan, replay)
    plan['confirmation'][0] = plan['train'][0]
    with pytest.raises(ValueError, match='overlap'):
        outcomes.validate_plan(plan, replay)


def test_completed_resume_does_not_change_artifact_bytes(tmp_path, monkeypatch):
    setup(tmp_path)
    monkeypatch.setattr(outcomes, 'policies', lambda *args: {'g0':Fixed(), 'zeb':Fixed()})
    common=['--plan', str(tmp_path/'plan.json'), '--out', str(tmp_path/'run'), '--mode', 'grpo', '--device', 'cpu']
    report=outcomes.run(outcomes.parser().parse_args(common))
    before={p.name:p.read_bytes() for p in (tmp_path/'run').iterdir() if p.suffix in ('.pt', '.json')}
    assert outcomes.run(outcomes.parser().parse_args(common+['--resume'])) == report
    assert before == {p.name:p.read_bytes() for p in (tmp_path/'run').iterdir() if p.suffix in ('.pt', '.json')}
    assert digest(tmp_path/'run/best.pt') == report['best_sha256']


def test_starting_policy_is_separate_from_frozen_reference_and_retention(tmp_path, monkeypatch):
    reference, plan = setup(tmp_path); distinct_start(tmp_path, plan)
    monkeypatch.setattr(outcomes, 'policies', lambda *args: {'g0':Fixed(), 'zeb':Fixed()})
    original_update = outcomes.update; seen = []
    def checked_update(model, frozen, *args):
        assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
        nested_equal(reference.state_dict(), frozen.state_dict())
        result, distributions = original_update(model, frozen, *args)
        assert not np.array_equal(distributions['before'], distributions['reference'])
        nested_equal(reference.state_dict(), frozen.state_dict())
        seen.append(True)
        return result, distributions
    monkeypatch.setattr(outcomes, 'update', checked_update)
    common = ['--plan', str(tmp_path/'plan.json'), '--out', str(tmp_path/'run'), '--mode', 'grpo', '--device', 'cpu']
    report = outcomes.run(outcomes.parser().parse_args(common))
    assert len(seen) == 2
    assert report['reference_retention'] == outcomes.Rehearsal(plan['replay'], 'cpu').evaluate(reference)
    initial = torch.load(tmp_path/'run/initial.pt', weights_only=True)
    nested_equal(initial['model'], torch.load(plan['initial'], weights_only=True)['model'])
    # A different declared reference must invalidate exact recovery, even if
    # architecture and all training hyperparameters remain unchanged.
    plan.update(reference=plan['initial'], reference_sha256=plan['initial_sha256'])
    atomic_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='resume identity changed'):
        outcomes.run(outcomes.parser().parse_args(common+['--resume']))


def test_reference_bytes_and_initial_feasibility_are_required(tmp_path, monkeypatch):
    _, plan = setup(tmp_path); distinct_start(tmp_path, plan)
    monkeypatch.setattr(outcomes, 'policies', lambda *args: {'g0':Fixed(), 'zeb':Fixed()})
    common = ['--plan', str(tmp_path/'plan.json'), '--out', str(tmp_path/'run'), '--mode', 'grpo', '--device', 'cpu']
    good = plan['reference_sha256']; plan['reference_sha256'] = 'changed'
    atomic_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='reference identity changed'):
        outcomes.run(outcomes.parser().parse_args(common))
    plan['reference_sha256'] = good
    saved = torch.load(plan['initial'], weights_only=True)
    saved['model']['outcomes.bias'][0] = 100
    atomic_save(plan['initial'], saved); plan['initial_sha256'] = digest(plan['initial'])
    atomic_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='initial policy violates reference retention bounds'):
        outcomes.run(outcomes.parser().parse_args(common))
    assert not list((tmp_path/'run').glob('*.pt'))


def test_training_output_has_one_writer(tmp_path):
    import fcntl
    out=tmp_path/'run'; out.mkdir()
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args=outcomes.parser().parse_args(['--plan','unused','--out',str(out),'--mode','grpo'])
        with pytest.raises(ValueError, match='already owned'):
            outcomes.run(args)


def test_public_settlement_boundaries_and_remaining_points():
    # Exhaust every publicly possible banked-score pair and allocation of what
    # remains. This is a point-total proof, independent of either game engine.
    for bidder in range(4):
        for a in range(43):
            for b in range(43-a):
                points=(a,b); made=points[bidder%2]>=30
                settled=outcomes.publicly_settled(points,bidder)
                futures={points[bidder%2]+extra>=30 for extra in range(43-a-b)}
                assert settled == (len(futures)==1)
                if settled:
                    assert futures=={made}
    assert not outcomes.publicly_settled((29,12),0)
    assert outcomes.publicly_settled((30,0),0)
    assert outcomes.publicly_settled((0,13),0)


def test_settlement_mask_follows_public_replay_and_rejects_tampering(tmp_path):
    model,plan=setup(tmp_path)
    data,episodes,stats=outcomes.trajectories(model, {'g0':Fixed(),'zeb':Fixed()}, plan['train'],4,'mask','cpu')
    assert data['pre_settlement'].any() and (~data['pre_settlement']).any()
    outcomes.audit(data,episodes)
    for hand in range(len(episodes)):
        active=data['pre_settlement'][data['hand']==hand]
        assert not np.any(np.diff(active.astype(int))>0)
    data['pre_settlement'][0] = ~data['pre_settlement'][0]
    with pytest.raises(ValueError,match='settlement mask'):
        outcomes.audit(data,episodes)


def test_post_terminal_score_terms_have_zero_mean_but_sample_noise():
    # One relevant binary decision determines success; a later independent
    # binary move has no possible effect. Exact expectation cancels its score.
    theta=torch.tensor([.4,-.7],requires_grad=True)
    p=theta.sigmoid(); gradients=[]; probabilities=[]
    for first in (0,1):
        for later in (0,1):
            prob=(p[0] if first else 1-p[0])*(p[1] if later else 1-p[1])
            logp=torch.stack([torch.log(p[0] if first else 1-p[0]),torch.log(p[1] if later else 1-p[1])])
            full,=torch.autograd.grad(-first*logp.sum()/2,theta,retain_graph=True)
            cut,=torch.autograd.grad(-first*logp[0]/2,theta,retain_graph=True)
            gradients.append((full,cut));probabilities.append(prob.detach())
    mean_full=sum(w*g[0] for w,g in zip(probabilities,gradients))
    mean_cut=sum(w*g[1] for w,g in zip(probabilities,gradients))
    torch.testing.assert_close(mean_full,mean_cut,atol=1e-7,rtol=1e-6)
    assert any(float(g[0][1])!=0 for g in gradients)
    assert all(float(g[1][1])==0 for g in gradients)


def test_public_mask_uses_independent_partial_score_referee(tmp_path,monkeypatch):
    from dataclasses import replace
    model,plan=setup(tmp_path)
    data,episodes,_=outcomes.trajectories(model,{'g0':Fixed(),'zeb':Fixed()},plan['train'],2,'referee','cpu')
    original=outcomes.apply_action
    monkeypatch.setattr(outcomes,'apply_action',lambda *args:replace(original(*args),team_points=(1,0)))
    with pytest.raises(ValueError,match='partial-score replay mismatch'):
        outcomes.audit(data,episodes)


def test_masked_post_terminal_credit_equals_rehearsal_update(tmp_path):
    model,plan=setup(tmp_path)
    data,_,_=outcomes.trajectories(model,{'g0':Fixed(),'zeb':Fixed()},plan['train'],4,'gradient','cpu')
    # Nonzero advantages only where the public outcome is already fixed.
    data['advantage']=(~data['pre_settlement']).astype(np.float32)
    reference=copy.deepcopy(model).requires_grad_(False)
    replay=outcomes.Rehearsal(plan['replay'],'cpu');trained={}
    for name,mode,horizon in [('masked','grpo','unsettled'),('control','rehearsal','full'),('full','grpo','full')]:
        candidate=copy.deepcopy(model);opt=torch.optim.AdamW(candidate.parameters(),lr=.001)
        # Isolate reward credit from floating-point cancellation in the KL gradient.
        recipe=dict(plan['recipe'],mode=mode,reward_horizon=horizon,inner=1,beta=0.)
        outcomes.update(candidate,reference,opt,data,replay,torch.Generator().manual_seed(4),recipe,'cpu')
        trained[name]=candidate.state_dict()
    for name in trained['masked']:
        assert torch.equal(trained['masked'][name],trained['control'][name]),name
    assert any(not torch.equal(trained['full'][name],trained['control'][name]) for name in trained['full'])

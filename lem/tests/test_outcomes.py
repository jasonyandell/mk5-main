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
            'retention_nll': .02, 'retention_splat': .02, 'reward_horizon': 'full', 'selection': 'retention'}}
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


@pytest.mark.parametrize('mode,horizon,distinct,unanchored', [('grpo','full',False,False), ('grpo','unsettled',False,False),
    ('rehearsal','full',False,False), ('grpo','full',True,False), ('ppo','full',False,False), ('ppo','unsettled',False,False),
    ('grpo','full',False,True), ('rloo','full',False,True)])
def test_actual_sigterm_and_crash_recovery(tmp_path, monkeypatch, mode, horizon, distinct, unanchored):
    _, plan = setup(tmp_path)
    if distinct:
        distinct_start(tmp_path, plan)
    if unanchored:
        plan['recipe'].update(replay_weight=0., beta=0., selection='outcomes')
    if mode == 'ppo':
        plan['recipe']['value'] = {'representation':'frozen_reference', 'estimator':'monte_carlo_batch_normalized',
                                   'lr':.001, 'steps':3, 'clip_norm':1.}
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


def test_unanchored_update_ignores_teacher_targets_and_preserves_unused_head(tmp_path):
    model, plan = setup(tmp_path)
    data, _, _ = outcomes.trajectories(model, {'g0':Fixed(),'zeb':Fixed()}, plan['train'], 8, 'unanchored', 'cpu')
    assert np.any(data['advantage'])
    recipe = dict(plan['recipe'], mode='grpo', replay_weight=0., beta=0.)
    trained = []
    for changed in (False, True):
        candidate = copy.deepcopy(model); reference = copy.deepcopy(model).requires_grad_(False)
        replay = outcomes.Rehearsal(plan['replay'], 'cpu')
        if changed:
            with torch.no_grad():
                reference.backbone.token.weight.add_(torch.randn_like(reference.backbone.token.weight))
            replay.arrays['targets'] = (replay.arrays['targets']+1)%28
            replay.arrays['histograms'] = torch.roll(replay.arrays['histograms'],1,-1)
        opt = torch.optim.AdamW(candidate.parameters(), lr=.001)
        outcomes.update(candidate,reference,opt,data,replay,torch.Generator().manual_seed(4),recipe,'cpu')
        assert all(p.grad is None for p in candidate.outcomes.parameters())
        nested_equal(candidate.outcomes.state_dict(), model.outcomes.state_dict())
        trained.append(candidate.state_dict())
    nested_equal(*trained)
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in trained[0].items())


def test_outcome_selection_can_choose_a_winner_despite_imitation_drift(tmp_path, monkeypatch):
    _, plan = setup(tmp_path)
    monkeypatch.setattr(outcomes,'policies',lambda *a:{'g0':Fixed(),'zeb':Fixed()})
    original = outcomes.trajectories
    for selection in ('retention','outcomes'):
        counters = {'dev':0,'retention':0}
        def trajectories(*a, **kw):
            data,episodes,stats = original(*a, **kw)
            if a[4].endswith('/development'):
                stats['win_rate'] = (counters['dev']//2)/2
                counters['dev'] += 1
            return data,episodes,stats
        def retention(*a, **kw):
            drift = counters['retention'] >= 2
            counters['retention'] += 1
            return {'tile_nll':float(drift), 'splat_ce':float(drift), 'tile_accuracy':0.}
        monkeypatch.setattr(outcomes,'trajectories',trajectories)
        monkeypatch.setattr(outcomes.Rehearsal,'evaluate',retention)
        plan['recipe']['selection']=selection;atomic_json(tmp_path/'plan.json',plan)
        args=outcomes.parser().parse_args(['--plan',str(tmp_path/'plan.json'),'--out',str(tmp_path/selection),'--mode','grpo','--device','cpu'])
        report=outcomes.run(args)
        assert report['best']['round'] == (2 if selection=='outcomes' else 0)
        assert report['best']['selection_eligible']
        assert report['best']['retention_pass'] == (selection=='retention')


@pytest.mark.parametrize('patch,mode,match', [({'selection':'unknown'},'grpo','declare selection'),
    ({'replay_weight':0.},'rehearsal','invalid full-hand'), ({'replay_weight':-1.},'grpo','invalid full-hand')])
def test_invalid_selection_and_empty_rehearsal_are_rejected(tmp_path, patch, mode, match):
    _,plan=setup(tmp_path);plan['recipe'].update(patch);atomic_json(tmp_path/'plan.json',plan)
    args=outcomes.parser().parse_args(['--plan',str(tmp_path/'plan.json'),'--out',str(tmp_path/'out'),'--mode',mode,'--device','cpu'])
    with pytest.raises(ValueError,match=match):outcomes.run(args)


@pytest.mark.parametrize('probability', [.13, .5, .82])
def test_leave_one_out_exact_expected_policy_gradient(probability):
    # Exhaust all 2^3 independent binary completions. For a Bernoulli logit,
    # d E[success]/d logit = p(1-p); baseline must preserve that expectation.
    import itertools
    rewards = torch.tensor(list(itertools.product((0., 1.), repeat=3)), dtype=torch.float64)
    probability_mass = torch.where(rewards.bool(), torch.full_like(rewards, probability), torch.full_like(rewards, 1-probability)).prod(-1)
    advantage = outcomes.leave_one_out(rewards)
    estimate = (probability_mass * (advantage * (rewards-probability)).mean(-1)).sum()
    assert float(estimate) == pytest.approx(probability*(1-probability), abs=1e-12)
    for i in range(3):
        others = torch.cat((rewards[:, :i], rewards[:, i+1:]), dim=1).mean(-1)
        torch.testing.assert_close(advantage[:, i], rewards[:, i]-others)
    assert torch.equal(advantage[[0,-1]], torch.zeros(2,3,dtype=torch.float64))


def test_leave_one_out_uses_verified_success_and_preserves_sampling(tmp_path):
    model, plan = setup(tmp_path); field = {'g0':Fixed(), 'zeb':Fixed()}
    standard, first, _ = outcomes.trajectories(model,field,plan['train'],8,'paired','cpu')
    actual, episodes, _ = outcomes.trajectories(model,field,plan['train'],8,'paired','cpu',estimator='rloo')
    assert first == episodes
    rewards = np.array([(e['reward']+1)/2 for e in episodes]).reshape(-1,8)
    expected = np.empty_like(rewards)
    for i in range(8):
        expected[:,i] = rewards[:,i] - np.delete(rewards,i,axis=1).mean(-1)
    np.testing.assert_allclose(actual['advantage'],expected.ravel()[actual['hand']],atol=1e-7)
    for k in actual:
        if k != 'advantage':
            assert np.array_equal(actual[k], standard[k])
    for bad in (torch.ones(2,1), torch.ones(2), torch.tensor([[0.,float('nan')]])):
        with pytest.raises(ValueError,match='finite groups'):
            outcomes.leave_one_out(bad)

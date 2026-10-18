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
    plan = {'initial': str(initial), 'initial_sha256': digest(initial), 'replay': str(tmp_path/'replay'),
        'replay_sha256': digest(tmp_path/'replay/rehearsal.npz'), 'seed': 12, 'namespace': 'outcome-unit', 'field': {},
        'train': specs[:4], 'development': specs[4:6], 'confirmation': specs[6:],
        'recipe': {'rounds': 2, 'roots': 2, 'group': 3, 'inner': 2, 'batch': 16, 'eval_every': 1,
            'replay_batch': 3, 'replay_weight': 1., 'lr': .001, 'beta': .02, 'clip': .2,
            'retention_nll': .02, 'retention_splat': .02}}
    atomic_json(tmp_path/'plan.json', plan)
    return model, plan


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


@pytest.mark.parametrize('mode', ['grpo', 'rehearsal'])
def test_actual_sigterm_and_crash_recovery(tmp_path, monkeypatch, mode):
    _, plan = setup(tmp_path)
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
    before={p.name:p.read_bytes() for p in (tmp_path/'run').glob('*.pt')}
    assert outcomes.run(outcomes.parser().parse_args(common+['--resume'])) == report
    assert before == {p.name:p.read_bytes() for p in (tmp_path/'run').glob('*.pt')}
    assert digest(tmp_path/'run/best.pt') == report['best_sha256']


def test_training_output_has_one_writer(tmp_path):
    import fcntl
    out=tmp_path/'run'; out.mkdir()
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args=outcomes.parser().parse_args(['--plan','unused','--out',str(out),'--mode','grpo'])
        with pytest.raises(ValueError, match='already owned'):
            outcomes.run(args)

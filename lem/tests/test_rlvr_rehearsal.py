# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
from dataclasses import asdict
import signal

import numpy as np
import pytest
import torch

from lem.observer.data import VERSION
from lem.observer.model import Config
from lem.observer.population import atomic_json, digest
from lem.observer.rlvr.rehearsal import SCHEMA, Rehearsal
from lem.observer.rlvr.train import grow, objective, parser, run
from lem.observer.splat.model import SplatObserver
from lem.tests.test_rlvr import tiny_run, nested_equal


def replay_fixture(root):
    root.mkdir(); rng = np.random.default_rng(3)
    arrays = dict(x=rng.integers(0, 46, (12, 68)), targets=rng.integers(0, 28, (12, 28)),
        own_positions=np.tile(np.arange(13, 27, 2), (12, 1)),
        histograms=np.ones((12, 7, 7, 43), np.float32) / 43,
        candidates=np.tile(np.arange(7), (12, 7, 1)), split=np.array([0] * 8 + [1] * 4),
        hands=np.arange(48).reshape(12, 4) + 500)
    np.savez_compressed(root / 'rehearsal.npz', **arrays)
    atomic_json(root / 'manifest.json', {'settings': {'schema': SCHEMA, 'tokenizer': VERSION},
                                       'sha256': digest(root / 'rehearsal.npz')})


def test_exact_objective_gradient_and_ties():
    logits = torch.tensor([[0., 1.], [0., 0.]], requires_grad=True)
    targets = torch.tensor([[1., 0.], [1., 1.]])
    lp = logits.log_softmax(-1)
    loss, _ = objective(lp, lp.detach(), 'exact', targets, beta=0)
    loss.backward()
    p = lp[0, 0].exp().detach()
    torch.testing.assert_close(logits.grad[0], torch.tensor([-p * (1-p), p * (1-p)]) / 2)
    torch.testing.assert_close(logits.grad[1], torch.zeros(2))
    assert float(loss.detach()) == pytest.approx(-(float(p) + 1) / 2)


def test_growth_preserves_both_heads_and_new_block_learns():
    torch.manual_seed(7); model = SplatObserver(Config(width=8, layers=1, heads=2)).eval()
    bigger = grow(model, 1, 42)
    x = torch.randint(46, (3, 32)); pos = torch.tensor([[15], [19], [31]])
    original = model(x, pos); grown = bigger(x, pos)
    for a,b in zip(original, grown):
        assert torch.equal(a,b)
    assert bigger.config.layers == 2
    optimizer = torch.optim.AdamW(bigger.parameters(), lr=.001)
    loss = grown[0].square().mean() + grown[1].square().mean()
    loss.backward(); optimizer.step()
    assert torch.count_nonzero(bigger.backbone.blocks[-1].projection.weight) > 0
    assert not torch.equal(bigger(x, pos)[0], original[0])


@pytest.mark.parametrize('mode,extra,reward', [('grpo', 0, 'best'), ('exact', 1, 'best'),
                                             ('sft', 0, 'best'), ('exact', 0, 'success'),
                                             ('rehearsal', 0, 'best')])
def test_rehearsal_signal_resume_preserves_all_samplers_and_heads(tmp_path, monkeypatch, mode, extra, reward):
    common = tiny_run(tmp_path) + ['--mode', mode, '--extra-layers', str(extra), '--reward', reward]
    replay = tmp_path / 'replay'; replay_fixture(replay)
    common += ['--replay', str(replay), '--replay-weight', '1', '--replay-batch', '3']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    run(parser().parse_args(common + ['--out', str(whole)]))
    original_step = torch.optim.AdamW.step; calls = 0
    def interrupt(self, *a, **kw):
        nonlocal calls
        result = original_step(self, *a, **kw); calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupt)
    paused = run(parser().parse_args(common + ['--out', str(split)]))
    assert paused['state'] == 'paused' and paused['round'] == 1
    monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    run(parser().parse_args(common + ['--out', str(split), '--resume']))
    a, b = [torch.load(path / 'latest.pt', weights_only=True) for path in (whole, split)]
    nested_equal(a, b)
    initial = torch.load(tmp_path / 'initial.pt', weights_only=True)
    assert not torch.equal(a['model']['outcomes.weight'], initial['model']['outcomes.weight'])
    assert 'retention_pass' in a['best']
    with pytest.raises(ValueError, match='identity changed'):
        run(parser().parse_args(common + ['--out', str(split), '--resume', '--replay-weight', '.5']))
    if reward == 'success':
        with pytest.raises(ValueError, match='identity changed'):
            run(parser().parse_args(common + ['--out', str(split), '--resume', '--reward', 'best']))


def test_success_reward_preserves_gap_and_ignores_action_independent_baseline():
    logits = torch.zeros((3, 2), requires_grad=True)
    q = torch.tensor([[.8, .4], [.8, .79], [.4, .4]])
    lp = logits.log_softmax(-1)
    loss, _ = objective(lp, lp.detach(), 'exact', q, beta=0)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient[:, 0], -(q[:, 0] - q[:, 1]) / 12)
    torch.testing.assert_close(gradient[:, 1], -gradient[:, 0])
    torch.testing.assert_close(gradient[0], 40 * gradient[1], rtol=1e-5, atol=1e-7)
    shifted = logits.log_softmax(-1)
    other, _ = objective(shifted, shifted.detach(), 'exact', q - .4, beta=0)
    torch.testing.assert_close(torch.autograd.grad(other, logits)[0], gradient)


def test_rehearsal_control_has_no_policy_or_reference_gradient():
    logits = torch.tensor([[0., 1.]], requires_grad=True)
    loss, _ = objective(logits.log_softmax(-1), torch.tensor([[.8, .2]]).log(),
                        'rehearsal', torch.tensor([[1., 0.]]), beta=1.)
    loss.backward()
    assert float(loss.detach()) == 0 and torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize('mode', ['grpo', 'sft'])
def test_probability_objective_rejects_a_different_loss(mode):
    args = parser().parse_args(['--data', 'unused', '--initial', 'unused', '--out', 'unused',
                               '--mode', mode, '--reward', 'success'])
    with pytest.raises(ValueError, match='success reward requires exact'):
        run(args)


def test_rehearsal_deal_split_and_corruption_fail_closed(tmp_path):
    replay_fixture(tmp_path / 'replay')
    data = Rehearsal(tmp_path / 'replay', 'cpu')
    assert len(data.splits['train']) == 8
    (tmp_path / 'replay/rehearsal.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        Rehearsal(tmp_path / 'replay', 'cpu')

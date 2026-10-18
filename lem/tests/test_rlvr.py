# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Independent game-tree verification and the learning/recovery boundary."""
from dataclasses import asdict, replace
from fractions import Fraction
from functools import lru_cache
import gzip
import itertools
import json
from pathlib import Path
import signal

import pytest
import torch

from forge.zeb.game import new_game, legal_actions, current_player, apply_action
from forge.zeb.types import BidState
from lem.observer.data import VERSION
from lem.observer.model import Config
from lem.observer.population import atomic_json, digest
from lem.observer.splat.model import SplatObserver, TASK
from lem.observer.rlvr.data import candidate_questions, tokens
from lem.observer.rlvr.verifier import solve, verify, MODEL_ID, BudgetExceeded
from lem.observer.rlvr.train import advantages, objective, parser, run


@lru_cache(None)
def examples():
    result = []
    for seed in range(400):
        hands, questions = candidate_questions(seed)
        for q in questions:
            result.append((hands, q))
        if len(result) >= 12:
            return result[:12]
    raise AssertionError('no exercise examples')


def independent_values(q):
    """Enumerate allocations and replay with forge's distinct referee, no Silas."""
    public = [{t for s, t in q['plays'] if s == i} for i in range(4)]
    known = set(q['own']) | set().union(*public)
    hidden = set(range(28)) - known
    others = [s for s in range(4) if s != q['viewer']]
    roots = []
    for a in itertools.combinations(sorted(hidden), 7 - len(public[others[0]])):
        for b in itertools.combinations(sorted(hidden - set(a)), 7 - len(public[others[1]])):
            allocation = (set(a), set(b), hidden - set(a) - set(b))
            hands = [set(h) for h in public]; hands[q['viewer']] = set(q['own'])
            for seat, tiles in zip(others, allocation):
                hands[seat] |= tiles
            state = replace(new_game(0), hands=tuple(tuple(sorted(h)) for h in hands),
                            bidder=q['bidder'], trick_leader=q['bidder'], decl_id=q['decl'],
                            bid_state=BidState(tuple(30 if i == q['bidder'] else 0 for i in range(4)), q['bidder'], 30))
            try:
                for actor, tile in q['plays']:
                    assert current_player(state) == actor
                    state = apply_action(state, state.hands[actor].index(tile))
            except ValueError:
                continue
            roots.append(state)
    @lru_cache(None)
    def recurse(state, style):
        bidder_team = q['bidder'] % 2
        made = state.team_points[bidder_team] >= 30
        if made or state.team_points[1 - bidder_team] >= 13 or len(state.played) == 28:
            return Fraction(int(made == (q['viewer'] % 2 == bidder_team)))
        actor = current_player(state)
        actions = sorted(legal_actions(state), key=lambda a: state.hands[actor][a])
        if actor == q['viewer']:
            assert len(actions) == 1
        elif style == 'lowest':
            actions = actions[:1]
        elif style == 'highest':
            actions = actions[-1:]
        return sum((recurse(apply_action(state, a), style) for a in actions), Fraction(0)) / len(actions)
    legal = sorted(roots[0].hands[q['viewer']][a] for a in legal_actions(roots[0]))
    by_field = {s: {str(a): str(sum((recurse(apply_action(r, r.hands[q['viewer']].index(a)), s)
                                   for r in roots), Fraction(0)) / len(roots)) for a in legal}
                for s in ('lowest', 'highest', 'uniform')}
    return len(roots), by_field


@pytest.mark.parametrize('index', range(12))
def test_exact_values_match_independent_referee(index):
    _, q = examples()[index]
    actual = solve(q)
    count, values = independent_values(q)
    assert actual['worlds'] == count
    assert actual['by_field'] == values
    for a in actual['legal']:
        expected = sum((Fraction(v[str(a)]) for v in values.values()), Fraction(0)) / 3
        assert Fraction(actual['values'][str(a)]) == expected
        assert verify(a, actual) == int(a in actual['best'])
    assert verify(99, actual) == 0


def test_no_hidden_deal_input_and_seat_rotation():
    _, q = examples()[2]
    cert = solve(q)
    with_secret = {**q, 'other_hands': [[999]], 'teacher': 'arbitrary'}
    assert solve(with_secret) == cert
    rotated = {**q, 'viewer': (q['viewer'] + 1) % 4, 'bidder': (q['bidder'] + 1) % 4,
               'plays': [[(s + 1) % 4, t] for s, t in q['plays']]}
    assert solve(rotated)['values'] == cert['values']
    assert tokens(rotated) == tokens(q)
    with pytest.raises(BudgetExceeded):
        solve(q, node_cap=1)
    with pytest.raises(ValueError):
        solve({**q, 'bid': 31})
    with pytest.raises(ValueError):
        verify(cert['best'][0], {**cert, 'model_id': 'wrong'})


def test_group_relative_and_clipping_math():
    rewards = torch.tensor([[0., 1., 1., 0.], [1., 1., 1., 1.]])
    assert torch.equal(advantages(rewards), torch.tensor([[-1., 1., 1., -1.], [0., 0., 0., 0.]]))
    logits = torch.tensor([[0., 0.]], requires_grad=True)
    old = logits.detach().log_softmax(-1)
    loss, info = objective(logits.log_softmax(-1), old, 'grpo', None,
                           torch.tensor([[0, 1]]), old, torch.tensor([[-1., 1.]]), beta=0.)
    loss.backward()
    assert logits.grad[0, 1] < 0 < logits.grad[0, 0]
    logp = torch.tensor([[.1, .9]]).log()
    loss, _ = objective(logp, old, 'grpo', None, torch.tensor([[0, 1]]), old,
                         torch.tensor([[-1., 1.]]), clip=.2, beta=0.)
    assert float(loss) == pytest.approx(-.2)


def tiny_run(tmp_path):
    torch.manual_seed(12)
    config = Config(width=8, layers=1, heads=2)
    initial = tmp_path / 'initial.pt'
    torch.save(dict(task=TASK, tokenizer=VERSION, config=asdict(config), model=SplatObserver(config).state_dict(),
                    step=0, data_sha256='test'), initial)
    data = tmp_path / 'data'; data.mkdir()
    rows = []
    seen = set()
    for hands, question in examples():
        if tuple(hands) in seen:
            continue
        seen.add(tuple(hands))
        q = {**question, 'certificate': solve(question)}
        rows.append(dict(physical_hands=hands, question=q, tokens=tokens(q),
                         split='validation' if len(rows) < 2 else 'train'))
    with gzip.open(data / 'questions.jsonl.gz', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    atomic_json(data / 'manifest.json', {'model_id': MODEL_ID, 'questions_sha256': digest(data / 'questions.jsonl.gz')})
    return ['--data', str(data), '--initial', str(initial), '--device', 'cpu', '--rounds', '4',
            '--batch', '4', '--group', '4', '--eval-every', '2']


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('mode', ['sft', 'grpo'])
def test_signal_pause_exact_resume_and_frozen_outcome_head(tmp_path, monkeypatch, mode):
    common = tiny_run(tmp_path) + ['--mode', mode]
    whole = tmp_path / 'whole'; split = tmp_path / 'split'
    run(parser().parse_args(common + ['--out', str(whole)]))
    original = torch.optim.AdamW.step
    count = 0
    def interrupt(self, *a, **kw):
        nonlocal count
        result = original(self, *a, **kw); count += 1
        if count == 1:
            signal.raise_signal(signal.SIGINT)
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupt)
    status = run(parser().parse_args(common + ['--out', str(split)]))
    assert status['state'] == 'paused' and status['round'] == 1
    assert not (split / 'report.json').exists()
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    run(parser().parse_args(common + ['--out', str(split), '--resume']))
    a = torch.load(whole / 'latest.pt', weights_only=True)
    b = torch.load(split / 'latest.pt', weights_only=True)
    nested_equal(a, b)
    before = {p.name: digest(p) for p in split.iterdir() if p.is_file()}
    run(parser().parse_args(common + ['--out', str(split), '--resume']))
    assert before == {p.name: digest(p) for p in split.iterdir() if p.is_file()}
    initial = torch.load(tmp_path / 'initial.pt', weights_only=True)
    for k in ('outcomes.weight', 'outcomes.bias'):
        assert torch.equal(a['model'][k], initial['model'][k])
    assert any(not torch.equal(v, initial['model'][k]) for k, v in a['model'].items() if k.startswith('backbone'))
    with pytest.raises(ValueError, match='identity changed'):
        run(parser().parse_args(common + ['--out', str(split), '--resume', '--lr', '.01']))

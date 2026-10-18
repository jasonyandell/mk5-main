# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import signal

import pytest
import torch

from lem.observer.benchmark.build import identity
from lem.observer.data import VERSION
from lem.observer.model import Config
from lem.observer.outcomes import source_identity
from lem.observer.population import atomic_json, digest, make_specs
from lem.observer.rlvr import fixed_data, fixed_field
from lem.observer.rlvr.data import load
from lem.observer.rlvr.verifier import root_frame
from lem.observer.splat.model import SplatObserver, TASK
from lem.tests.test_benchmark_extension import Lowest
from lem.tests.test_rlvr import examples, independent_values
from lem.tests.test_rlvr_rehearsal import replay_fixture


def unsettled():
    for _, q in examples():
        frame, _, _ = root_frame(q)
        if frame.banked[q['bidder'] % 2] < 30 and frame.banked[1-q['bidder'] % 2] < 13:
            yield q


def test_exact_fixed_policy_values_match_independent_referee():
    count = 0
    for q in unsettled():
        cert, witness = fixed_field.solve(q, {'g0': Lowest(), 'zeb': Lowest()}, 'lowest-test')
        worlds, values = independent_values(q)
        assert cert['worlds'] == worlds
        assert cert['values'] == values['lowest']
        assert len(witness['continuations']) == 2 * worlds
        for row in witness['continuations']:
            assert row['episode']['plays'][len(q['plays'])] == [q['viewer'], row['action']]
            assert len(row['episode']['plays']) == 28
        assert fixed_field.solve({**q, 'secret': [[999]], 'teacher': 'ignored'},
                                 {'g0': Lowest(), 'zeb': Lowest()}, 'lowest-test') == (cert, witness)
        count += 1
    assert count >= 2


def test_fixed_policy_illegal_output_fails_closed():
    class Illegal:
        def choose(self, states, bids):
            return [99 for _ in states]
    with pytest.raises(ValueError, match='illegal'):
        fixed_field.solve(next(unsettled()), {'g0': Illegal(), 'zeb': Illegal()}, 'bad-policy')


def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1); torch.manual_seed(7)
    initial = tmp_path / 'initial.pt'; config = Config(width=8, layers=1, heads=2)
    torch.save({'task': TASK, 'tokenizer': VERSION, 'config': asdict(config),
                'model': SplatObserver(config).state_dict(), 'step': 0, 'data_sha256': 'test'}, initial)
    replay_fixture(tmp_path / 'replay')
    field_path = tmp_path / 'field.npz'; field_path.write_bytes(b'controlled test policies')
    field = {'zeb_field': str(field_path), 'zeb_field_sha256': digest(field_path),
             'zeb_spec': 'k0,F=' + str(field_path),
             'pool': {'g0': {'path': str(initial), 'sha256': digest(initial)}}}
    monkeypatch.setattr(fixed_data, 'policies', lambda *args: {'g0': Lowest(), 'zeb': Lowest()})
    specs = make_specs(96, 'fixed-curriculum-test', ['g0'], set())
    for s in specs:
        s['seats'] = ['zeb'] * 4; s['seats'][s['learner']] = 'learner'
        s['seats'][(s['learner'] + 2) % 4] = 'g0'
    model = fixed_field.specification(field)
    plan = {'schema': fixed_data.SCHEMA, 'source': source_identity(), 'namespace': 'fixed-curriculum-test',
            'initial': str(initial), 'initial_sha256': digest(initial), 'field': field,
            'replay': str(tmp_path / 'replay'), 'replay_sha256': digest(tmp_path / 'replay/rehearsal.npz'),
            'model': model, 'model_id': identity(model), 'batch': 4,
            'counts': dict(train=1, validation=1, test=1),
            'roots': dict(train=specs[:32], validation=specs[32:64], test=specs[64:])}
    path = tmp_path / 'plan.json'; atomic_json(path, plan)
    return path, plan


def files(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*')
            if p.is_file() and p.suffix in ('.json', '.gz')}


def test_builder_signal_resume_recheck_and_completed_bytes(tmp_path, monkeypatch):
    plan_path, _ = fixture(tmp_path, monkeypatch)
    def args(name, *extra):
        return fixed_data.parser().parse_args(['--plan', str(plan_path), '--out', str(tmp_path / name), *extra])
    fixed_data.build(args('whole'))
    produce = fixed_data.produce; calls = 0
    def interrupted(*a, **kw):
        nonlocal calls
        result = produce(*a, **kw); calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(fixed_data, 'produce', interrupted)
    assert fixed_data.build(args('split')) is None
    monkeypatch.setattr(fixed_data, 'produce', produce)
    fixed_data.build(args('split'))
    assert files(tmp_path / 'whole') == files(tmp_path / 'split')
    before = files(tmp_path / 'split')
    fixed_data.build(args('split', '--recheck'))
    assert before == files(tmp_path / 'split')
    monkeypatch.setattr(fixed_data, 'produce', lambda *a, **kw: pytest.fail('completed units reran'))
    fixed_data.build(args('split'))
    assert before == files(tmp_path / 'split')
    _, rows = load(tmp_path / 'split')
    assert len(rows) == 3 and len({tuple(r['physical_hands']) for r in rows}) == 3


def test_loader_rejects_rewritten_certificate_even_with_new_archive_hash(tmp_path, monkeypatch):
    plan_path, plan = fixture(tmp_path, monkeypatch)
    root = tmp_path / 'data'
    fixed_data.build(fixed_data.parser().parse_args(['--plan', str(plan_path), '--out', str(root)]))
    manifest, rows = load(root)
    cert = rows[0]['question']['certificate']
    changed = '0' if set(cert['values'].values()) == {'1'} else '1'
    cert['values'] = {str(a): changed for a in cert['legal']}
    cert['best'] = cert['legal']
    with gzip.open(root / 'questions.jsonl.gz', 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    manifest['questions_sha256'] = digest(root / 'questions.jsonl.gz')
    atomic_json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='verified unit'):
        load(root)
    plan['field']['zeb_spec'] = 'k0,F=' + str(tmp_path / 'wrong.npz')
    with pytest.raises(ValueError, match='pinned field'):
        fixed_field.specification(plan['field'])

# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest
import numpy as np

from forge.zeb.game import current_player, legal_actions
from lem.observer.benchmark.build import canonical
from lem.observer.benchmark.extension import Extension, sample_seed
from lem.observer.benchmark.continue_rows import continuation, remaining_worlds, replay_root
from lem.observer.rlvr.verifier import solve
from lem.observer.play import prefix
from lem.tests.test_rlvr import examples


def row(index=0):
    _, question = examples()[index]
    return {'id': f'row-{index}', 'question': question, 'legal': solve(question)['legal']}


def produce(row, sample):
    return {'seed': sample_seed({'root': 42}, row, sample, 'deal'), 'value': [row['id'], sample]}


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.json')}


def test_resume_order_batches_and_snapshot_are_byte_identical(tmp_path):
    a, b = [Extension(tmp_path / name, {'field': 'fixed'}) for name in ('a', 'b')]
    rows = [row(0), row(1)]
    for r in rows:
        for i in range(4):
            a.ensure(r, i, produce)
    a.snapshot()
    for r in reversed(rows):
        b.ensure(r, 3, produce)
    # Reopen and append in another order, including a duplicate request.
    b = Extension(tmp_path / 'b', {'field': 'fixed'})
    for i in range(4):
        for r in rows:
            b.ensure(r, i, produce)
    b.snapshot()
    assert files(a.root) == files(b.root)
    before = files(b.root)
    b.ensure(rows[0], 0, lambda *a: pytest.fail('completed unit reran'))
    b.ensure(rows[0], 0, produce, recheck=True)
    assert files(b.root) == before
    b.ensure(rows[0], 4, produce)
    b.snapshot()
    for name, data in before.items():
        assert (b.root / name).read_bytes() == data


def test_failure_corruption_recipe_drift_and_recompute_fail_closed(tmp_path):
    store = Extension(tmp_path, {'weights': 'a'})
    def fail(*args):
        raise RuntimeError('interrupted before commit')
    with pytest.raises(RuntimeError):
        store.ensure(row(), 0, fail)
    assert not list(store.records.glob('*.json'))
    store.ensure(row(), 0, produce)
    with pytest.raises(ValueError, match='recipe changed'):
        Extension(tmp_path, {'weights': 'b'})
    with pytest.raises(ValueError, match='recomputation changed'):
        store.ensure(row(), 0, lambda *a: {'wrong': True}, recheck=True)
    path, = store.records.glob('*.json')
    record = json.loads(path.read_text()); record['result']['seed'] += 1
    path.write_text(canonical(record) + '\n')
    with pytest.raises(ValueError, match='checksum'):
        store.ensure(row(), 0, produce)
    with pytest.raises(ValueError, match='checksum'):
        store.snapshot()


def test_concurrent_duplicate_is_busy_then_completed(tmp_path):
    store = Extension(tmp_path, {'field': 'fixed'})
    entered, finish = Event(), Event()
    def slow(r, s):
        entered.set()
        assert finish.wait(5)
        return produce(r, s)
    r = row()
    with ThreadPoolExecutor(1) as pool:
        work = pool.submit(store.ensure, r, 0, slow)
        assert entered.wait(5)
        try:
            assert store.ensure(r, 0, produce) is None
        finally:
            finish.set()
        expected = work.result()
    assert store.ensure(r, 0, produce) == expected
    assert len(list(store.records.glob('*.json'))) == 1


class Lowest:
    def __init__(self):
        self.inputs = []
    def choose(self, states, bids):
        self.inputs += [prefix(s, 30) for s in states]
        return [min(legal_actions(s), key=lambda i: s.hands[current_player(s)][i]) for s in states]


def test_paired_roots_clean_inputs_and_independent_terminal_replay():
    r = row(2); law = remaining_worlds(json.dumps(r['question'], sort_keys=True))
    for world in law:
        state = replay_root(r['question'], world)
        assert current_player(state) == r['question']['viewer']
    coupling = {'field': 'lowest', 'seed': 42}
    before = canonical(r)
    for i in range(6):
        learner, field = Lowest(), Lowest()
        a = continuation(r, i, coupling, learner, (field,) * 4)
        b = continuation(r, i, coupling, Lowest(), (Lowest(),) * 4)
        assert a == b
        assert len(a['episode']['plays']) == 28 and sum(a['episode']['points']) == 42
        assert a['episode']['reward'] == (1 if a['won'] else -1)
        np.testing.assert_array_equal(learner.inputs[0], prefix(replay_root(r['question'], law[0]), 30))
        assert a['seed'] == sample_seed(coupling, r, i, 'hidden-deal')
        assert a['seed'] != sample_seed(coupling, r, i, 'action')
    assert canonical(r) == before
    with pytest.raises(ValueError, match='legal outputs'):
        continuation({**r, 'legal': []}, 0, coupling, Lowest(), (Lowest(),) * 4)

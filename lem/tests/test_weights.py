# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
import copy

import pytest
import torch

from lem.weights import state_identity
from lem.observer.benchmark.extension import Extension


def test_state_identity_ignores_mapping_order_and_memory_layout():
    x = torch.arange(6., requires_grad=True).reshape(2, 3)
    state = {'weight': x, 'counter': torch.tensor(7)}
    other = {'counter': state['counter'].clone(), 'weight': x.T.contiguous().T}
    before = x.detach().clone()
    assert state_identity(state, {'head': 'tile'}) == state_identity(other, {'head': 'tile'})
    assert torch.equal(x, before) and x.requires_grad and x.grad_fn is not None
    metadata = {'shape': [2, 3]}
    identity = state_identity(state, metadata)
    metadata['shape'][0] = 9
    assert identity['metadata']['shape'] == [2, 3]


@pytest.mark.parametrize('change', ['value', 'shape', 'dtype', 'name', 'metadata'])
def test_every_inference_change_gets_a_new_identity(change):
    state = {'weight': torch.arange(6.).reshape(2, 3)}; metadata = {'head': 'tile'}
    original = state_identity(state, metadata)['sha256']; other = copy.deepcopy(state)
    if change == 'value': other['weight'][0, 0] = 1.
    if change == 'shape': other['weight'] = other['weight'].reshape(3, 2)
    if change == 'dtype': other['weight'] = other['weight'].double()
    if change == 'name': other['other'] = other.pop('weight')
    if change == 'metadata': metadata['head'] = 'splat'
    assert state_identity(other, metadata)['sha256'] != original


def test_invalid_states_and_ambiguous_metadata_are_rejected():
    for value in ({}, {'weight': 'not a tensor'}, {'weight': torch.eye(2).to_sparse()},
                  {'weight': torch.ones(1), 0: torch.ones(1)}):
        with pytest.raises(ValueError): state_identity(value, {})
    with pytest.raises(ValueError): state_identity({'weight': torch.ones(1)}, {'bad': float('nan')})


def test_checkpoint_history_can_share_evidence_but_changed_weights_cannot(tmp_path):
    calls = []
    for step, weight in ((0, 2.), (8, 2.), (9, 3.)):
        path = tmp_path / f'{step}.pt'
        torch.save({'model': {'weight': torch.tensor([weight])}, 'step': step}, path)
        state = torch.load(path, weights_only=True)['model']
        identity = state_identity(state, {'model': 'scalar', 'head': 'value'})
        store = Extension(tmp_path / identity['sha256'],
                          {'weights': identity, 'runtime': 'same-cpu-test'})
        def produce(row, sample):
            calls.append(step)
            return {'output': float(state['weight'][0]) * row['question']}
        record = store.ensure({'id': 'row', 'question': 5}, 0, produce)
        assert record['result']['output'] == weight * 5
    assert calls == [0, 9]

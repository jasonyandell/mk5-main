# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pytest
import torch

from lem.observer.benchmark.build import SCHEMA, digest, write_json, write_rows
from lem.observer.benchmark.evaluate import capture, clustered, observations, parser, run
from lem.observer.benchmark.h2h import asset, check_runtime
from lem.observer.data import VERSION
from lem.observer.model import Config, Observer
from lem.observer.rlvr.data import candidate_questions, tokens
from lem.observer.rlvr.verifier import solve
from lem.observer.splat.model import SplatObserver, TASK


def rows():
    result = []
    for seed in range(400):
        _, questions = candidate_questions(seed)
        for q in questions:
            c = solve(q)
            result.append({'id': f'row-{len(result)}', 'pair_id': f'pair-{seed}', 'panel': 'test',
                           'behavior_id': 'recorded', 'question': q, 'tokens': tokens(q),
                           'legal': c['legal'], 'reference_action': c['legal'][0], 'exact_certificate': c,
                           'public': {'settled': False}, 'outcome': {'team_points': [21, 21], 'bidder_made': False}})
            if len(result) == 4:
                return result
    raise AssertionError('no examples')


@pytest.mark.parametrize('joint', [False, True])
def test_capture_matches_forward_and_labels_never_enter_net(joint):
    torch.manual_seed(1)
    config = Config(width=8, layers=2, heads=2)
    model = (SplatObserver if joint else Observer)(config).eval()
    examples = rows(); arrays = capture(model, joint, examples)
    assert arrays['layer_residual'].shape == (4, 2, 8)
    assert arrays['mlp_post'].shape == (4, 2, 32)
    assert arrays['attention_to_prefix'].shape == (4, 2, 2, 68)
    for i, row in enumerate(examples):
        x = torch.tensor([row['tokens']])
        with torch.no_grad():
            direct = model.backbone(x) if joint else model(x)
        np.testing.assert_allclose(arrays['raw_logits'][i], direct[0, -1].numpy(), rtol=1e-5, atol=1e-7)
        attention = arrays['attention_to_prefix'][i]
        assert np.count_nonzero(attention[..., len(row['tokens']):]) == 0
        np.testing.assert_allclose(attention.sum(-1), 1., atol=1e-6)
    changed = deepcopy(examples)
    for row in changed:
        row['outcome'] = {'secret': 'future outcome changed'}
        row['full_hands'] = [[999]]
        row['reference_action'] = row['legal'][-1]
        row.pop('exact_certificate')
    repeated = capture(model, joint, changed)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])


def test_observational_outcome_is_scored_for_recorded_action_only():
    examples = rows(); n = len(examples)
    arrays = {'raw_logits': np.zeros((n, 28)), 'raw_probs': np.full((n, 28), 1/28),
              'legal_probs': np.zeros((n, 28)), 'splat_probs': np.full((n, 28, 43), 1/43)}
    for i, row in enumerate(examples):
        recorded, alternate = row['legal']
        arrays['raw_logits'][i, alternate] = 100.
        arrays['legal_probs'][i, row['legal']] = .5
        arrays['splat_probs'][i, recorded] = 0.
        arrays['splat_probs'][i, recorded, 21] = 1.
    scores = observations(examples, arrays, 'tile')
    assert np.all(arrays['chosen_tile'] != [r['reference_action'] for r in examples])
    np.testing.assert_array_equal(scores['recorded_action_outcome_nll'], 0.)
    for row, choice, regret in zip(examples, arrays['chosen_tile'], scores['exact_endgame_regret']):
        q = {int(k): Fraction(v) for k, v in row['exact_certificate']['values'].items()}
        assert regret == float(max(q.values()) - q[int(choice)])


def test_cluster_unit_and_missing_exact_coverage():
    examples = rows()
    for r in examples:
        r['pair_id'] = 'one-duplicate-pair'
    result = clustered(np.array([1., 2., np.nan, 3.]), examples, [True] * 4)
    assert result == {'estimate': 2., 'ci95': None, 'rows': 3, 'pairs': 1}
    assert clustered(np.full(4, np.nan), examples, [True] * 4) is None


def test_same_model_paired_delta_zero_and_frozen_output_identity(tmp_path):
    torch.manual_seed(4); config = Config(width=8, layers=1, heads=2)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': asdict(config), 'tokenizer': VERSION, 'model': Observer(config).state_dict()}, checkpoint)
    panel = tmp_path / 'panel'; panel.mkdir()
    examples = rows(); write_rows(panel / 'rows.jsonl.gz', examples)
    write_json(panel / 'manifest.json', {'schema': SCHEMA, 'tokenizer': VERSION, 'panel_id': 'test',
                                        'rows': len(examples), 'rows_sha256': digest(panel / 'rows.jsonl.gz')})
    common = ['--benchmark', str(panel), '--checkpoint', str(checkpoint)]
    first = tmp_path / 'first'; second = tmp_path / 'second'
    a = run(parser().parse_args(common + ['--out', str(first)]))
    b = run(parser().parse_args(common + ['--out', str(second), '--reference', str(first)]))
    for measure in b['paired_delta']['all'].values():
        if measure:
            assert measure['estimate'] == 0
            assert measure['ci95'] is None or measure['ci95'] == [0., 0.]
    before = digest(first / 'report.json')
    assert run(parser().parse_args(common + ['--out', str(first)])) == a
    assert digest(first / 'report.json') == before
    with pytest.raises(ValueError, match='different model'):
        run(parser().parse_args(common + ['--out', str(first), '--batch', '64']))
    with pytest.raises(ValueError, match='no splat'):
        run(parser().parse_args(common + ['--out', str(tmp_path / 'wrong'), '--head', 'splat']))
    payload = json.loads((first / 'report.json').read_text()); payload['identity']['rows_sha256'] = 'wrong'
    write_json(first / 'report.json', payload)
    with pytest.raises(ValueError, match='panel or representation'):
        run(parser().parse_args(common + ['--out', str(tmp_path / 'third'), '--reference', str(first)]))


def test_frozen_h2h_checks_runtime_versions_source_and_assets(tmp_path):
    source = 'lem/observer/model.py'
    root = Path(__file__).resolve().parents[2]
    pins = {'schema': 'gus-frozen-runtime-pins-v1', 'source_files': {source: digest(root / source)},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'torch': str(torch.__version__), 'scenario_core': version('scenario_core')}}
    pinfile = tmp_path / 'runtime-pins.json'; write_json(pinfile, pins)
    assert check_runtime(tmp_path) == digest(pinfile)
    pins['source_files'][source] = 'wrong'; write_json(pinfile, pins)
    with pytest.raises(ValueError, match='runtime changed'):
        check_runtime(tmp_path)
    pins['source_files'][source] = digest(root / source)
    pins['versions']['torch'] = 'different'; write_json(pinfile, pins)
    with pytest.raises(ValueError, match='versions differ'):
        check_runtime(tmp_path)
    model = tmp_path / 'model.pt'; model.write_bytes(b'pinned model')
    manifest = {'assets': {'g0': {'path': model.name, 'sha256': digest(model)}}}
    assert asset(tmp_path, manifest, 'g0') == model
    model.write_bytes(b'changed model')
    with pytest.raises(ValueError, match='asset hash mismatch'):
        asset(tmp_path, manifest, 'g0')

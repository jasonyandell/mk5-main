# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""One frozen test readout, paired uncertainty, probes and pretraining retention."""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..population import atomic_json, digest
from ..splat.data import Corpus
from ..splat.model import SplatObserver
from ..splat.play import SplatPlay
from ..splat.train import DeviceCorpus, evaluate as retention_evaluate
from .data import load
from .train import Exercises, predictions, metrics


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(42)
    estimates = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(2000)]
    return {'estimate': float(values.mean()), 'ci95': np.quantile(estimates, [.025, .975]).tolist()}


@torch.inference_mode()
def features(model, data):
    captured = []
    def hook(module, arguments):
        captured.append(arguments[0].detach())
    handle = model.backbone.norm.register_forward_pre_hook(hook)
    result = []
    try:
        for lo in range(0, len(data.rows), 128):
            index = torch.arange(lo, min(lo + 128, len(data.rows)), device=data.x.device)
            data.logp(model, index)
            x = captured.pop()
            result.append(x[torch.arange(len(index), device=x.device), data.pos[index]].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(result)


def ridge_probe(train_x, test_x, train_y, test_y):
    mean, std = train_x.mean(0), train_x.std(0).clip(1e-8)
    x = ((train_x - mean) / std).astype(np.float64)
    test = ((test_x - mean) / std).astype(np.float64)
    y_mean = train_y.mean(0)
    weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ (train_y - y_mean))
    prediction = test @ weights + y_mean
    return {'mse': float(((prediction - test_y) ** 2).mean()),
            'constant_mse': float(((y_mean - test_y) ** 2).mean())}, prediction


@torch.inference_mode()
def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest, rows = load(args.data)
    train = Exercises([r for r in rows if r['split'] == 'train'], args.device)
    test = Exercises([r for r in rows if r['split'] == 'test'], args.device)
    train_y = np.array(train.q, dtype=np.float64); test_y = np.array(test.q, dtype=np.float64)
    paths = {'pretrained': args.initial, 'sft': args.sft, 'grpo': args.grpo}
    corpus = Corpus([args.retention_cache]); retained = DeviceCorpus(corpus, args.device)
    indices = retained.splits['test'][:1024]
    report = {'scope': 'fresh exact finite-model test; old pretraining retention is a diagnostic',
              'data_sha256': manifest['questions_sha256'], 'test_deals': len(test.rows),
              'uncertainty': '2000 paired physical-deal percentile bootstrap draws, seed42',
              'retention': {'corpus_sha256': corpus.sha256, 'views': len(indices), 'indices': indices.tolist()},
              'models': {}, 'probes': {}}
    arrays = {'test_values': test_y, 'test_deal_index': np.array([r['deal_index'] for r in test.rows])}
    regrets = {}
    for name, path in paths.items():
        player = SplatPlay(path, args.device, 'tile'); model = player.model
        lp = predictions(model, test); choices = lp.argmax(-1).numpy()
        regrets[name] = np.array([float(max(q) - q[a]) for q, a in zip(test.q, choices)])
        hits = np.array([a in [i for i, v in enumerate(q) if v == max(q)] for q, a in zip(test.q, choices)], dtype=float)
        fields = {}
        for field in ('lowest', 'highest', 'uniform'):
            values = [[Fraction(r['question']['certificate']['by_field'][field][str(a)])
                       for a in r['question']['certificate']['legal']] for r in test.rows]
            fields[field] = {'return': interval([float(q[a]) for q, a in zip(values, choices)]),
                             'regret_to_field_best': interval([float(max(q) - q[a]) for q, a in zip(values, choices)])}
        report['models'][name] = {'identity': player.identity, 'metrics': metrics(lp, test),
                                  'regret_interval': interval(regrets[name]), 'accuracy_interval': interval(hits),
                                  'fields': fields, 'retention': retention_evaluate(model, retained, indices)}
        arrays[name + '_logp'] = lp.numpy(); arrays[name + '_regret'] = regrets[name]; arrays[name + '_hit'] = hits
        train_x, test_x = features(model, train), features(model, test)
        result, prediction = ridge_probe(train_x, test_x, train_y, test_y)
        report['probes'][name] = result
        arrays[name + '_probe_prediction'] = prediction
        arrays[name + '_train_features'] = train_x; arrays[name + '_test_features'] = test_x
        if name == 'pretrained':
            torch.manual_seed(42)
            twin = SplatObserver(model.config).to(args.device).eval()
            random_train, random_test = features(twin, train), features(twin, test)
            result, prediction = ridge_probe(random_train, random_test, train_y, test_y)
            report['probes']['untrained_twin'] = result
            arrays['untrained_probe_prediction'] = prediction
            arrays['untrained_train_features'] = random_train; arrays['untrained_test_features'] = random_test
            torch.save({'model': twin.cpu().state_dict(), 'seed': 42, 'purpose': 'matched random representation control'}, out / 'untrained-probe-twin.pt')
            del twin
        del model, player
    report['paired'] = {name + '_regret_reduction_vs_pretrained': interval(regrets['pretrained'] - regrets[name]) for name in ('sft', 'grpo')}
    report['paired']['grpo_regret_reduction_vs_sft'] = interval(regrets['sft'] - regrets['grpo'])
    report['paired'].update({name + '_accuracy_gain_vs_pretrained': interval(arrays[name + '_hit'] - arrays['pretrained_hit']) for name in ('sft', 'grpo')})
    report['uniform_legal'] = metrics(torch.full((len(test.rows), 2), -.6931471805599453), test)
    # Uniform sampling has no distinguished argmax. Replace those quantities
    # with exact expectations; never use the arbitrary first-tile tie break.
    report['uniform_legal'].update(accuracy=float(test.best.mean(-1).mean()),
                                   regret=float(np.mean([float(max(q) - sum(q) / 2) for q in test.q])),
                                   **{'return': float(test_y.mean())})
    report['uniform_legal'].pop('regret_exact')
    report['material_gain'] = {name: r['estimate'] >= .01 and r['ci95'][0] > 0
                              for name in ('sft', 'grpo') for r in [report['paired'][name + '_regret_reduction_vs_pretrained']]}
    report['probes']['method'] = 'final actor residual before final LayerNorm; ridge lambda1, train-only standardization and centering; two sorted-legal action Q targets; diagnostic, no causal neuron claim'
    np.savez_compressed(out / 'per-deal-and-features.npz', **arrays)
    report['arrays_sha256'] = digest(out / 'per-deal-and-features.npz')
    atomic_json(out / 'report.json', report)
    print({k: report[k] for k in ('paired', 'material_gain', 'probes')}, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'initial', 'sft', 'grpo', 'out', 'retention-cache'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='mps')
    run(p.parse_args())

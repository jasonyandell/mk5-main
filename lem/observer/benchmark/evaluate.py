# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Replay-free regression scoring and native activation capture on frozen rows.

These scores describe a policy on fixed reference histories. They do not reuse
a recorded win as the outcome of a different action or claim off-policy wins.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from ..model import Config, Observer
from ..data import VERSION
from ..population import atomic_json, digest
from ..splat.model import TASK, SplatObserver
from .build import load

CAPTURE_VERSION = 'gus-last-actor-native-v1'


def load_model(checkpoint, head, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['tokenizer'] != VERSION:
        raise ValueError('checkpoint tokenizer mismatch')
    joint = saved.get('task') == TASK
    if not joint and saved.get('task') is not None:
        raise ValueError('unsupported checkpoint task')
    if head == 'splat' and not joint:
        raise ValueError('the original observer has no splat head')
    model = (SplatObserver if joint else Observer)(Config(**saved['config'])).to(device).eval()
    model.load_state_dict(saved['model'])
    return model, joint, saved


@torch.inference_mode()
def capture(model, joint, rows, device='cpu', batch=128):
    """Only tokens enter the net. Legal sets are used after the forward pass."""
    backbone = model.backbone if joint else model
    config = backbone.config
    gathered = {}
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        positions = torch.tensor([len(r['tokens']) - 1 for r in part], device=device)
        x = torch.zeros((len(part), config.context), dtype=torch.long, device=device)
        for i, row in enumerate(part):
            x[i, :len(row['tokens'])] = torch.tensor(row['tokens'], device=device)
        ix = torch.arange(len(part), device=device)
        blocks = []; final = {}
        def block_hook(module, arguments, output):
            residual, trace = output
            attention = trace['attention'][ix, :, positions, :].detach()
            blocks.append((residual[ix, positions].detach(), trace['pre'][ix, positions].detach(),
                           trace['post'][ix, positions].detach(), attention))
        def norm_hook(module, arguments, output):
            final['actor_residual'] = arguments[0][ix, positions].detach()
            final['actor_normalized'] = output[ix, positions].detach()
        handles = [b.register_forward_hook(block_hook) for b in backbone.blocks]
        handles.append(backbone.norm.register_forward_hook(norm_hook))
        try:
            if joint:
                logits, splat = model(x, positions[:, None])
                final['splat_probs'] = splat[:, 0].softmax(-1)
            else:
                logits = model(x)
            final['raw_logits'] = logits[ix, positions]
        finally:
            for handle in handles:
                handle.remove()
        final['raw_probs'] = final['raw_logits'].softmax(-1)
        mask = torch.zeros((len(part), 28), dtype=torch.bool, device=device)
        for i, row in enumerate(part):
            mask[i, row['legal']] = True
        final['legal_probs'] = final['raw_logits'].masked_fill(~mask, float('-inf')).softmax(-1)
        for i, name in enumerate(('layer_residual', 'mlp_pre', 'mlp_post', 'attention_to_prefix')):
            final[name] = torch.stack([b[i] for b in blocks], dim=1)
        for key, value in final.items():
            array = value.cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError('nonfinite model output: ' + key)
            gathered.setdefault(key, []).append(array)
    result = {k: np.concatenate(v) for k, v in gathered.items()}
    result['row_ids'] = np.asarray([r['id'] for r in rows])
    return result


def observations(rows, arrays, head):
    n = len(rows); indices = np.arange(n)
    action = np.array([r['reference_action'] for r in rows])
    legal = np.zeros((n, 28), bool)
    for i, r in enumerate(rows):
        legal[i, r['legal']] = True
    choice = np.where(legal, arrays['raw_logits'], -np.inf).argmax(-1)
    if head == 'splat':
        if 'splat_probs' not in arrays:
            raise ValueError('missing splat predictions')
        make = arrays['splat_probs'][..., 30:].sum(-1)
        bidder_side = np.array([r['question']['viewer'] % 2 == r['question']['bidder'] % 2 for r in rows])
        score = np.where(bidder_side[:, None], make, 1 - make)
        choice = np.where(legal, score, -np.inf).argmax(-1)
    arrays['chosen_tile'] = choice
    result = {'raw_tile_nll': -np.log(arrays['raw_probs'][indices, action].clip(1e-30)),
              'legal_tile_nll': -np.log(arrays['legal_probs'][indices, action].clip(1e-30)),
              'raw_action_agreement': (arrays['raw_logits'].argmax(-1) == action).astype(float),
              'policy_action_agreement': (choice == action).astype(float),
              'illegal_raw_mass': (arrays['raw_probs'] * ~legal).sum(-1),
              'legal_entropy': -(arrays['legal_probs'] * np.log(arrays['legal_probs'].clip(1e-30))).sum(-1)}
    if 'splat_probs' in arrays:
        distribution = arrays['splat_probs'][indices, action]
        points = np.array([r['outcome']['team_points'][r['question']['bidder'] % 2] for r in rows])
        made = np.array([r['outcome']['bidder_made'] for r in rows], dtype=float)
        estimate = distribution[:, 30:].sum(-1)
        result['recorded_action_outcome_nll'] = -np.log(distribution[indices, points].clip(1e-30))
        result['recorded_action_make_brier'] = (estimate - made) ** 2
    exact = np.full(n, np.nan); optimal = np.full(n, np.nan)
    for i, row in enumerate(rows):
        cert = row.get('exact_certificate')
        if cert:
            from ..rlvr.verifier import MODEL_ID
            if cert['model_id'] != MODEL_ID:
                raise ValueError('exact certificate model mismatch')
            values = {int(k): Fraction(v) for k, v in cert['values'].items()}
            if set(values) != set(row['legal']):
                raise ValueError('exact certificate legal support mismatch')
            exact[i] = float(max(values.values()) - values[int(choice[i])])
            optimal[i] = float(values[int(choice[i])] == max(values.values()))
    result['exact_endgame_regret'] = exact; result['exact_endgame_accuracy'] = optimal
    return result


def clustered(values, rows, mask, draws=2000):
    """Ratio of totals, resampling complete fixed-seed duplicate pairs."""
    mask = np.asarray(mask, bool) & np.isfinite(values)
    if not mask.any():
        return None
    keys = sorted({r['pair_id'] for r, use in zip(rows, mask) if use})
    mapping = {key: i for i, key in enumerate(keys)}
    cluster = np.array([mapping[r['pair_id']] for r, use in zip(rows, mask) if use])
    sums = np.bincount(cluster, weights=values[mask], minlength=len(keys))
    counts = np.bincount(cluster, minlength=len(keys))
    ci = None
    if len(keys) > 1:
        ix = np.random.default_rng(42).integers(len(keys), size=(draws, len(keys)))
        means = sums[ix].sum(-1) / counts[ix].sum(-1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return {'estimate': float(sums.sum() / counts.sum()), 'ci95': ci,
            'rows': int(mask.sum()), 'pairs': len(keys)}


def summarize(rows, arrays, head, reference=None):
    obs = observations(rows, arrays, head)
    slices = {'all': np.ones(len(rows), bool),
              'unsettled_discretionary': np.array([len(r['legal']) > 1 and not r['public']['settled'] for r in rows])}
    for name in sorted({r['panel'] for r in rows}):
        slices['panel/' + name] = np.array([r['panel'] == name for r in rows])
    for behavior in sorted({r['behavior_id'] for r in rows}):
        slices['behavior/' + behavior] = np.array([r['behavior_id'] == behavior for r in rows])
    result = {'metrics': {name: {metric: clustered(value, rows, mask) for metric, value in obs.items()}
                          for name, mask in slices.items()}}
    if reference:
        report, prior = reference
        old = observations(rows, prior, report['head'])
        deltas = {key: value - old[key] for key, value in obs.items() if key in old}
        deltas['action_change_rate'] = (arrays['chosen_tile'] != prior['chosen_tile']).astype(float)
        midpoint = (arrays['legal_probs'] + prior['legal_probs']) / 2
        deltas['legal_js_divergence'] = sum((p * (np.log(p.clip(1e-30)) - np.log(midpoint.clip(1e-30)))).sum(-1)
                                           for p in (arrays['legal_probs'], prior['legal_probs'])) / 2
        result['paired_delta'] = {name: {key: clustered(value, rows, mask) for key, value in deltas.items()}
                                  for name, mask in slices.items()}
        if arrays['actor_residual'].shape == prior['actor_residual'].shape:
            a, b = arrays['actor_residual'], prior['actor_residual']
            cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)).clip(1e-30)
            result['residual_coordinate_drift'] = {
                'mean_cosine': float(cosine.mean()), 'rms_difference': float(np.sqrt(((a - b) ** 2).mean())),
                'scope': 'same coordinates at fixed inputs; not a neuron identity or causal interpretation claim'}
    return result, obs


def run(args):
    if args.batch < 1:
        raise ValueError('positive batch size required')
    start = time.monotonic(); torch.set_num_threads(1)
    manifest, rows = load(args.benchmark)
    identity = {'panel_id': manifest['panel_id'], 'rows_sha256': manifest['rows_sha256'],
                'manifest_sha256': digest(Path(args.benchmark) / 'manifest.json'),
                'capture_source_sha256': digest(__file__),
                'checkpoint_sha256': digest(args.checkpoint), 'head': args.head,
                'capture_version': CAPTURE_VERSION, 'device': args.device, 'batch': args.batch,
                'torch_version': str(torch.__version__), 'reference_report_sha256': digest(Path(args.reference) / 'report.json') if args.reference else None}
    out = Path(args.out)
    if out.exists():
        saved = json.loads((out / 'report.json').read_text())
        if saved['identity'] != identity or digest(out / 'representations.npz') != saved['representations_sha256']:
            raise ValueError('existing output has a different model, panel or configuration')
        return saved
    model, joint, checkpoint = load_model(args.checkpoint, args.head, args.device)
    arrays = capture(model, joint, rows, args.device, args.batch)
    reference = None
    if args.reference:
        prior_path = Path(args.reference)
        report = json.loads((prior_path / 'report.json').read_text())
        if report['identity']['rows_sha256'] != identity['rows_sha256'] or digest(prior_path / 'representations.npz') != report['representations_sha256']:
            raise ValueError('reference panel or representation bytes differ')
        with np.load(prior_path / 'representations.npz', allow_pickle=False) as raw:
            prior = dict(raw)
        if not np.array_equal(arrays['row_ids'], prior['row_ids']):
            raise ValueError('reference row order differs')
        reference = report, prior
    report, obs = summarize(rows, arrays, args.head, reference)
    arrays.update({'metric_' + k: v for k, v in obs.items()})
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'representations.npz', **arrays)
    report.update(identity=identity, head=args.head, config=checkpoint['config'], rows=len(rows),
                  reference_checkpoint_sha256=reference[0]['identity']['checkpoint_sha256'] if reference else None,
                  representations_sha256=digest(out / 'representations.npz'),
                  bytes=(out / 'representations.npz').stat().st_size,
                  array_shapes={k: list(v.shape) for k, v in arrays.items()},
                  seconds=time.monotonic() - start, platform=platform.platform(),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  uncertainty='2000 duplicate-pair percentile bootstrap draws, seed42; regression diagnostic, not fresh confirmation',
                  scopes={'behavior': 'agreement/NLL on frozen recorded histories, not optimal-move accuracy',
                          'outcomes': 'proper scoring on recorded-action continuation outcomes; not outcomes of new chosen actions',
                          'exact_endgames': 'only covered rows under the pinned three-style finite model, not the H2H population',
                          'activations': 'last actor residual, MLP pre/post and attention to its causal prefix; no neuron semantics inferred',
                          'strength': 'changed-policy wins require rerunning the same frozen seed schedule'})
    atomic_json(out / 'report.json', report)
    print(json.dumps({'rows': len(rows), 'seconds': report['seconds'], 'bytes': report['bytes'],
                      'unsettled_discretionary': report['metrics']['unsettled_discretionary']}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--head', choices=('tile', 'splat'), default='tile'); p.add_argument('--reference')
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu'); p.add_argument('--batch', type=int, default=128)
    return p


if __name__ == '__main__':
    run(parser().parse_args())

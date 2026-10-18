# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Small, deal-disjoint rehearsal bundles from the original lawful splat data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import QUERIES, VERSION
from ..population import atomic_json, digest
from ..splat.data import Corpus
from ..splat.model import joint_loss

SCHEMA = 'gus-rehearsal-v1'
KEYS = ('x', 'targets', 'own_positions', 'histograms', 'candidates')


def build(args):
    out = Path(args.out)
    settings = {'schema': SCHEMA, 'tokenizer': VERSION, 'train_deals_per_source': args.train,
                'validation_deals_per_source': args.validation, 'seed': args.seed,
                'sources': [digest(Path(p) / 'manifest.json') for p in args.cache],
                'exclusions_sha256': digest(args.exclude)}
    if out.exists():
        existing = json.loads((out / 'manifest.json').read_text())
        if existing['settings'] != settings or digest(out / 'rehearsal.npz') != existing['sha256']:
            raise ValueError('rehearsal output differs')
        return existing
    excluded = {tuple(map(int, h)) for h in np.load(args.exclude, allow_pickle=False)}
    rng = np.random.default_rng(args.seed); gathered = {}; seen = set(); receipts = []
    for source, path in enumerate(args.cache):
        corpus = Corpus([path]); shard = corpus.shards[0]
        for split, number in enumerate((args.train, args.validation)):
            eligible = np.flatnonzero(shard['split'][::4] == split)
            chosen = []
            for i in rng.permutation(eligible):
                hands = tuple(map(int, shard['hands'][i]))
                if hands in excluded or hands in seen:
                    continue
                chosen.append(int(i)); seen.add(hands)
                if len(chosen) == number:
                    break
            if len(chosen) != number:
                raise ValueError('insufficient distinct unreserved rehearsal deals')
            indices = (np.asarray(chosen)[:, None] * 4 + np.arange(4)).ravel()
            values = corpus.batch(indices)
            values.update(split=np.full(len(indices), split, np.int8),
                          source=np.full(len(indices), source, np.int16), indices=indices,
                          hands=np.repeat(shard['hands'][chosen], 4, axis=0))
            for key, value in values.items():
                gathered.setdefault(key, []).append(value)
            receipts.append({'source': str(path), 'split': split, 'deal_indices': chosen})
    arrays = {k: np.concatenate(v) for k, v in gathered.items()}
    out.mkdir(parents=True)
    np.savez_compressed(out / 'rehearsal.npz', **arrays)
    manifest = {'settings': settings, 'sha256': digest(out / 'rehearsal.npz'),
                'views': len(arrays['x']), 'physical_deals': len(seen), 'selection': receipts,
                'scope': 'original train rows for rehearsal; original validation rows for retention selection; no test or benchmark rows'}
    atomic_json(out / 'manifest.json', manifest)
    return manifest


class Rehearsal:
    def __init__(self, root, device):
        root = Path(root); self.manifest = json.loads((root / 'manifest.json').read_text())
        if self.manifest['settings']['schema'] != SCHEMA or self.manifest['settings']['tokenizer'] != VERSION:
            raise ValueError('rehearsal schema or tokenizer differs')
        if digest(root / 'rehearsal.npz') != self.manifest['sha256']:
            raise ValueError('rehearsal checksum differs')
        with np.load(root / 'rehearsal.npz', allow_pickle=False) as data:
            self.arrays = {k: torch.as_tensor(data[k], device=device) for k in KEYS}
            self.hands = data['hands'].copy()
            self.splits = {name: np.flatnonzero(data['split'] == i) for i, name in enumerate(('train', 'validation'))}
        if any(not len(v) for v in self.splits.values()):
            raise ValueError('rehearsal needs train and validation')
        a, b = [{tuple(map(int, h)) for h in self.hands[ix]} for ix in self.splits.values()]
        if a & b:
            raise ValueError('rehearsal physical-deal split overlap')
        self.device = device

    def loss(self, model, indices):
        ix = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        batch = {k: v[ix] for k, v in self.arrays.items()}
        tile, splat = model(batch['x'], batch['own_positions'])
        total, tile_ce, splat_ce = joint_loss(tile[:, QUERIES], batch['targets'], splat,
                                               batch['histograms'], batch['candidates'])
        accuracy = (tile[:, QUERIES].argmax(-1) == batch['targets']).float().mean()
        return total, tile_ce, splat_ce, accuracy

    @torch.inference_mode()
    def evaluate(self, model, split='validation'):
        indices = self.splits[split]; totals = np.zeros(3); positions = 0
        for start in range(0, len(indices), 128):
            ix = indices[start:start + 128]
            _, tile, splat, accuracy = self.loss(model, ix)
            count = int((self.arrays['histograms'][ix].sum(-1).sum(-1) > 0).sum())
            totals += [float(tile) * len(ix), float(splat) * count, float(accuracy) * len(ix)]
            positions += count
        tile, splat, accuracy = totals / [len(indices), max(positions, 1), len(indices)]
        return {'joint_loss': float(tile + splat), 'tile_nll': float(tile),
                'splat_ce': float(splat), 'tile_accuracy': float(accuracy)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', action='append', required=True); p.add_argument('--exclude', required=True)
    p.add_argument('--out', required=True); p.add_argument('--train', type=int, default=128)
    p.add_argument('--validation', type=int, default=32); p.add_argument('--seed', type=int, default=42)
    print(build(p.parse_args()), flush=True)

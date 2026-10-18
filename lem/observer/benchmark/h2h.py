# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""One challenger faceoff on an immutable schedule; wrap in cpu-slot timeout60.

The same initial deals and contracts are reused, with fresh play after every
changed action. Known reference outcomes provide a paired regression baseline.
"""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import torch

from arena.bidders import Bid30Bidder
from arena.engine import ArenaConfig
from arena.match import run_match
from arena.walt_play import parse_walt_spec
from ..play import ObserverPlay
from ..splat.model import TASK
from ..splat.play import SplatPlay
from .build import decode_games, digest, identity, load, schedule, verify_games, write_json, write_rows


def asset(root, manifest, key):
    entry = manifest['assets'][key]
    path = Path(root) / entry['path']
    if digest(path) != entry['sha256']:
        raise ValueError('frozen asset hash mismatch: ' + key)
    return path


def check_runtime(root):
    path = Path(root) / 'runtime-pins.json'
    pins = json.loads(path.read_text())
    if pins['schema'] != 'gus-frozen-runtime-pins-v1':
        raise ValueError('unsupported frozen runtime protocol')
    checkout = Path(__file__).resolve().parents[3]
    for name, sha in pins['source_files'].items():
        if digest(checkout / name) != sha:
            raise ValueError('frozen H2H runtime changed: ' + name)
    current = {'python': platform.python_version(), 'numpy': np.__version__,
               'torch': str(torch.__version__), 'scenario_core': version('scenario_core')}
    if pins['versions'] != current:
        raise ValueError('frozen H2H runtime versions differ; new protocol/parity required')
    return digest(path)


def checked_pair(root, manifest, pair_id):
    pairs = [p for p in manifest['pairs'] if p['id'] == pair_id]
    if len(pairs) != 1:
        raise ValueError('unknown or ambiguous frozen pair')
    pair = pairs[0]; audit = Path(root) / pair['audit']
    for name, field in (('schedule.json', 'schedule_sha256'), ('games.jsonl.gz', 'games_sha256'),
                        ('manifest.json', 'source_manifest_sha256')):
        if digest(audit / name) != pair[field]:
            raise ValueError('frozen pair changed: ' + name)
    frozen = json.loads((audit / 'schedule.json').read_text())['hands']
    if schedule(pair['seed']) != frozen:
        raise ValueError('deal/auction implementation differs from frozen schedule')
    for name, sha in manifest['source_files'].items():
        if digest(name) != sha:
            raise ValueError('frozen H2H implementation changed: ' + name)
    return pair, frozen, decode_games(audit / 'games.jsonl.gz')


def run(args):
    torch.set_num_threads(1)
    root = Path(args.benchmark); manifest, _ = load(root)
    runtime_sha = check_runtime(root)
    pair, frozen, known = checked_pair(root, manifest, args.pair)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('task') == TASK:
        challenger = SplatPlay(args.checkpoint, 'cpu', args.head)
    else:
        if args.head != 'tile':
            raise ValueError('original observer requires tile head')
        challenger = ObserverPlay(args.checkpoint, 'cpu')
    opponent_identity = pair['opponent']
    if 'policy' in opponent_identity and opponent_identity['policy'].startswith('walt:'):
        field = asset(root, manifest, 'zeb_field')
        if opponent_identity['field_sha256'] != digest(field):
            raise ValueError('opponent field differs from pinned asset')
        spec = ','.join('F=' + str(field) if part.startswith('F=') else part
                        for part in opponent_identity['policy'][5:].split(','))
        opponent = parse_walt_spec(spec)
    else:
        pinned = asset(root, manifest, 'g0')
        if opponent_identity['sha256'] != digest(pinned):
            raise ValueError('unsupported opponent; frozen portable asset missing')
        opponent = ObserverPlay(pinned, 'cpu')
    key = {'panel_id': manifest['panel_id'], 'manifest_sha256': digest(root / 'manifest.json'),
           'runtime_pins_sha256': runtime_sha,
           'pair_id': args.pair, 'seed': pair['seed'], 'challenger_sha256': digest(args.checkpoint),
           'head': args.head, 'opponent': opponent_identity, 'torch_version': str(torch.__version__),
           'runner_sha256': digest(__file__)}
    out = Path(args.out)
    if out.exists():
        report = json.loads((out / 'report.json').read_text())
        if report['identity'] != key or digest(out / 'games.jsonl.gz') != report['games_sha256']:
            raise ValueError('existing result belongs to another frozen faceoff')
        return report
    out.mkdir(parents=True)
    write_json(out / 'manifest.json', key)
    result = run_match(bid_a=Bid30Bidder(), bid_b=Bid30Bidder(), play_a=challenger, play_b=opponent,
                       n_games=2, cfg=ArenaConfig(base_seed=pair['seed']), fast_batching=True,
                       label_a='challenger', label_b='pinned opponent', verbose=True)
    verify_games(result.games, frozen, set())
    write_rows(out / 'games.jsonl.gz', [asdict(g) for g in result.games])
    old = {g.a_team: g for g in known}
    differences = [{'challenger_team': g.a_team, 'won': g.a_won, 'known_won': old[g.a_team].a_won,
                    'marks_margin': g.a_marks - g.b_marks,
                    'known_marks_margin': old[g.a_team].a_marks - old[g.a_team].b_marks,
                    'identical_record': g == old[g.a_team]} for g in result.games]
    report = {'identity': key, 'games': 2, 'wins': sum(g.a_won for g in result.games),
              'known_reference_wins': pair['known_candidate_game_wins'], 'paired_outcomes': differences,
              'marks_per_game_delta': sum(d['marks_margin'] - d['known_marks_margin'] for d in differences) / 2,
              'games_sha256': digest(out / 'games.jsonl.gz'), 'schedule_and_replay_verified': True,
              'same_reference_trajectory': all(d['identical_record'] for d in differences),
              'elapsed_s': result.elapsed_s, 'scope': 'one fixed duplicate pair; no population strength interval'}
    write_json(out / 'report.json', report); print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark', 'checkpoint', 'pair', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--head', choices=('tile', 'splat'), default='tile')
    run(p.parse_args())

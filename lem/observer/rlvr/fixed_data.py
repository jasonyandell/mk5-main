# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Resumable fixed-field curriculum: reserved roots, immutable complete units.

Each unit collects a fixed batch of parent trajectories, then verifies every
eligible learner endgame. Split quotas select the first eligible physical
deals, never outcomes or model correctness. The plan reserves all possible roots.
"""
import argparse
import fcntl
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from forge.zeb.game import apply_action, current_player, legal_actions
from ..benchmark.build import canonical, identity
from ..benchmark.extension import Extension
from ..outcomes import trajectories, source_identity, publicly_settled, validate_plan
from ..population import atomic_json, digest, initial_state
from ..reward import policies
from ..splat.play import SplatPlay
from ..splat.recovery import StopRequest
from .data import tokens, load
from .fixed_field import specification, solve
from .rehearsal import Rehearsal
from .verifier import mask

SCHEMA = 'gus-fixed-field-curriculum-v1'


def produce(specs, split, model, field, model_id, namespace):
    decisions, episodes, _ = trajectories(model, field, specs, 1, namespace, 'cpu')
    questions, witnesses = [], {}
    for hand, episode in enumerate(episodes):
        state = initial_state(episode)
        for ply, (actor, tile) in enumerate(episode['plays']):
            if (actor == episode['learner'] and 20 <= ply < 24
                    and len(legal_actions(state)) == 2
                    and not publicly_settled(state.team_points, episode['bidder'])):
                q = {'bid': 30, 'decl': state.decl_id, 'bidder': state.bidder,
                     'viewer': actor, 'own': list(state.hands[actor]),
                     'plays': [list(p) for p in state.play_history]}
                context = {'dealer': episode['dealer'], 'bids': episode['bids']}
                certificate, witness = solve(q, field, model_id, context)
                q['certificate'] = certificate
                ix = int(np.flatnonzero(decisions['hand'] == hand)[5])
                clean = tokens(q)
                if decisions['x'][ix, :decisions['length'][ix]].tolist() != clean:
                    raise ValueError('curriculum prefix differs from trajectory input')
                key = identity([split, episode['seed'], clean])
                row = {'id': key, 'split': split, 'deal_seed': episode['seed'],
                       'physical_hands': [mask(h) for h in episode['hands']],
                       'question': q, 'tokens': clean, 'audit_context': context,
                       'source_action': tile, 'source_probabilities': decisions['old_probs'][ix].tolist(),
                       'witness_sha256': identity(witness)}
                questions.append(row); witnesses[key] = witness
            state = apply_action(state, state.hands[current_player(state)].index(tile))
    return {'episodes': episodes, 'questions': questions, 'witnesses': witnesses}


def build(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / 'writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another curriculum builder owns this output') from None
        return _build(args, out)


def _build(args, out):
    if args.seconds <= 0:
        raise ValueError('positive cooperative seconds required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    plan = json.loads(Path(args.plan).read_text())
    if plan['schema'] != SCHEMA or plan['source'] != source_identity():
        raise ValueError('curriculum source or schema changed')
    if digest(plan['initial']) != plan['initial_sha256'] or plan['batch'] < 1:
        raise ValueError('invalid parent or batch pin')
    replay = Rehearsal(plan['replay'], 'cpu')
    if replay.manifest['sha256'] != plan['replay_sha256']:
        raise ValueError('rehearsal bundle pin changed')
    validate_plan({**plan, 'development': plan['roots']['validation'],
                   'confirmation': plan['roots']['test'], 'train': plan['roots']['train']}, replay)
    if any(plan['counts'][s] < 1 or len(plan['roots'][s]) < plan['counts'][s] for s in plan['counts']):
        raise ValueError('positive quotas within reserved root budgets required')
    model_spec = specification(plan['field']); model_id = identity(model_spec)
    if model_spec != plan['model'] or model_id != plan['model_id']:
        raise ValueError('fixed continuation model changed')
    store = Extension(out / 'units', {'schema': SCHEMA, 'plan': plan})
    if (out / 'manifest.json').exists() and not args.recheck:
        return load(out)[0]
    player = SplatPlay(plan['initial'], 'cpu', 'tile')
    field = policies(plan['field'], 'cpu')
    selected, unit_files, screened = [], {}, {s: 0 for s in plan['counts']}
    counts = {s: 0 for s in plan['counts']}
    with StopRequest(args.stop_file, args.seconds) as stop:
        for split in ('train', 'validation', 'test'):
            roots = plan['roots'][split]
            for start in range(0, len(roots), plan['batch']):
                if counts[split] >= plan['counts'][split]:
                    break
                if stop.requested():
                    atomic_json(out / 'status.json', {'state': 'paused', 'counts': counts, 'screened': screened})
                    return None
                unit = {'id': f'{split}-{start:06d}', 'question': {'specs': roots[start:start+plan['batch']], 'split': split}}
                record = store.ensure(unit, 0, lambda row, _: produce(row['question']['specs'], split,
                    player.model, field, model_id, plan['namespace']), recheck=args.recheck)
                if record is None:
                    raise ValueError('curriculum unit is busy; resume after its owner finishes')
                name = identity(record['key']) + '.json'
                unit_files[name] = digest(store.records / name)
                screened[split] += len(record['result']['episodes'])
                for row in record['result']['questions']:
                    if counts[split] < plan['counts'][split]:
                        selected.append({**row, 'unit': name}); counts[split] += 1
                print({'split': split, 'counts': counts, 'screened': screened}, flush=True)
            if counts[split] != plan['counts'][split]:
                raise ValueError(f'reserved {split} roots exhausted before quota; register a new plan')
    temporary = out / 'questions.tmp'
    with temporary.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as stream:
        for row in selected:
            stream.write((canonical(row) + '\n').encode())
    temporary.replace(out / 'questions.jsonl.gz')
    result = {'schema': SCHEMA, 'model': model_spec, 'model_id': model_id,
              'plan_sha256': identity(plan), 'counts': counts, 'screened': screened,
              'questions_sha256': digest(out / 'questions.jsonl.gz'), 'units': unit_files,
              'selection': 'first eligible unsettled two-legal learner decision per reserved physical deal; no correctness filtering',
              'reserved_physical_deals': sum(map(len, plan['roots'].values()))}
    atomic_json(out / 'manifest.json', result)
    load(out)  # Validate the learner rows against their immutable unit witnesses.
    store.snapshot()
    atomic_json(out / 'status.json', {'state': 'complete', 'counts': counts, 'screened': screened})
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=45); p.add_argument('--stop-file')
    p.add_argument('--recheck', action='store_true')
    return p


if __name__ == '__main__':
    result = build(parser().parse_args())
    if result:
        print({'complete': result['counts'], 'screened': result['screened'],
               'questions_sha256': result['questions_sha256']}, flush=True)

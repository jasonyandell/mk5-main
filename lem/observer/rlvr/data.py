# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Fresh deal-disjoint exercises; SQLite transactions make construction resumable."""
import argparse
from dataclasses import replace
from fractions import Fraction
import gzip
import json
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import torch

from forge.zeb.game import new_game, apply_action, current_player, legal_actions
from forge.zeb.types import BidState
from ..data import encode, SEAT, partition
from ..population import atomic_json, digest, seed_for
from ..splat.play import SplatPlay
from ..splat.recovery import StopRequest
from .verifier import MODEL, MODEL_ID, solve, mask

NAMESPACE = 'gus-rlvr-silas-v1-20261018'


def tokens(question):
    return np.append(encode(mask(question['own']), question['viewer'], question['bidder'],
                            question['decl'], question['plays']), SEAT).tolist()


def candidate_questions(seed):
    state = new_game(seed)
    bidder = state.bidder
    # Contract and declaration are exogenous public problem parameters.
    state = replace(state, decl_id=seed % 7,
                    bid_state=BidState(tuple(30 if i == bidder else 0 for i in range(4)), bidder, 30))
    full_hands = [mask(h) for h in state.hands]
    rng = random.Random(seed_for(NAMESPACE, 'history', seed))
    candidates = []
    for ply in range(24):
        legal = legal_actions(state)
        viewer = current_player(state)
        if (ply >= 20 and len(legal) == 2 and state.team_points[bidder % 2] < 30
                and state.team_points[1 - bidder % 2] < 13):
            candidates.append({'bid': 30, 'decl': state.decl_id, 'bidder': bidder,
                               'viewer': viewer, 'own': list(state.hands[viewer]),
                               'plays': [list(p) for p in state.play_history]})
        state = apply_action(state, rng.choice(legal))
    return full_hands, candidates


@torch.inference_mode()
def baseline_choices(model, questions):
    rows = [tokens(q) for q in questions]
    x = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
    for i, row in enumerate(rows):
        x[i, :len(row)] = torch.tensor(row)
    logits = model.backbone(x)[torch.arange(len(rows)), [len(r) - 1 for r in rows]]
    return [max(c['legal'], key=lambda a: float(logits[i, a]))
            for i, c in enumerate(q['certificate'] for q in questions)]


def build(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    settings = {'namespace': args.namespace, 'counts': {'train': args.train, 'validation': args.validation, 'test': args.test},
                'model': MODEL, 'model_id': MODEL_ID, 'baseline_sha256': digest(args.baseline),
                'excluded_sha256': digest(args.exclude),
                'selection': 'highest frozen-baseline exact regret among penultimate-trick choices in each TRAIN deal; first discretionary unsettled position in validation/test; ties earliest'}
    db = sqlite3.connect(out / 'build.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS questions (deal TEXT PRIMARY KEY, split TEXT, payload TEXT)')
    old = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
    if old and json.loads(old[0]) != settings:
        raise ValueError('builder resume settings differ')
    if not old:
        with db:
            db.execute('INSERT INTO meta VALUES (?,?)', ('settings', json.dumps(settings)))
            db.execute('INSERT INTO meta VALUES (?,?)', ('cursor', '0'))
    manifest = out / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if digest(out / 'questions.jsonl.gz') != saved['questions_sha256']:
            raise ValueError('completed exercise archive changed')
        db.close(); return saved
    excluded = {tuple(map(int, h)) for h in np.load(args.exclude)}
    # The mechanical feasibility probe is never part of train/dev/test.
    excluded.update(tuple(mask(h) for h in new_game(i).hands) for i in range(400))
    player = SplatPlay(args.baseline, 'cpu', 'tile')
    counts = dict(db.execute('SELECT split, COUNT(*) FROM questions GROUP BY split'))
    counts = {s: counts.get(s, 0) for s in settings['counts']}
    cursor = int(db.execute("SELECT value FROM meta WHERE key='cursor'").fetchone()[0])
    start = time.monotonic()
    with StopRequest(seconds=args.seconds) as stop:
        while any(counts[s] < n for s, n in settings['counts'].items()):
            if stop.requested():
                db.close(); print({'paused': stop.reason, 'cursor': cursor, 'counts': counts}, flush=True); return
            seed = seed_for(args.namespace, 'deal', cursor)
            hands, questions = candidate_questions(seed)
            split = partition(hands)
            key = json.dumps(hands)
            row = None
            if (tuple(hands) not in excluded and counts[split] < settings['counts'][split]
                    and questions and not db.execute('SELECT 1 FROM questions WHERE deal=?', (key,)).fetchone()):
                if split != 'train':
                    questions = questions[:1]
                for q in questions:
                    q['certificate'] = solve(q)
                choices = baseline_choices(player.model, questions)
                regrets = []
                for q, action in zip(questions, choices):
                    values = {int(a): Fraction(v) for a, v in q['certificate']['values'].items()}
                    regrets.append(max(values.values()) - values[action])
                index = max(range(len(questions)), key=lambda i: regrets[i]) if split == 'train' else 0
                row = {'deal_seed': seed, 'deal_index': cursor, 'physical_hands': hands, 'split': split,
                       'question': questions[index], 'tokens': tokens(questions[index]),
                       'baseline_action': choices[index], 'baseline_regret': str(regrets[index]),
                       'selection_candidates': [{'question': q, 'baseline_action': a, 'regret': str(r)}
                                                for q, a, r in zip(questions, choices, regrets)]}
            cursor += 1
            # Row and cursor commit together. A crash can repeat work but cannot
            # silently omit, duplicate or relabel a committed physical deal.
            with db:
                if row:
                    db.execute('INSERT INTO questions VALUES (?,?,?)', (key, split, json.dumps(row)))
                    counts[split] += 1
                db.execute("UPDATE meta SET value=? WHERE key='cursor'", (str(cursor),))
            if row and sum(counts.values()) % 128 == 0:
                print({'cursor': cursor, 'counts': counts, 'segment_seconds': time.monotonic() - start}, flush=True)
    rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM questions ORDER BY rowid')]
    temporary = out / 'questions.tmp'
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True) + '\n').encode())
    temporary.replace(out / 'questions.jsonl.gz')
    result = {**settings, 'counts': counts, 'attempted_deals': cursor, 'excluded_physical_deals': len(excluded),
              'questions_sha256': digest(out / 'questions.jsonl.gz'), 'segment_seconds': time.monotonic() - start,
              'ambiguous_ties': {s: sum(len(r['question']['certificate']['best']) > 1 for r in rows if r['split'] == s)
                                 for s in counts}}
    atomic_json(manifest, result); db.close(); print(result, flush=True); return result


def load(root):
    root = Path(root); manifest = json.loads((root / 'manifest.json').read_text())
    from .fixed_field import valid_model
    fixed = valid_model(manifest)
    if (manifest['model_id'] != MODEL_ID and not fixed) or digest(root / 'questions.jsonl.gz') != manifest['questions_sha256']:
        raise ValueError('exercise specification or bytes changed')
    with gzip.open(root / 'questions.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len({tuple(r['physical_hands']) for r in rows}) != len(rows):
        raise ValueError('physical deal overlap')
    for r in rows:
        if r['tokens'] != tokens(r['question']):
            raise ValueError('input receipt differs from clean encoder')
        certificate = r['question']['certificate']
        values = {int(a): Fraction(v) for a, v in certificate['values'].items()}
        if (certificate['model_id'] != manifest['model_id'] or sorted(values) != certificate['legal']
                or len(values) != 2 or any(not 0 <= q <= 1 for q in values.values())
                or certificate['best'] != [a for a in sorted(values) if values[a] == max(values.values())]):
            raise ValueError('certificate identity, support or rational values changed')
    if fixed:
        from ..benchmark.extension import Extension
        from ..benchmark.build import identity
        recipe = json.loads((root / 'units/recipe.json').read_text())
        store = Extension(root / 'units', recipe['recipe'])
        if (identity(recipe['recipe']['plan']) != manifest['plan_sha256']
                or recipe['recipe']['plan']['model'] != manifest['model']):
            raise ValueError('curriculum plan identity changed')
        witnessed = {}
        for name, checksum in manifest['units'].items():
            path = store.records / name
            if path.parent != store.records or digest(path) != checksum:
                raise ValueError('curriculum unit checksum changed')
            result = store.read(path)['result']
            for row in result['questions']:
                if identity(result['witnesses'][row['id']]) != row['witness_sha256']:
                    raise ValueError('curriculum witness changed')
                witnessed[row['id']] = {**row, 'unit': name}
        if any(witnessed.get(row['id']) != row for row in rows):
            raise ValueError('training row differs from its verified unit')
    return manifest, rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--baseline', required=True); p.add_argument('--exclude', required=True)
    p.add_argument('--namespace', default=NAMESPACE)
    p.add_argument('--train', type=int, default=2048); p.add_argument('--validation', type=int, default=512)
    p.add_argument('--test', type=int, default=512); p.add_argument('--seconds', type=float, default=240)
    build(p.parse_args())

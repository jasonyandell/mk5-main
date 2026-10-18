# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Exact two-choice outcomes under pinned G0/Zeb continuation policies.

The root belief is uniform legal-consistent, not a behavioral posterior. One
root action is shared across worlds and the learner's last move is forced.
Hidden hands are referee state; every seated policy retains its own encoder.
"""
from fractions import Fraction
import json
from pathlib import Path
import platform

import numpy as np
import torch

from forge.zeb.game import apply_action, current_player, legal_actions
from ..benchmark.build import identity
from ..benchmark.continue_rows import remaining_worlds, replay_root
from ..population import digest, verify_episode
from .verifier import legal_policy, root_frame

VERSION = 'gus-fixed-g0-zeb-two-trick-v1'


def specification(config):
    """Pin policy content and execution, independently of asset path names."""
    from ..outcomes import source_identity
    if set(config['pool']) != {'g0'}:
        raise ValueError('exactly the fixed G0 partner is required')
    if digest(config['zeb_field']) != config['zeb_field_sha256']:
        raise ValueError('Zeb field pin changed')
    g0 = config['pool']['g0']
    if digest(g0['path']) != g0['sha256']:
        raise ValueError('G0 weight pin changed')
    parts = config['zeb_spec'].split(',')
    paths = [p[2:] for p in parts if p.startswith('F=')]
    if len(paths) != 1 or Path(paths[0]).resolve() != Path(config['zeb_field']).resolve():
        raise ValueError('Zeb policy must reference the pinned field')
    return {
        'version': VERSION, 'bid': 30,
        'root_belief': 'uniform legal-consistent remaining worlds; no pre-root behavior likelihood',
        'continuation': 'fixed G0 partner and pinned Zeb opponents; common root action across worlds; later learner move forced',
        'objective': 'viewer team makes or sets the bid30 contract',
        'g0_sha256': g0['sha256'], 'zeb_field_sha256': config['zeb_field_sha256'],
        'zeb_spec': ','.join('F=<pinned-field>' if p.startswith('F=') else p for p in parts),
        'source': source_identity(), 'platform': [platform.system(), platform.release(), platform.machine()],
        'python': platform.python_version(), 'numpy': np.__version__,
        'torch': str(torch.__version__), 'torch_build': torch.__config__.show(),
        'inference': 'individual CPU float32; torch threads1; Zeb threads2; deterministic policy budgets',
    }


def solve(question, field, model_id, context=None):
    """Return a rational certificate and complete independently replayed witness."""
    frame, _, _ = root_frame(question)
    legal = sorted(legal_policy(question['decl']).allowed(frame)[1])
    if len(legal) != 2:
        raise ValueError('a discretionary two-tile root is required')
    if frame.banked[question['bidder'] % 2] >= 30 or frame.banked[1-question['bidder'] % 2] >= 13:
        raise ValueError('the public outcome must remain unsettled')
    # Only the declared information enters the question, even if callers keep
    # unrelated source-deal or teacher sidecars in the same Python dictionary.
    q = {k: question[k] for k in ('bid', 'decl', 'bidder', 'viewer', 'own', 'plays')}
    worlds = remaining_worlds(json.dumps(q, sort_keys=True))
    records, values = [], {}
    for tile in legal:
        wins = 0
        for index, remaining in enumerate(worlds):
            state = replay_root(q, remaining, context)
            actor = current_player(state)
            options = legal_actions(state)
            if actor != q['viewer'] or sorted(state.hands[actor][i] for i in options) != legal:
                raise ValueError('root legal support changed across worlds')
            state = apply_action(state, state.hands[actor].index(tile))
            while len(state.played) < 28:
                actor = current_player(state); options = legal_actions(state)
                if actor == q['viewer']:
                    if len(options) != 1:
                        raise ValueError('a second learner choice requires information-state planning')
                    choice = options[0]
                else:
                    policy = field['g0' if actor % 2 == q['viewer'] % 2 else 'zeb']
                    choice, = policy.choose([state], [30])
                if choice not in options:
                    raise ValueError('fixed policy chose an illegal move')
                state = apply_action(state, choice)
            won = (state.team_points[q['bidder'] % 2] >= 30) == (q['viewer'] % 2 == q['bidder'] % 2)
            episode = {'hands': state.hands, 'decl': q['decl'], 'bidder': q['bidder'],
                       'learner': q['viewer'], 'plays': state.play_history,
                       'points': state.team_points, 'reward': 1. if won else -1.}
            verify_episode(episode)
            wins += won
            records.append({'world': index, 'action': tile, 'episode': episode})
        values[tile] = Fraction(wins, len(worlds))
    certificate = {
        'model_id': model_id, 'authority': f'exact_given({model_id})',
        'root_authority': 'exact', 'physics': frame.physics_identity,
        'worlds': len(worlds), 'legal': legal,
        'values': {str(a): str(v) for a, v in values.items()},
        'best': [a for a in legal if values[a] == max(values.values())],
        'uniform_value': str(sum(values.values()) / 2),
        'gap': str(max(values.values()) - min(values.values())),
    }
    witness = {'question': q, 'context': context, 'remaining_worlds': worlds, 'continuations': records}
    return json.loads(json.dumps(certificate)), json.loads(json.dumps(witness))


def valid_model(manifest):
    model = manifest.get('model', {})
    return model.get('version') == VERSION and identity(model) == manifest['model_id']

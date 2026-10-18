# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Lazily replay endgame continuations against the unchanged pinned Zeb field.

The reference uses Zeb in the viewer seat; a candidate replaces that seat only.
The other three seats retain the original anchor's G0/Zeb policies. This
measured field is separate from the exact curriculum's three-style field.
No lookahead is added to Gus. Hidden deals are referee/audit data, never tokens.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import platform
import random

import scenario_core
from scenario_core import SilasBelief
import torch

from arena.walt_play import parse_walt_spec
from forge.zeb.game import apply_action, current_player, legal_actions
from ..population import initial_state, verify_episode
from ..play import ObserverPlay
from ..rlvr.verifier import HOLDINGS, root_frame, tile_id
from ..splat.play import SplatPlay
from ..splat.recovery import StopRequest
from .build import digest, identity, load
from .extension import Extension, sample_seed
from .h2h import asset, check_runtime, checked_pair


@lru_cache(maxsize=128)
def remaining_worlds(question_json):
    question = json.loads(question_json)
    frame, own, _ = root_frame(question)
    count = frame.count_worlds()
    if not 0 < count <= 90:
        raise ValueError('endgame root exceeds the exact hidden-deal bound')
    summary = HOLDINGS.summarize(SilasBelief.uniform(frame, 90), 1_000_000)
    worlds = []
    for answers, weight in summary.answer_sets.items():
        hands = [0] * 4
        for seat, tile in answers:
            hands[int(seat[1:])] |= 1 << tile_id(tile)
        if hands[question['viewer']] != own or weight != Fraction(1, count):
            raise ValueError('root belief differs from the declared uniform law')
        worlds.append(tuple(hands))
    if summary.authority.label != 'exact' or len(worlds) != count:
        raise ValueError('root belief must be complete and exact')
    return tuple(sorted(worlds))


def replay_root(question, remaining, context=None):
    public = [{tile for seat, tile in question['plays'] if seat == i} for i in range(4)]
    hands = [sorted(public[i] | {t for t in range(28) if remaining[i] & (1 << t)}) for i in range(4)]
    if hands[question['viewer']] != sorted(question['own']):
        raise ValueError('sampled own hand changed')
    context = context or {'dealer': 0, 'bids': [30 if i == question['bidder'] else 0 for i in range(4)]}
    state = initial_state({'hands': hands, 'dealer': context['dealer'], 'bidder': question['bidder'],
        'decl': question['decl'], 'bids': context['bids']})
    for seat, tile in question['plays']:
        if current_player(state) != seat:
            raise ValueError('sampled history has the wrong actor')
        choice = state.hands[seat].index(tile)
        if choice not in legal_actions(state):
            raise ValueError('sampled hidden deal contradicts public play')
        state = apply_action(state, choice)
    return state


def continuation(row, sample, coupling, learner, field):
    """Complete one hand; independent rules replay certifies its terminal reward."""
    question = row['question']
    worlds = remaining_worlds(json.dumps(question, sort_keys=True))
    seed = sample_seed(coupling, row, sample, 'hidden-deal')
    world_index = random.Random(seed).randrange(len(worlds))
    state = replay_root(question, worlds[world_index], row.get('audit_context'))
    if current_player(state) != question['viewer']:
        raise ValueError('root viewer is not to act')
    actual_legal = sorted(state.hands[question['viewer']][a] for a in legal_actions(state))
    if actual_legal != row['legal']:
        raise ValueError('sampled root has different legal outputs')
    while len(state.played) < 28:
        actor = current_player(state)
        policy = learner if actor == question['viewer'] else field[actor]
        choice, = policy.choose([state], [30])
        if choice not in legal_actions(state):
            raise ValueError('continuation chose an illegal tile')
        state = apply_action(state, choice)
    made = state.team_points[question['bidder'] % 2] >= 30
    won = made == (question['viewer'] % 2 == question['bidder'] % 2)
    episode = {'hands': state.hands, 'decl': question['decl'], 'bidder': question['bidder'],
        'learner': question['viewer'], 'plays': state.play_history,
        'points': state.team_points, 'reward': 1. if won else -1.}
    verify_episode(episode)
    # JSON primitives also make a recomputation compare equal to a disk read.
    return json.loads(json.dumps({'seed': seed, 'world_index': world_index, 'worlds': len(worlds),
        'remaining': worlds[world_index], 'context': row.get('audit_context'),
        'root_action': state.play_history[len(question['plays'])][1],
        'won': won, 'episode': episode}))


def run(args):
    if args.samples < 1 or args.start < 0 or args.seconds <= 0:
        raise ValueError('positive samples/time and nonnegative starting index required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(args.benchmark); manifest, rows = load(root)
    runtime_sha = check_runtime(root)
    pair, = [p for p in manifest['pairs'] if p['id'] == 'anchor-0']
    opponent = pair['opponent']
    field_path = asset(root, manifest, 'zeb_field')
    if not opponent['policy'].startswith('walt:') or opponent['field_sha256'] != digest(field_path):
        raise ValueError('the pinned Zeb field is required')
    parts = opponent['policy'][5:].split(',')
    spec = ','.join('F=' + str(field_path) if p.startswith('F=') else p for p in parts)
    field = parse_walt_spec(spec)
    learner = SplatPlay(args.checkpoint, 'cpu', args.head) if args.checkpoint else field
    g0_path = asset(root, manifest, 'g0'); g0 = ObserverPlay(g0_path, 'cpu')
    field_identity = {'policy': ','.join('F=<pinned-field>' if p.startswith('F=') else p for p in parts),
                      'field_sha256': digest(field_path)}
    coupling = {'panel_sha256': digest(root / 'manifest.json'), 'seed': args.seed,
        'belief': 'uniform legal-consistent remaining deals; no pre-root behavior likelihood',
        'other_seats': {'zeb': field_identity, 'g0_sha256': digest(g0_path),
                        'assignment': 'unchanged original anchor seating except the viewer'},
        'terminal': 'complete 28 plays; viewer make/set bid30',
        'seed_scheme': 'sha256-canonical-v1; first 64 bits; named independent streams',
        'auction': 'original frozen anchor dealer/bids; never redealt', 'batch': 1}
    checkout = Path(__file__).resolve().parents[3]
    sources = ['lem/observer/benchmark/extension.py', 'lem/observer/benchmark/continue_rows.py',
               'lem/observer/benchmark/h2h.py', 'lem/observer/rlvr/verifier.py',
               'lem/observer/population.py', 'lem/observer/splat/recovery.py']
    native = {p.name: digest(p) for p in sorted(Path(scenario_core.__file__).parent.glob('*.so'))}
    if not native:
        raise ValueError('native runtime binary identity is required')
    recipe = {'schema': 'gus-lazy-continuations-v1', 'coupling': coupling,
        'runtime_pins_sha256': runtime_sha, 'source_sha256': {p: digest(checkout / p) for p in sources},
        'native_binary_sha256': native,
        'platform': [platform.system(), platform.release(), platform.machine()],
        'torch_build': torch.__config__.show(), 'device': 'cpu', 'dtype': 'float32', 'threads': 1,
        'learner': {'checkpoint_sha256': digest(args.checkpoint), 'head': args.head} if args.checkpoint else field_identity}
    selected = set(args.row or [])
    if args.row_file:
        selected.update(line.strip() for line in Path(args.row_file).read_text().splitlines() if line.strip())
    if not selected or selected - {r['id'] for r in rows}:
        raise ValueError('select existing frozen row IDs with --row or --row-file')
    rows = sorted((r for r in rows if r['id'] in selected), key=lambda r: r['id'])
    if any(r['pair_id'] not in ('anchor-0', 'anchor-1') or r['public']['settled']
           or len(r['legal']) != 2 or not 20 <= r['ply'] <= 23 for r in rows):
        raise ValueError('initial continuation scope: anchor rows, unsettled two-legal penultimate trick')
    audits = {}
    for pair_id in sorted({r['pair_id'] for r in rows}):
        pinned, _, games = checked_pair(root, manifest, pair_id)
        if pinned['opponent'] != opponent or pinned['candidate']['sha256'] != digest(g0_path):
            raise ValueError('anchor seating differs from the pinned policies')
        audits[pair_id] = {g.a_team: g for g in games}
    for row in rows:
        a_team = int(row['game_id'].rsplit('-', 1)[1])
        hand_index = int(row['hand_id'].rsplit('-', 1)[1])
        hand = audits[row['pair_id']][a_team].hands[hand_index]
        q = row['question']
        if (list(hand.hands[q['viewer']]) != q['own'] or hand.bidder != q['bidder']
                or hand.decl_id != q['decl'] or hand.bid_value != q['bid']
                or [list(p) for p in hand.plays[:row['ply']]] != q['plays']):
            raise ValueError('frozen row does not match its audited hand')
        row['audit_context'] = {'dealer': hand.dealer, 'bids': list(hand.bids), 'a_team': a_team}
    extension = Extension(args.out, recipe)
    completed = 0
    with StopRequest(args.stop_file, args.seconds) as stop:
        for row in rows:
            field_seats = tuple(g0 if i % 2 == row['audit_context']['a_team'] else field for i in range(4))
            for sample in range(args.start, args.samples):
                if stop.requested():
                    break
                result = extension.ensure(row, sample,
                    lambda r, s: continuation(r, s, coupling, learner, field_seats), args.recheck)
                completed += result is not None
            if stop.requested():
                break
        snapshot = extension.snapshot()
        report = {'requested': len(rows) * max(0, args.samples - args.start), 'completed': completed,
                  'stop': stop.reason, 'snapshot': str(snapshot), 'extension_id': extension.id}
    print(json.dumps(report), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True); p.add_argument('--out', required=True)
    p.add_argument('--row', action='append'); p.add_argument('--row-file')
    p.add_argument('--samples', type=int, default=1, help='desired total samples per row, exclusive end index')
    p.add_argument('--start', type=int, default=0, help='first sample index; allows independent work slices')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', help='omit for pinned Zeb baseline')
    p.add_argument('--head', choices=('tile', 'splat'), default='tile')
    p.add_argument('--seconds', type=float, default=45); p.add_argument('--stop-file')
    p.add_argument('--recheck', action='store_true', help='recompute completed units and require identical bytes')
    return p


if __name__ == '__main__':
    run(parser().parse_args())

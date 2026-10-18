# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""An exact finite endgame model, with Silas as the rules and belief authority.

The question is a single choice with two tiles left. The viewer's later tile
is forced; the other seats follow one of three declared lawful fields. There
is no optimization inside a hidden world. Fractions decide ties and rewards;
float conversion belongs only at the neural training boundary.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import json

from scenario_core import SilasBelief, SilasFrame, SilasPolicy, SilasQuery

HOLDINGS = SilasQuery('(query (roles (chair c) (domino d)) (out c d) (pattern (holds c d)))')
LEGAL_SOURCE = ('(policy all-legal (fallback lowest-legal) '
                '(rule all (select answer) (guard (query (roles (domino d)) '
                '(out d) (pattern (own-legal d))))))')
MODEL = {
    'version': 'silas-two-trick-rational-v1', 'bid': 30,
    'root_belief': 'uniform over every legal-consistent remaining hidden deal; behavior likelihood before the root is deliberately not modeled',
    'fields': ['all-other-seats-lowest-legal', 'all-other-seats-highest-legal', 'all-other-seats-uniform-legal'],
    'field_weights': ['1/3', '1/3', '1/3'],
    'field_timing': 'one shared table style drawn at the root and held to the end',
    'objective': 'viewer team makes or sets the bid30 contract',
    'future_viewer': 'one remaining tile, necessarily forced',
}
MODEL_ID = hashlib.sha256(json.dumps(MODEL, sort_keys=True).encode()).hexdigest()


class BudgetExceeded(ValueError):
    pass


def mask(tiles):
    return sum(1 << int(t) for t in tiles)


def tile_id(name):
    h, l = map(int, name.split('-'))
    return h * (h + 1) // 2 + l


@lru_cache(None)
def legal_policy(decl):
    return SilasPolicy(LEGAL_SOURCE, decl)


def root_frame(question):
    if question.get('bid', 30) != 30 or question['decl'] not in range(7):
        raise ValueError('this curriculum uses bid30 and pip declarations')
    plays = tuple(tuple(map(int, p)) for p in question['plays'])
    if not 20 <= len(plays) <= 23:
        raise ValueError('only decisions during the penultimate trick')
    viewer = int(question['viewer'])
    original = list(map(int, question['own']))
    if len(original) != 7 or len(set(original)) != 7 or any(t not in range(28) for t in original):
        raise ValueError('exactly seven distinct original own tiles are required')
    played = {t for _, t in plays}
    if {t for s, t in plays if s == viewer} - set(original):
        raise ValueError('own public plays disagree with the original hand')
    remaining = mask(t for t in original if t not in played)
    f = SilasFrame(question['decl'], viewer, remaining, question['bidder'], 30, list(plays))
    if (f.leader + len(f.trick)) % 4 != viewer or remaining.bit_count() != 2:
        raise ValueError('the viewer must be to act with exactly two tiles')
    return f, remaining, plays


def solve(question, node_cap=100_000):
    """Return exact action values and an independently checkable answer key.

    `question` has only the player's own original hand and public record. No
    source deal or teacher value is read, even when a caller retains one.
    """
    f, own, plays = root_frame(question)
    legal = sorted(legal_policy(question['decl']).allowed(f)[1])
    if len(legal) != 2:
        raise ValueError('the curriculum requires a discretionary two-tile choice')
    worlds = f.count_worlds()
    if not 0 < worlds <= 90:
        raise ValueError('two-trick fiber exceeds its combinatorial bound')
    belief = SilasBelief.uniform(f, 90)
    summary = HOLDINGS.summarize(belief, 1_000_000)
    if summary.authority.label != 'exact':
        raise ValueError('the root support must be enumerated, never sampled')
    law = []
    for answers, weight in summary.answer_sets.items():
        hands = [0] * 4
        for seat, tile in answers:
            hands[int(seat[1:])] |= 1 << tile_id(tile)
        if hands[question['viewer']] != own:
            raise ValueError('Silas holdings disagree with the viewer')
        law.append((tuple(hands), weight))
    law.sort()
    assert len(law) == worlds and sum(w for _, w in law) == 1
    assert all(w == Fraction(1, worlds) for _, w in law)
    nodes = 0
    viewer, bidder, decl = question['viewer'], question['bidder'], question['decl']

    @lru_cache(None)
    def value(hands, history, style):
        nonlocal nodes
        nodes += 1
        if nodes > node_cap:
            raise BudgetExceeded('exact continuation exceeded its deterministic node cap')
        # A frame replays the entire public record and independently derives
        # the acting seat and banked score. Only that seat's own hand is used
        # to obtain its legal choices. The analyst's world is not a policy input.
        public = SilasFrame(decl, viewer, hands[viewer], bidder, 30, list(history))
        made = public.banked[bidder % 2] >= 30
        set_hand = public.banked[1 - bidder % 2] >= 13
        if made or set_hand or len(history) == 28:
            return Fraction(int(made == (viewer % 2 == bidder % 2)))
        actor = (public.leader + len(public.trick)) % 4
        frame = SilasFrame(decl, actor, hands[actor], bidder, 30, list(history))
        options = sorted(legal_policy(decl).allowed(frame)[1])
        if actor == viewer:
            if len(options) != 1:
                raise ValueError('a second discretionary learner choice requires an information-state solver')
        elif style == 'lowest':
            options = options[:1]
        elif style == 'highest':
            options = options[-1:]
        total = Fraction(0)
        for tile in options:
            after = list(hands); after[actor] ^= 1 << tile
            total += value(tuple(after), history + ((actor, tile),), style)
        return total / len(options)

    by_field = {}
    for style in ('lowest', 'highest', 'uniform'):
        by_field[style] = {}
        for action in legal:
            total = Fraction(0)
            for hands, weight in law:
                after = list(hands); after[viewer] ^= 1 << action
                total += weight * value(tuple(after), plays + ((viewer, action),), style)
            by_field[style][action] = total
    q = {a: sum(by_field[s][a] for s in by_field) / 3 for a in legal}
    best = max(q.values())
    return {'model_id': MODEL_ID, 'authority': 'exact_given(' + MODEL_ID + ')',
            'root_authority': summary.authority.label, 'physics': f.physics_identity,
            'worlds': worlds, 'nodes': nodes, 'legal': legal,
            'values': {str(a): str(v) for a, v in q.items()},
            'by_field': {s: {str(a): str(v) for a, v in values.items()} for s, values in by_field.items()},
            'best': [a for a in legal if q[a] == best],
            'uniform_value': str(sum(q.values()) / len(q)),
            'gap': str(max(q.values()) - min(q.values()))}


def verify(answer, certificate):
    """Binary math-style correctness, including exact rational ties."""
    if certificate['model_id'] != MODEL_ID:
        raise ValueError('unknown finite-game specification')
    return int(int(answer) in certificate['best'])

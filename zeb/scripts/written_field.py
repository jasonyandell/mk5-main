# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""Bourne-0: write a small field policy directly into integer ReLU weights.

Run from the repository root: python -u -m zeb.scripts.written_field OUTDIR
No optimizer, examples, checkpoints, or forward passes enter construction.
The output uses the existing field-v2 encoder and legal-action mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scenario.field import CANON, OFFSET, FieldNet
from scenario.rules import rules_for, tiles_of


def build() -> tuple[FieldNet, dict]:
    """Compile explicit public-information heuristics, starting with zero arrays."""
    rules = rules_for(6)  # field-v2 always relabels the trump pip to 6
    first: list[tuple[str, dict[int, int], int]] = []
    second: list[tuple[str, dict[int, int], int, dict[int, int]]] = []

    def one(name, terms, bias=0):
        first.append((name, terms, bias))
        return len(first) - 1

    def two(name, terms, bias, outputs):
        second.append((name, terms, bias, outputs))

    led = {}
    for suit in range(6):
        led[suit] = []
        for n in (1, 2, 3):
            terms = {OFFSET['trick'] + (3 - n) * 28 + t: 1
                     for t in range(28) if rules.led_suit[t] == suit}
            terms[OFFSET['open'] + n] = 1
            led[suit].append(one(f'led_suit_{suit}_open_{n}', terms, -1))

    # A follower can take the trick iff no higher-ranked tile is already down.
    # Opposing trumps count as higher than an off-suit follower here.
    higher = {}
    for suit in (*range(6), 7):
        for tile in tiles_of(rules.follow_bits[suit]):
            terms = {OFFSET['trick'] + slot * 28 + t: -1
                     for slot in range(3) for t in range(28)
                     if rules.rank[suit][t] > rules.rank[suit][tile]}
            higher[suit, tile] = one(f'no_higher_suit_{suit}_tile_{tile}', terms, 1)

    opponent = one('opponent_winning', {OFFSET['winning']: 1, OFFSET['winning'] + 2: 1})
    lead = one('on_lead', {OFFSET['open']: 1})
    safe_partner = one('last_seat_partner_winning',
                       {OFFSET['open'] + 3: 1, OFFSET['winning'] + 1: 1}, -1)
    bidder_lead = one('bidding_side_on_lead', {OFFSET['side']: 1, OFFSET['open']: 1}, -1)

    # Count is NOT invariant under the pip relabeling: read the count feature.
    count_slots = sorted({CANON[d][t] for d in range(7) for t in range(28)
                          if rules_for(d).count[t]})
    count = {t: one(f'count_div_5_tile_{t}',
                    {OFFSET['count'] + t: 2, OFFSET['count'] + 28 + t: 1})
             for t in count_slots}
    masters = {}
    for tile in range(28):
        suit = rules.led_suit[tile]
        above = [t for t in tiles_of(rules.follow_bits[suit])
                 if rules.rank[suit][t] > rules.rank[suit][tile]]
        terms = {OFFSET[block] + t: 1 for block in ('hand', 'gone') for t in above}
        masters[tile] = one(f'no_higher_outside_own_hand_tile_{tile}', terms, 1 - len(above))

    for (suit, tile), unit in higher.items():
        terms = {unit: 1, opponent: 1}
        if suit == 7:
            bias = -1
        else:
            terms.update({i: 1 for i in led[suit]})
            bias = -2
        two(f'take_from_opponent_suit_{suit}_tile_{tile}', terms, bias, {tile: 256})
    for tile, unit in count.items():
        two(f'preserve_count_tile_{tile}', {unit: 1}, 0, {tile: -4})
        two(f'feed_safe_partner_tile_{tile}', {unit: 1, safe_partner: 2}, -2, {tile: 64})
    for tile, unit in masters.items():
        two(f'lead_suit_master_tile_{tile}', {unit: 1, lead: 1}, -1, {tile: 64})
    two('pull_trump_when_bidder_leads', {bidder_lead: 1}, 0, {t: 32 for t in range(21, 28)})

    h1, h2 = len(first), len(second)
    w1, b1 = np.zeros((343, h1), dtype=np.int64), np.zeros(h1, dtype=np.int64)
    w2, b2 = np.zeros((h2, h1), dtype=np.int64), np.zeros(h2, dtype=np.int64)
    w3, b3 = np.zeros((28, h2), dtype=np.int64), np.zeros(28, dtype=np.int64)
    for i, (_, terms, bias) in enumerate(first):
        b1[i] = bias
        for j, weight in terms.items():
            w1[j, i] = weight
    for i, (_, terms, bias, outputs) in enumerate(second):
        b2[i] = bias
        for j, weight in terms.items():
            w2[i, j] = weight
        for tile, weight in outputs.items():
            w3[tile, i] = weight
    for a in range(7):
        for b in range(a + 1):
            b3[a * (a + 1) // 2 + b] = -(a + b + 12 * (a == b) + 20 * (a == 6))
    net = FieldNet(w1, b1, w2, b2, w3, b3, 0, 0, 'field-v2')
    arrays = (w1, b1, w2, b2, w3, b3)
    manifest = {
        'name': 'Bourne-0', 'version': 0, 'layout': net.layout,
        'architecture': [343, h1, h2, 28],
        'parameters': sum(a.size for a in arrays),
        'nonzero_parameters': sum(int(np.count_nonzero(a)) for a in arrays),
        'hidden_1': [n for n, *_ in first], 'hidden_2': [n for n, *_ in second],
        'construction': 'Deterministic handwritten weights; zero initialization; no training or checkpoint input.',
        'context': 'Author previously examined the trained Zeb field. This is not a blinded replication.',
        'scope': 'Pip trumps only; existing feature encoder and legal-action mask; no bidder or lookahead.',
        'evaluation_plan': {
            'seed': 20261102, 'pairs_per_opponent': 1008,
            'opponents': ['uniform-random-legal', 'trained-field'],
            'bid': 30, 'declarations': 'cycle 0..6', 'bidder': 'cycle seats 0..3',
            'pair': 'same physical deal and bidder, written field on each team once',
            'interval': '95% percentile bootstrap of deal pairs, 4000 resamples',
            'adaptation': 'No weight changes after the first behavioral evaluation.',
        },
    }
    return net, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    net, manifest = build()
    blob = net.to_bytes()
    manifest.update(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                    frozen_utc=datetime.now(timezone.utc).isoformat(),
                    writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a frozen specimen or its timestamp.
    with (args.out / 'bourne-0.fnet').open('xb') as f:
        f.write(blob)
    with (args.out / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in manifest.items() if not k.startswith('hidden_')}, indent=2))


if __name__ == '__main__':
    main()

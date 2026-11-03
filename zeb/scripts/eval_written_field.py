# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""Verify and play a frozen Bourne-0 blob; never update any weights.

python -u -m zeb.scripts.eval_written_field OUTDIR
The fixed protocol is recorded by written_field before this script runs.
This measures standalone field policies, not the strength of complete Zeb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from scenario.compiled import CompiledWalt
from scenario.field import CANON, FieldNet, features
from scenario.planner import Config, Rung
from scenario.record import empty_record
from scenario.rules import mask_of, rules_for, tiles_of


def symbolic_scores(rules, own, rec, bidder_team):
    """An independent direct policy referee, given only the actor's information."""
    legal = tiles_of(rules.legal(own, rec.led))
    ranks, best, winning_team = None, None, None
    if rec.trick:
        ranks = rules.rank[rules.led_suit[rec.led]]
        best = max(range(len(rec.trick)), key=lambda i: ranks[rec.trick[i]])
        winning_team = (rec.leader + best) % 2
    scores = {}
    for tile in legal:
        canonical = CANON[rules.decl_id][tile]
        high = (math.isqrt(8 * canonical + 1) - 1) // 2
        low = canonical - high * (high + 1) // 2
        score = -(high + low + 12 * (high == low) + 20 * (high == 6))
        count = rules.count[tile] // 5
        score -= 4 * count
        if len(rec.trick) == 3 and winning_team == rec.seat % 2:
            score += 64 * count
        if winning_team is not None and winning_team != rec.seat % 2:
            if ranks[tile] > ranks[rec.trick[best]]:
                score += 256
        if not rec.trick:
            suit = rules.led_suit[tile]
            higher = mask_of(t for t in tiles_of(rules.follow_bits[suit])
                             if rules.rank[suit][t] > rules.rank[suit][tile])
            if not (higher & ~(own | rec.played)):
                score += 64
            if rec.seat % 2 == bidder_team and suit == 7:
                score += 32
        scores[tile] = score
    return scores


def deal(seed):
    deck = list(range(28))
    random.Random(seed).shuffle(deck)
    return [mask_of(deck[7 * seat:7 * (seat + 1)]) for seat in range(4)]


def verify(net, seed):
    totals = dict(positions=0, legal_scores=0, last_seat_can_take=0,
                  last_seat_safe_feed=0, bid_30_42_invariant=0)
    reqs, expected, smoke, witnesses = [], [], [], {}
    for game in range(364):
        decl, bidder, bid = game % 7, game % 4, 30 + game % 13
        rules, hands = rules_for(decl), deal(seed + 100000 + game)
        rec, history = empty_record(bidder), []
        rng = random.Random(seed + 200000 + game)
        for step in range(28):
            seat, own = rec.seat, hands[rec.seat]
            active = features(rules, seat, own, rec, bid, bidder % 2, net.layout)
            z = net.scores(active)
            referee = symbolic_scores(rules, own, rec, bidder % 2)
            assert {t: int(z[CANON[decl][t]]) for t in referee} == referee, (game, step)
            chosen = max(referee, key=lambda t: (referee[t], -t))
            assert chosen == net.move(rules, seat, own, rec, bid, bidder % 2)
            a = net.scores(features(rules, seat, own, rec, 30, bidder % 2, net.layout))
            b = net.scores(features(rules, seat, own, rec, 42, bidder % 2, net.layout))
            assert np.array_equal(a, b)
            totals['bid_30_42_invariant'] += 1
            totals['positions'] += 1
            totals['legal_scores'] += len(referee)
            req = (decl, seat, own, rec.played, rec.leader, list(rec.trick),
                   rec.banked, rec.forbid, bid, bidder, list(history))
            if totals['positions'] % 17 == 0:
                reqs.append(req)
                expected.append((chosen, z.tolist()))
            if step == 20 and len(referee) > 1 and len(smoke) < 28:
                smoke.append(req)
            if len(rec.trick) == 3:
                ranks = rules.rank[rules.led_suit[rec.led]]
                best = max(range(3), key=lambda i: ranks[rec.trick[i]])
                winning_team = (rec.leader + best) % 2
                can_take = [t for t in referee if ranks[t] > ranks[rec.trick[best]]]
                if winning_team != seat % 2 and can_take:
                    assert chosen in can_take
                    totals['last_seat_can_take'] += 1
                    kind = 'take_from_opponent'
                elif winning_team == seat % 2:
                    assert rules.count[chosen] == max(rules.count[t] for t in referee)
                    totals['last_seat_safe_feed'] += 1
                    kind = 'feed_safe_partner'
                else:
                    kind = None
                if kind and kind not in witnesses and len(referee) > 1:
                    witnesses[kind] = dict(request=req, legal_scores=referee, chosen=chosen)
            # Both random and on-policy public records, with separate fixed randomness.
            tile = chosen if game % 2 else rng.choice(list(referee))
            hands[seat] ^= 1 << tile
            history.append((seat, tile))
            rec = rec.advance(tile, rules)
        assert rec.finished and sum(rec.banked) == 42 and not any(hands)

    compiled = CompiledWalt(Config((Rung(32, None, 16),), field=net, tie='margin'), threads=1)
    actual = compiled._core.field_scores(reqs)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got[0] == want[0] and list(got[1]) == want[1]
    planned = compiled._core.solve(smoke, [None] * len(smoke), 1)
    assert len(planned) == len(smoke)
    for req, result in zip(smoke, planned, strict=True):
        legal = rules_for(req[0]).legal(req[2], req[5][0] if req[5] else None)
        assert result[0] in tiles_of(legal)
    totals.update(native_exact_positions=len(actual), planner_legal_positions=len(planned),
                  planner_nodes=sum(r[2] for r in planned), witnesses=witnesses)
    return totals


def play(hands, decl, bidder, bid, policies, random_seed):
    """The environment owns all hands; each policy receives only its own and public record."""
    hands, rules, rec = list(hands), rules_for(decl), empty_record(bidder)
    rng, history = random.Random(random_seed), []
    for _ in range(28):
        seat, own = rec.seat, hands[rec.seat]
        legal = tiles_of(rules.legal(own, rec.led))
        policy = policies[seat % 2]
        tile = rng.choice(legal) if policy is None else policy.move(rules, seat, own, rec, bid, bidder % 2)
        assert tile in legal
        history.append(tile)
        hands[seat] ^= 1 << tile
        rec = rec.advance(tile, rules)
    assert rec.finished and sum(rec.banked) == 42 and not any(hands)
    winner = bidder % 2 if rec.banked[bidder % 2] >= bid else 1 - bidder % 2
    return winner, rec.banked, history


def evaluate(net, trained, protocol, out):
    seed, pairs, bid = protocol['seed'], protocol['pairs_per_opponent'], protocol['bid']
    summary = {}
    with (out / 'games.jsonl').open('x') as ledger:
        for label, opponent in [('uniform-random-legal', None), ('trained-field', trained)]:
            values, point_diffs = [], []
            bid_wins, defend_wins = 0, 0
            for i in range(pairs):
                hands, decl, bidder = deal(seed + i), i % 7, i % 4
                wins, diffs = [], []
                for team in (0, 1):
                    policies = [opponent, opponent]
                    policies[team] = net
                    winner, points, history = play(hands, decl, bidder, bid, policies, seed + 10000 + i)
                    won = int(winner == team)
                    wins.append(won)
                    diffs.append(points[team] - points[1 - team])
                    if team == bidder % 2:
                        bid_wins += won
                    else:
                        defend_wins += won
                    ledger.write(json.dumps(dict(opponent=label, pair=i, hands=hands, decl=decl,
                                                 bidder=bidder, bid=bid, candidate_team=team,
                                                 random_seed=seed + 10000 + i, winner=winner,
                                                 points=points, history=history)) + '\n')
                values.append(sum(wins) / 2)
                point_diffs.append(sum(diffs) / 2)
                if (i + 1) % 252 == 0:
                    print(f'{label}: {i + 1}/{pairs} deal pairs', flush=True)
            rng = np.random.default_rng(seed + 300000)
            resamples = rng.integers(0, pairs, size=(4000, pairs))
            interval = np.quantile(np.asarray(values)[resamples].mean(axis=1), [.025, .975])
            summary[label] = dict(pairs=pairs, hands=2 * pairs, wins=bid_wins + defend_wins,
                                  win_fraction=float(np.mean(values)), ci95=interval.tolist(),
                                  bidder_wins=bid_wins, defender_wins=defend_wins,
                                  mean_point_difference=float(np.mean(point_diffs)))
    # Same-policy pair controls must split wins exactly, including the random tape.
    controls = {}
    for name, policy in [('written', net), ('trained', trained), ('random', None)]:
        wins = 0
        for i in range(56):
            for team in (0, 1):
                winner, _, _ = play(deal(seed + i), i % 7, i % 4, bid,
                                    [policy, policy], seed + 10000 + i)
                wins += winner == team
        assert wins == 56
        controls[name] = dict(wins=wins, hands=112, win_fraction=.5)
    return dict(opponents=summary, same_policy_controls=controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--trained', type=Path, default=Path('zeb/net/margin_d250k_h128x64.fnet'))
    args = parser.parse_args()
    manifest = json.loads((args.out / 'manifest.json').read_text())
    original = (args.out / 'bourne-0.fnet').read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    assert digest == manifest['sha256']
    net = FieldNet.from_bytes(original)
    assert net.to_bytes() == original
    verification = verify(net, manifest['evaluation_plan']['seed'])
    print(json.dumps({'verification': verification}, indent=2), flush=True)
    trained_blob = args.trained.read_bytes()  # Only evaluation loads a trained checkpoint.
    results = evaluate(net, FieldNet.from_bytes(trained_blob), manifest['evaluation_plan'], args.out)
    assert net.to_bytes() == original == (args.out / 'bourne-0.fnet').read_bytes()
    results.update(verification=verification, candidate_sha256=digest,
                   trained_sha256=hashlib.sha256(trained_blob).hexdigest(),
                   evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (args.out / 'results.json').open('x') as f:
        json.dump(results, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in results.items() if k != 'verification'}, indent=2))


if __name__ == '__main__':
    main()

# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Freeze replayed games as clean static questions plus a separate audit sidecar.

A recorded win labels its recorded continuation, never an unplayed action.
The schedule fixes the initial deals/contracts, not a new policy's outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np

from arena.auction import run_auction, score_hand, settle_points
from arena.bidders import Bid30Bidder
from arena.engine import GameRecord, HandRecord, hand_seed
from forge.oracle.rng import deal_from_seed
from scenario.record import empty_record
from scenario.rules import mask_of, rules_for, tiles_of
from scenario_core import SilasFrame
from ..data import SEAT, VERSION, encode
from ..rlvr.verifier import MODEL_ID, solve

SCHEMA = 'gus-frozen-public-regression-v1'
SOURCE_FILES = ('arena/engine.py', 'arena/auction.py', 'arena/bidders.py',
                'arena/hand_metrics.py', 'arena/walt_play.py', 'scenario/rules.py',
                'scenario/record.py', 'scenario/core/src/physics.rs',
                'lem/observer/data.py', 'lem/observer/model.py', 'lem/observer/play.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as stream:
            for row in rows:
                stream.write((canonical(row) + '\n').encode())


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def deal_key(hands):
    """Physical seat allocation, matching the observer's split/exclusion unit."""
    return identity([mask_of(hand) for hand in hands])


def schedule(seed):
    """All thirteen possible one-mark hands, irrespective of when play stops."""
    bidder_policy = Bid30Bidder()
    result = []
    for index in range(13):
        seed_hand = hand_seed(seed, 0, index, 0)
        hands = tuple(tuple(h) for h in deal_from_seed(seed_hand))
        dealer = index % 4
        auction = run_auction(hands, dealer, (bidder_policy,) * 4,
                              random.Random(hash((seed_hand, 0xA0C7))),
                              marks=(0, 0), marks_to_win=7)
        if auction is None or auction.high_bid != 30 or auction.forced:
            raise ValueError('fixed bidder schedule changed')
        result.append({'hand_idx': index, 'seed': seed_hand, 'dealer': dealer,
                       'hands': [list(h) for h in hands], 'deal_key': deal_key(hands),
                       'bidder': auction.winner, 'bid': auction.high_bid,
                       'decl': auction.decl_id, 'bids': list(auction.bids)})
    return result


def decode_games(path):
    result = []
    for raw in read_rows(path):
        raw['hands'] = tuple(HandRecord(**{**h, 'hands': tuple(tuple(x) for x in h['hands']),
                            'plays': tuple(tuple(x) for x in h['plays']),
                            **{key: tuple(h[key]) for key in ('bids','team_points','marks_delta',
                               'marks_after','points_delta','points_after')}}) for h in raw['hands'])
        for key in ('marks', 'points'):
            raw[key] = tuple(raw[key])
        result.append(GameRecord(**raw))
    return result


def verify_games(games, frozen, excluded):
    """Replay legality, scores, game stopping, and shared schedule independently."""
    if len(games) != 2 or {g.a_team for g in games} != {0, 1}:
        raise ValueError('one complete duplicate pair is required')
    seen = set()
    for game in games:
        if game.game_idx != 0 or not 7 <= len(game.hands) <= 13:
            raise ValueError('unsupported game or incomplete first-to-seven record')
        marks, points = [0, 0], [0, 0]
        for index, hand in enumerate(game.hands):
            expected = frozen[index]
            masks = tuple(mask_of(h) for h in hand.hands)
            if masks in excluded:
                raise ValueError('benchmark deal overlaps training/reserved exclusions')
            seen.add(masks)
            for actual, want in ((hand.hand_idx,index), (hand.seed,expected['seed']),
                    (hand.dealer,expected['dealer']), (hand.bidder,expected['bidder']),
                    (hand.bid_value,expected['bid']), (hand.decl_id,expected['decl']),
                    (hand.hands,tuple(tuple(h) for h in expected['hands'])),
                    (hand.bids,tuple(expected['bids'])), (hand.a_team,game.a_team),
                    (hand.redeals,0), (hand.forced,False)):
                if actual != want:
                    raise ValueError('frozen schedule mismatch')
            rules, rec = rules_for(hand.decl_id), empty_record(hand.bidder)
            if len(hand.plays) != 28:
                raise ValueError('hand has incomplete plays')
            for seat,tile in hand.plays:
                legal = rules.legal(masks[seat] & ~rec.played, rec.led)
                if seat != rec.seat or not legal & (1 << tile):
                    raise ValueError('illegal recorded action')
                rec = rec.advance(tile,rules)
            score = score_hand(hand.bid_value, hand.bidder % 2, rec.banked)
            delta = settle_points(hand.bid_value, hand.bidder % 2, rec.banked)
            marks = [marks[t] + score.marks[t] for t in (0,1)]
            points = [points[t] + delta[t] for t in (0,1)]
            if (rec.banked != hand.team_points or score.made != hand.made or
                score.marks != hand.marks_delta or tuple(marks) != hand.marks_after or
                delta != hand.points_delta or tuple(points) != hand.points_after):
                raise ValueError('replayed outcome differs')
            if max(marks) >= 7 and index != len(game.hands)-1:
                raise ValueError('game continued after reaching seven marks')
        winner = marks.index(7)
        if tuple(marks) != game.marks or tuple(points) != game.points or winner != game.winner_team:
            raise ValueError('game outcome differs')
    return seen


def question_tokens(question):
    if question['bid'] != 30:
        raise ValueError('Gus only accepts bid 30')
    return [*map(int, encode(mask_of(question['own']), question['viewer'],
              question['bidder'], question['decl'], question['plays'])), SEAT]


def pair_rows(pair_id, panel, games, manifest):
    result=[]
    policies={role: identity({k:v for k,v in manifest[role].items() if k!='path'})
              for role in ('candidate','opponent')}
    for game in sorted(games,key=lambda g:g.a_team):
        game_id=f'{pair_id}/side-{game.a_team}'
        for hand in game.hands:
            hand_id=f'{game_id}/hand-{hand.hand_idx:02d}'
            masks=tuple(mask_of(h) for h in hand.hands)
            rules, rec=rules_for(hand.decl_id),empty_record(hand.bidder)
            for ply,(seat,tile) in enumerate(hand.plays):
                question={'own':list(hand.hands[seat]),'viewer':seat,'bidder':hand.bidder,
                          'bid':hand.bid_value,'decl':hand.decl_id,
                          'plays':[list(p) for p in hand.plays[:ply]]}
                legal=tiles_of(rules.legal(masks[seat] & ~rec.played,rec.led))
                frame=SilasFrame(hand.decl_id,seat,masks[seat]&~rec.played,
                                  hand.bidder,hand.bid_value,[tuple(p) for p in question['plays']])
                if (tuple(frame.banked),frame.leader,tuple(frame.trick)) != (rec.banked,rec.leader,rec.trick):
                    raise ValueError('Silas/public referee divergence')
                result.append({'id':f'{hand_id}/ply-{ply:02d}','panel':panel,'pair_id':pair_id,
                    'game_id':game_id,'hand_id':hand_id,'deal_key':deal_key(hand.hands),'ply':ply,
                    'question':question,'tokens':question_tokens(question),'legal':legal,
                    'reference_action':tile,'behavior_id':policies['candidate' if seat%2==game.a_team else 'opponent'],
                    'physics_identity':frame.physics_identity,
                    'public':{'banked':list(rec.banked),'leader':rec.leader,'trick':list(rec.trick),
                        'remaining_count':[rec.hand_size(s) for s in range(4)],
                        'forbid':list(rec.forbid),'settled':rec.banked[hand.bidder%2]>=30 or rec.banked[1-hand.bidder%2]>=13},
                    'outcome':{'bidder_made':hand.made,'actor_team_won_hand':bool(hand.marks_delta[seat%2]),
                        'actor_team_won_game':game.winner_team==seat%2,'team_points':list(hand.team_points),
                        'marks_delta':list(hand.marks_delta),'game_marks':list(game.marks),
                        'scope':'recorded trajectory only; not a counterfactual action value'}})
                rec=rec.advance(tile,rules)
    return result


def freeze_runtime_pins(root):
    """Record runtime source and version pins separately from the row manifest."""
    import importlib.metadata
    import platform
    import sys
    import torch
    roots=['scenario','silas','arena','forge/zeb','forge/oracle']
    paths=subprocess.check_output(['git','ls-files','--',*roots],text=True).splitlines()
    selected=[]
    for name in paths:
        path=Path(name)
        if path.suffix not in ('.py','.rs','.toml','.lock'):continue
        if any(part in ('tests','examples','benches') for part in path.parts) or path.name.startswith('test_'):continue
        selected.append(name)
    selected+=['lem/observer/__init__.py','lem/observer/model.py','lem/observer/data.py',
        'lem/observer/play.py','lem/observer/splat/__init__.py','lem/observer/splat/model.py','lem/observer/splat/play.py']
    native=[]
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and 'scenario_core' in path and Path(path).suffix in ('.so','.dylib','.pyd'):
            native.append({'file':Path(path).name,'sha256':digest(path)})
    pins={'schema':'gus-frozen-runtime-pins-v1',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_files':{name:digest(name) for name in sorted(set(selected))},
        'source_roots':roots,
        'source_scope':'tracked Python/Rust/Cargo runtime files; excludes tests/examples/benches; model and adapters explicitly listed',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
            'scenario_core':importlib.metadata.version('scenario_core')},
        'platform':{'system':platform.system(),'machine':platform.machine()},'native_binary_audit':native,
        'native_note':'Native binary bytes can change on rebuild with embedded paths; source pins and replay parity govern portability.',
        'policy_note':'Original G0/Splat model and clean encoder fixed; future checkpoint weights vary. New architecture/adapter requires a new declared protocol.'}
    write_json(Path(root)/'runtime-pins.json',pins)
    return pins


def attach_assets(root, checkpoint, field):
    """Package the pinned reference weights without changing any frozen row."""
    root=Path(root); manifest=json.loads((root/'manifest.json').read_text())
    checkpoint_sha,field_sha=digest(checkpoint),digest(field)
    identities=[p[role] for p in manifest['pairs'] for role in ('candidate','opponent')]
    if not any(p.get('sha256')==checkpoint_sha for p in identities):
        raise ValueError('reference checkpoint is absent from frozen participants')
    if not any(p.get('field_sha256')==field_sha for p in identities):
        raise ValueError('reference field is absent from frozen participants')
    assets=root/'assets';assets.mkdir(exist_ok=True)
    for source,name in ((checkpoint,'g0-full.pt'),(field,'zeb-field.npz')):
        destination=assets/name
        if destination.exists() and digest(destination)!=digest(source):
            raise ValueError('frozen asset differs')
        shutil.copyfile(source,destination)
    manifest['assets']={
        'g0':{'path':'assets/g0-full.pt','sha256':checkpoint_sha},
        'zeb_field':{'path':'assets/zeb-field.npz','sha256':field_sha}}
    write_json(root/'manifest.json',manifest)
    return manifest


def load(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema']!=SCHEMA or manifest['tokenizer']!=VERSION:
        raise ValueError('benchmark schema/tokenizer mismatch')
    if digest(root/'rows.jsonl.gz') != manifest['rows_sha256']:
        raise ValueError('benchmark rows checksum mismatch')
    for asset in manifest.get('assets',{}).values():
        if digest(root/asset['path']) != asset['sha256']:
            raise ValueError('benchmark reference asset checksum mismatch')
    rows=read_rows(root/'rows.jsonl.gz')
    if len(rows)!=manifest['rows'] or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('benchmark count or unique IDs disagree')
    for row in rows:
        if row['tokens']!=question_tokens(row['question']) or row['reference_action'] not in row['legal']:
            raise ValueError('benchmark clean inputs or reference action disagree')
    return manifest,rows


def build(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    excluded={tuple(map(int,r)) for r in np.load(args.exclude,allow_pickle=False)}
    rows,pairs,seen,reserved=[],[],set(),set()
    for specification in sorted(args.source):
        name,path=specification.split('=',1); source=Path(path)
        if not name.replace('-','').replace('_','').isalnum():
            raise ValueError('simple unique source names required')
        if name in {p['id'] for p in pairs}:raise ValueError('duplicate pair id')
        manifest=json.loads((source/'manifest.json').read_text())
        frozen=schedule(manifest['seed']); games=decode_games(source/'games.jsonl.gz')
        reserve={tuple(mask_of(h) for h in scheduled['hands']) for scheduled in frozen}
        if reserve & excluded:raise ValueError('potential future hand overlaps training/reserved exclusions')
        reserved.update(reserve)
        seen.update(verify_games(games,frozen,excluded))
        panel='anchor-v1' if name.startswith('anchor-') else 'reused-regression-v1'
        rows.extend(pair_rows(name,panel,games,manifest))
        dest=out/'audit'/name;dest.mkdir(parents=True)
        for filename in ('manifest.json','games.jsonl.gz'):
            shutil.copyfile(source/filename,dest/filename)
        write_json(dest/'schedule.json',{'seed':manifest['seed'],'hands':frozen,
            'meaning':'13 possible initial deals/contracts shared across both sides and future challengers'})
        pairs.append({'id':name,'panel':panel,'seed':manifest['seed'],
            'candidate':manifest['candidate'],'opponent':manifest['opponent'],
            'audit':str(dest.relative_to(out)),'source_commit':manifest.get('source_commit',manifest.get('git_commit')),
            'games_sha256':digest(dest/'games.jsonl.gz'),'source_manifest_sha256':digest(dest/'manifest.json'),
            'schedule_sha256':digest(dest/'schedule.json'),'known_candidate_game_wins':sum(g.a_won for g in games),
            'games':len(games),'hands':sum(len(g.hands) for g in games)})
    exact_cache={}
    for row in rows:
        if 20 <= row['ply'] <= 23 and len(row['legal']) == 2 and not row['public']['settled']:
            key=identity(row['question'])
            if key not in exact_cache:
                exact_cache[key]=solve(row['question'])
            row['exact_certificate']=exact_cache[key]
    write_rows(out/'rows.jsonl.gz',rows)
    np.save(out/'reserved-hands.npy',np.asarray(sorted(reserved),dtype=np.uint32))
    manifest={'schema':SCHEMA,'tokenizer':VERSION,'license':'CC0-1.0','panel_id':'gus-frozen-benchmark-2026-10-18-v1',
        'exact_verifier':{'model_id':MODEL_ID,'unique_questions':len(exact_cache),
            'covered_rows':sum('exact_certificate' in r for r in rows),
            'scope':'separate uniform-hidden-world, equal lowest/highest/uniform field approximation; not the actual H2H opponents'},
        'rows':len(rows),'rows_sha256':digest(out/'rows.jsonl.gz'),'physical_deals':len(seen),'pairs':pairs,
        'reserved_deals':len(reserved),'reserved_hands_sha256':digest(out/'reserved-hands.npy'),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_files':{p:digest(p) for p in SOURCE_FILES},
        'exclusion_sha256':digest(args.exclude),'excluded_deals':len(excluded),
        'row_order':'lexicographic source name, candidate absolute team 0 then 1, hand index, ply',
        'information_contract':'only question.own and public contract/history enter tokens; outcomes and audit are labels only',
        'outcome_scope':'recorded trajectories; no outcome is a Q label for an unplayed action',
        'statistical_scope':'reused static regression; correlated within physical deals and pairs; not fresh confirmation',
        'h2h_scope':'seeds/deals/contracts fixed; changed actions change history and outcomes; rerun play to measure wins'}
    write_json(out/'manifest.json',manifest)
    if bool(getattr(args,'reference_checkpoint',None)) != bool(getattr(args,'field',None)):
        raise ValueError('package both reference checkpoint and field together')
    if getattr(args,'reference_checkpoint',None):
        manifest=attach_assets(out,args.reference_checkpoint,args.field)
    freeze_runtime_pins(out)
    load(out)
    print(canonical({'rows':len(rows),'pairs':len(pairs),'physical_deals':len(seen),'rows_sha256':manifest['rows_sha256']}),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',action='append',required=True,help='unique-label=pair-directory')
    parser.add_argument('--exclude',required=True,help='N x 4 original hand bitmasks, NPY')
    parser.add_argument('--out',required=True)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--field')
    build(parser.parse_args())

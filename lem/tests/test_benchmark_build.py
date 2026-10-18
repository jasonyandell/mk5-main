# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Frozen fixtures preserve complete games, lawful inputs and scoped labels."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from arena.bidders import Bid30Bidder
from arena.engine import ArenaConfig
from arena.match import run_match
from forge.zeb.game import current_player, legal_actions
from lem.observer.benchmark.build import (build, decode_games, digest, load, mask_of,
    pair_rows, schedule, verify_games, write_json, write_rows)


class Lowest:
    def plays(self,decl):return decl in range(7)
    def choose(self,states,bid_values,marks=None,marks_to_win=7):
        return [min(legal_actions(s),key=lambda i:s.hands[current_player(s)][i]) for s in states]


@pytest.fixture(scope='module')
def recorded():
    policy=Lowest()
    result=run_match(bid_a=Bid30Bidder(),bid_b=Bid30Bidder(),play_a=policy,play_b=policy,
        n_games=2,cfg=ArenaConfig(base_seed=90123),fast_batching=True,verbose=False)
    return result.games


def test_shared_fixed_schedule_and_complete_replay(recorded):
    frozen=schedule(90123)
    assert len(frozen)==13
    seen=verify_games(recorded,frozen,set())
    assert len(seen)==len(recorded[0].hands)
    assert [h.seed for h in recorded[0].hands]==[h.seed for h in recorded[1].hands]
    for left,right in zip(recorded[0].hands,recorded[1].hands):
        assert left.plays==right.plays and left.team_points==right.team_points
    with pytest.raises(ValueError,match='overlap'):
        verify_games(recorded,frozen,seen)
    with pytest.raises(ValueError,match='schedule mismatch'):
        verify_games(recorded,schedule(90124),set())


def test_bad_scores_and_partial_games_cannot_be_frozen(recorded):
    changed=replace(recorded[0],hands=(replace(recorded[0].hands[0],team_points=(42,0)),*recorded[0].hands[1:]))
    if recorded[0].hands[0].team_points==(42,0):
        changed=replace(recorded[0],hands=(replace(recorded[0].hands[0],team_points=(0,42)),*recorded[0].hands[1:]))
    with pytest.raises(ValueError,match='outcome differs'):verify_games([changed,recorded[1]],schedule(90123),set())
    with pytest.raises(ValueError,match='complete duplicate'):verify_games([recorded[0]],schedule(90123),set())


def test_clean_questions_and_outcome_scope(recorded):
    manifest={'candidate':{'name':'lowest'},'opponent':{'name':'lowest'}}
    rows=pair_rows('test','reused-regression-v1',recorded,manifest)
    assert len(rows)==sum(len(g.hands)*28 for g in recorded)
    for row in rows:
        assert set(row['question'])=={'own','viewer','bidder','bid','decl','plays'}
        assert len(row['question']['own'])==7
        assert len(row['question']['plays'])==row['ply']
        assert row['reference_action'] in row['legal']
        assert len(row['tokens'])==14+2*row['ply']
        assert 'not a counterfactual' in row['outcome']['scope']
        assert 'hands' not in row and 'hands' not in row['question']
    altered=deepcopy(manifest);altered['candidate']['path']='some/local/cache'
    assert pair_rows('test','reused-regression-v1',recorded,altered)==rows


def test_archive_is_deterministic_and_tamper_detected(tmp_path,recorded):
    source=tmp_path/'source';source.mkdir()
    write_rows(source/'games.jsonl.gz',[asdict(g) for g in recorded])
    assert decode_games(source/'games.jsonl.gz')==recorded
    write_json(source/'manifest.json',{'seed':90123,'candidate':{'name':'lowest'},'opponent':{'name':'lowest'}})
    np.save(tmp_path/'excluded.npy',np.empty((0,4),dtype=np.uint32))
    for name in ('one','two'):
        build(SimpleNamespace(out=tmp_path/name,exclude=tmp_path/'excluded.npy',source=[f'test={source}']))
    assert digest(tmp_path/'one/rows.jsonl.gz')==digest(tmp_path/'two/rows.jsonl.gz')
    manifest,rows=load(tmp_path/'one')
    assert manifest['exact_verifier']['covered_rows']>0
    for row in rows:
        if 'exact_certificate' in row:
            assert row['exact_certificate']['authority'].startswith('exact_given')
    rows[0]['tokens'][0]=0
    write_rows(tmp_path/'one/rows.jsonl.gz',rows)
    with pytest.raises(ValueError,match='checksum'):load(tmp_path/'one')
    manifest['rows_sha256']=digest(tmp_path/'one/rows.jsonl.gz')
    write_json(tmp_path/'one/manifest.json',manifest)
    with pytest.raises(ValueError,match='clean inputs'):load(tmp_path/'one')


def test_reference_assets_are_pinned_and_checked(tmp_path):
    from lem.observer.benchmark.build import attach_assets
    root=tmp_path/'panel';root.mkdir()
    checkpoint=tmp_path/'g0.pt';checkpoint.write_bytes(b'checkpoint receipt')
    field=tmp_path/'field.npz';field.write_bytes(b'field receipt')
    write_json(root/'manifest.json',{'pairs':[{'candidate':{'sha256':digest(checkpoint)},
        'opponent':{'field_sha256':digest(field)}}]})
    manifest=attach_assets(root,checkpoint,field)
    assert digest(root/manifest['assets']['g0']['path'])==digest(checkpoint)
    field.write_bytes(b'changed field')
    with pytest.raises(ValueError,match='absent from frozen'):attach_assets(root,checkpoint,field)

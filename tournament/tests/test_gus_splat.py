# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
# STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
"""Joint models keep clean inputs and explicit heads in the common-bidder ladder."""
from dataclasses import asdict, replace
from hashlib import sha256

import pytest
import torch

from lem.observer.data import VERSION
from lem.observer.model import Config
from lem.observer.splat.model import SplatObserver, TASK
from lem.observer.splat.play import SplatPlay
from lem.tests.test_observer import recording
from lem.tests.test_observer_play import state_from_record
from tournament import faceoff, seats
from tournament.demon_bid30 import participant as demon_bid30, game_key
from tournament.gus_seat import LADDER
from tournament.gus_splat_seat import make_seat, participant
from tournament.participants import DEMON_BAIT40_REUSE_N100, supports_ladder


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    torch.manual_seed(42); torch.set_num_threads(1)
    config = Config(width=8, layers=1, heads=1)
    path = tmp_path / 'gus.pt'
    torch.save({'config': asdict(config), 'model': SplatObserver(config).state_dict(),
                'tokenizer': VERSION, 'task': TASK, 'step': 0, 'data_sha256': 'fixture'}, path)
    p = participant('fixture', 'a' * 40, sha256(path.read_bytes()).hexdigest(), 'b' * 40)
    monkeypatch.setattr(seats, 'materialize', lambda _: tmp_path)
    return p, tmp_path


@pytest.mark.parametrize('head', ['tile', 'splat'])
def test_joint_seat_parity_visibility_and_contract(frozen, head):
    p, path = frozen; p = replace(p, config={**p.config, 'head': head})
    wrapped = seats.seat(p, 'fixture')
    direct = SplatPlay(path / 'gus.pt', 'cpu', head)
    state = state_from_record(recording()); hands = list(state.hands)
    hands[1], hands[2] = hands[2], hands[1]
    secret = replace(state, hands=tuple(hands))
    assert wrapped.choose([state, secret], [30, 30]) == direct.choose([state, state], [30, 30])
    with pytest.raises(ValueError, match='bid30|bid 30'):
        wrapped.choose([state], [31])
    assert supports_ladder(p.to_record(), LADDER)
    assert not supports_ladder(p.to_record(), 'marks7')


def test_joint_weights_and_all_forward_sources_are_pinned(frozen):
    p, path = frozen
    for source in ('lem/observer/splat/model.py', 'lem/observer/splat/play.py', 'tournament/gus_splat_seat.py'):
        assert source in p.pin['source_files']
    changed = replace(p, pin={**p.pin, 'source_files': {'lem/observer/splat/model.py': '0' * 64}})
    with pytest.raises(RuntimeError, match='source changed'):
        make_seat(changed, path)
    (path / 'gus.pt').write_bytes(b'corruption')
    with pytest.raises(RuntimeError, match='checkpoint differs'):
        make_seat(p, path)


def test_joint_whole_games_repeat_and_use_the_common_auction(frozen):
    p, path = frozen; q = replace(p, id='joint-copy')
    participants = {r.id: r.to_record() for r in (p, q)}
    spec = faceoff.FaceoffSpec(LADDER, p.id, q.id, 712341, 1, '20261018T000000Z')
    games = [faceoff.play_game(spec.__dict__, 0, side, str(path / f'game-{i}.json'), participants)
             for i, side in enumerate((0, 1, 0))]
    assert games[0]['hands'] == games[1]['hands'] == games[2]['hands']
    assert all(h['bid_value'] == 30 and h['decl_id'] in range(7) for h in games[0]['hands'])
    record = faceoff.assemble(spec, participants, games[:2], {}, 'done')
    assert record['summary']['a_wins'] == record['summary']['b_wins'] == 1


def test_demon_common_contract_preserves_pinned_cardplay():
    original = DEMON_BAIT40_REUSE_N100; bounded = demon_bid30('a' * 40)
    assert all(bounded.pin[k] == v for k, v in original.pin.items())
    assert all(bounded.config[k] == v for k, v in original.config.items())
    assert bounded.id != original.id
    assert bounded.config['common_bidder'] == 'Bid30Bidder'
    assert bounded.config['worlds'] == 100 and bounded.config['margin_tiebreak']
    assert supports_ladder(bounded.to_record(), LADDER)
    assert not supports_ladder(original.to_record(), LADDER)
    assert not supports_ladder(bounded.to_record(), 'marks7')


def test_fixed_demon_draws_ignore_candidate_name_and_claim_time():
    first = game_key('date-one-candidate-a--demon-0000000000001234/0/1')
    assert first == game_key('date-two-candidate-b--demon-0000000000001234/0/1')
    assert first != game_key('date-one-candidate-a--demon-0000000000001235/0/1')
    assert first != game_key('date-one-candidate-a--demon-0000000000001234/0/0')
    with pytest.raises(ValueError, match='game identity'):
        game_key('unregistered game')

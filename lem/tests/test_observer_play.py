# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Acting-seat visibility, output legality, arena replay and paired estimates."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from arena.bidders import Bid30Bidder
from arena.engine import ArenaConfig
from arena.match import run_match
from forge.zeb.game import apply_action, current_player, legal_actions
from forge.zeb.types import BidState, GamePhase, ZebGameState
from lem.observer.data import QUERIES, VERSION, prepare
from lem.observer.h2h import paired_summary, verify_replays
from lem.observer.model import Config, Observer
from lem.observer.play import ObserverPlay, prefix
from lem.tests.test_observer import recording
from scenario.rules import tiles_of


def state_from_record(r):
    return ZebGameState(tuple(tuple(tiles_of(h)) for h in r["hands"]), 3, GamePhase.PLAYING,
                        BidState((30, 0, 0, 0), 0, 30), r["decl"], 0,
                        frozenset(), (), (), 0, (0, 0))


@pytest.fixture
def player(tmp_path):
    torch.manual_seed(47)
    torch.set_num_threads(2)
    config = Config(width=8, layers=1, heads=1)
    path = tmp_path / "model.pt"
    torch.save({"config": config.__dict__, "model": Observer(config).state_dict(),
                "tokenizer": VERSION, "step": 0, "data_sha256": "test-fixture"}, path)
    return ObserverPlay(path, "cpu")


@pytest.mark.parametrize("decl", [*range(8), 9])
def test_live_prefix_exactly_matches_training_at_every_turn(decl):
    r = recording(decl=decl)
    data, state = prepare([r]), state_from_record(r)
    for i, (actor, tile) in enumerate(r["plays"]):
        np.testing.assert_array_equal(prefix(state, 30), data["x"][actor, :QUERIES[i] + 1])
        state = apply_action(state, state.hands[actor].index(tile))


def test_hidden_hands_and_match_score_never_affect_choice(player):
    state = state_from_record(recording())
    # These two complete worlds have exactly the same acting-seat information.
    hands = list(state.hands)
    hands[1], hands[2] = hands[2], hands[1]
    other = replace(state, hands=tuple(hands))
    np.testing.assert_array_equal(prefix(state, 30), prefix(other, 30))
    assert player.choose([state, other], [30, 30], [(0, 6), (6, 0)]) == player.choose([state]*2, [30]*2)


def test_actual_contract_and_unsupported_declaration_are_refused(player):
    state = state_from_record(recording())
    with pytest.raises(ValueError, match="bid 30"):
        player.choose([state], [31])
    with pytest.raises(ValueError, match="bid 30"):
        player.choose([replace(state, bid_state=BidState((31, 0, 0, 0), 0, 31))], [30])
    with pytest.raises(ValueError, match="declaration"):
        player.choose([replace(state, decl_id=8)], [30])


def test_choose_masks_outputs_and_breaks_ties_by_tile_not_slot(player):
    state = state_from_record(recording())
    hand = state.hands[0]
    forbidden = next(tile for tile in range(28) if tile not in hand)

    class Scores(torch.nn.Module):
        def forward(self, tokens):
            scores = torch.zeros((*tokens.shape, 28))
            scores[..., forbidden] = 100
            return scores

    player.model = Scores()
    reverse = replace(state, hands=(tuple(reversed(hand)), *state.hands[1:]))
    for s, slot in zip([state, reverse], player.choose([state, reverse], [30, 30])):
        assert s.hands[0][slot] == min(hand)


def test_variable_prefix_batch_matches_single_choice(player):
    state, states = state_from_record(recording()), []
    for _ in range(28):
        states.append(state)
        state = apply_action(state, legal_actions(state)[0])
    singles = [player.choose([s], [30])[0] for s in states]
    assert player.choose(states, [30]*len(states)) == singles


def test_same_player_paired_arena_is_exactly_neutral_and_replays(player):
    def play():
        return run_match(bid_a=Bid30Bidder(), bid_b=Bid30Bidder(), play_a=player, play_b=player,
                         n_games=4, cfg=ArenaConfig(base_seed=71, marks_to_win=1), fast_batching=True)
    a, b = play(), play()
    assert a.games == b.games
    summary = paired_summary(a.games, draws=100)
    assert summary["game_win_rate"] == {"estimate": .5, "ci95": [.5, .5]}
    assert summary["marks_per_game"] == {"estimate": 0., "ci95": [0., 0.]}
    assert verify_replays(a.games, set())["all_plays_legal_and_scores_replayed"]
    hand = a.games[0].hands[0]
    excluded = {tuple(sum(1 << t for t in h) for h in hand.hands)}
    with pytest.raises(ValueError, match="previously recorded"):
        verify_replays(a.games, excluded)
    with pytest.raises(ValueError, match="both team assignments"):
        paired_summary(a.games[:2])
    with pytest.raises(ValueError, match="duplicate"):
        paired_summary(a.games + a.games)

"""Rules tables and the public record agree with the zeb engine, play by play."""
import random

import pytest

from forge.oracle.tables import resolve_trick
from forge.zeb.game import apply_action, current_player, is_terminal, legal_actions, new_game
from forge.zeb.types import GamePhase

from scenario.record import record_from_state, remaining_hand
from scenario.rules import ALL_TILES, mask_of, rules_for, tiles_of

DECLS = [d for d in range(10) if d != 8]  # decl 8 (doubles-suit) is purged from play


def random_playing_states(n_games: int, seed: int):
    """Yield every PLAYING state along random-play games with varied declarations."""
    rng = random.Random(seed)
    for g in range(n_games):
        state = new_game(seed * 1000 + g)
        decl = DECLS[g % len(DECLS)]
        state = state.__class__(**{**state.__dict__, "decl_id": decl})
        assert state.phase == GamePhase.PLAYING
        while not is_terminal(state):
            yield state
            state = apply_action(state, rng.choice(legal_actions(state)))
        yield state


def test_legal_mask_matches_engine():
    for state in random_playing_states(40, seed=1):
        if is_terminal(state):
            continue
        rules = rules_for(state.decl_id)
        seat = current_player(state)
        hand = remaining_hand(state, seat)
        led = state.current_trick[0] if state.current_trick else None
        mine = mask_of(state.hands[seat][s] for s in legal_actions(state))
        assert rules.legal(hand, led) == mine


def test_trick_resolution_matches_engine():
    rng = random.Random(7)
    for decl in DECLS:
        rules = rules_for(decl)
        for _ in range(500):
            trick = tuple(rng.sample(range(28), 4))
            out = resolve_trick(trick[0], trick, decl)
            assert rules.winner_offset(trick) == out.winner_offset
            assert rules.trick_points(trick) == out.points


def test_record_replays_engine_history_and_voids_are_sound():
    for state in random_playing_states(40, seed=3):
        if is_terminal(state):
            continue
        rec = record_from_state(state)  # asserts played/leader/trick/banked internally
        assert rec.seat == current_player(state)
        for seat in range(4):
            held = remaining_hand(state, seat)
            assert rec.hand_size(seat) == bin(held).count("1")
            assert held & rec.forbid[seat] == 0, "void inference contradicted by the real hand"


def test_record_advance_reaches_terminal_with_engine_points():
    rng = random.Random(11)
    for g in range(20):
        state = new_game(5000 + g)
        rules = rules_for(state.decl_id)
        rec = record_from_state(state)
        while not is_terminal(state):
            a = rng.choice(legal_actions(state))
            tile = state.hands[current_player(state)][a]
            state = apply_action(state, a)
            rec = rec.advance(tile, rules)
        assert rec.finished and rec.played == ALL_TILES
        assert rec.banked == tuple(state.team_points)
        assert sum(rec.banked) == 42


def test_tiles_of_roundtrip():
    rng = random.Random(0)
    for _ in range(100):
        tiles = sorted(rng.sample(range(28), rng.randint(0, 28)))
        assert tiles_of(mask_of(tiles)) == tiles

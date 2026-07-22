"""Information-boundary and session tests for the playable Hoyt adapter."""
from __future__ import annotations

from dataclasses import replace

from arena.play import RandomPlay
from forge.zeb.game import apply_action, current_player, legal_actions
from forge.zeb.types import GamePhase
from hoyt.play import (
    HoytPlay,
    SolveBudget,
    TableSession,
    _action_from_text,
    advance_to_horizon,
    format_domino,
    load_session,
    render_position,
    root_from_state,
    save_session,
)


def _h5_state(seed: int = 910001):
    return advance_to_horizon(seed, 5, RandomPlay(123))


def test_root_projection_ignores_referee_hidden_hands():
    state = _h5_state()
    me = current_player(state)
    hidden = [s for s in range(4) if s != me]
    hands = [list(h) for h in state.hands]
    # Swap two still-hidden, unplayed tiles.  The synthetic referee state need
    # not be a reachable prefix: the point of this unit gate is that projection
    # cannot observe either hidden holding at all.
    a = next(d for d in hands[hidden[0]] if d not in state.played)
    b = next(d for d in hands[hidden[1]] if d not in state.played)
    ia = hands[hidden[0]].index(a)
    ib = hands[hidden[1]].index(b)
    hands[hidden[0]][ia], hands[hidden[1]][ib] = b, a
    altered = replace(state, hands=tuple(tuple(h) for h in hands))
    assert root_from_state(state) == root_from_state(altered)


def test_table_session_round_trip_and_render_hide_other_hands(tmp_path):
    state = _h5_state()
    session = TableSession(
        state=state, human_seat=current_player(state), seed=910001,
        start_horizon=5, prefix="random")
    path = tmp_path / "session.json"
    save_session(session, path)
    loaded = load_session(path)
    assert loaded == session

    rendered = render_position(loaded)
    me = loaded.human_seat
    for seat in range(4):
        if seat == me:
            continue
        for tile in loaded.state.hands[seat]:
            if tile not in loaded.state.played:
                assert format_domino(tile) not in rendered


def test_displayed_choice_and_domino_text_resolve_to_same_legal_action():
    state = _h5_state()
    legal = legal_actions(state)
    first = int(legal[0])
    tile = state.hands[current_player(state)][first]
    assert _action_from_text(state, "1") == first
    assert _action_from_text(state, format_domino(tile)) == first
    assert _action_from_text(state, format_domino(tile).replace("-", "")) == first


def test_forced_play_does_not_construct_a_solver():
    state = _h5_state()
    # Advance with legal first actions until a forced position appears.  Late
    # in every hand the acting seat has one tile, so this is deterministic.
    while state.phase == GamePhase.PLAYING and len(legal_actions(state)) != 1:
        state = apply_action(state, legal_actions(state)[0])
    assert state.phase == GamePhase.PLAYING
    player = HoytPlay(
        budgets={h: SolveBudget(0, 1, 1, 2, capacity=1) for h in range(1, 7)})
    got = player.choose([state], [state.bid_state.high_bid])
    assert got == [legal_actions(state)[0]]
    assert player.last_decisions[0].forced
    assert player.last_decisions[0].wall_seconds == 0.0

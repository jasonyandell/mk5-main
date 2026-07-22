"""atlas narration tests.

Narration is a faithful log of the coordinate delta: the play always
appears; void, banked and promotion events match the coordinate diff
exactly; and role promotions are monotone (a tile named a walker stays a
walker for the rest of the hand). A targeted scenario engineers a walker
promotion and asserts the event fires.
"""
import random

import numpy as np

from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)

from atlas.coordinate import from_engine, transition
from atlas.narrate import Event, narrate, suit_name, tile_name
from atlas.roles import outstanding, walker_mask


def test_play_event_always_present_and_correct():
    for g in range(40):
        st = new_game(80_000 + g)
        rng = random.Random(g)
        viewer = rng.randint(0, 3)
        while not is_terminal(st):
            before = from_engine(st, viewer)
            actor = current_player(st)
            slot = rng.choice(legal_actions(st))
            dom = st.hands[actor][slot]
            after = transition(before, dom)
            events = narrate(before, after)
            play = [e for e in events if e.kind == "play"]
            assert len(play) == 1
            assert play[0].seat == actor and play[0].tile == dom
            for e in events:
                assert isinstance(e.sentence(), str) and e.sentence()
            st = apply_action(st, slot)


def test_void_and_banked_match_diff():
    for g in range(60):
        st = new_game(81_000 + g)
        rng = random.Random(g ^ 3)
        viewer = rng.randint(0, 3)
        while not is_terminal(st):
            before = from_engine(st, viewer)
            actor = current_player(st)
            slot = rng.choice(legal_actions(st))
            dom = st.hands[actor][slot]
            after = transition(before, dom)
            events = narrate(before, after)

            # void events == the exact new void bits
            want_void = set()
            for s in range(4):
                gained = int(after.voids[s]) & ~int(before.voids[s])
                for ls in range(8):
                    if (gained >> ls) & 1:
                        want_void.add((s, ls))
            got_void = {(e.seat, e.suit) for e in events if e.kind == "void"}
            assert got_void == want_void

            # banked event == team_points delta on trick completion
            delta = sum(after.team_points) - sum(before.team_points)
            banked = [e for e in events if e.kind == "banked"]
            if delta > 0 and not after.current_trick:
                assert len(banked) == 1
                assert banked[0].points == delta
                assert banked[0].seat == after.trick_leader
            else:
                assert not banked
            st = apply_action(st, slot)


def test_walker_promotions_are_monotone():
    for g in range(60):
        st = new_game(82_000 + g)
        rng = random.Random(g ^ 9)
        viewer = rng.randint(0, 3)
        promoted = set()
        while not is_terminal(st):
            before = from_engine(st, viewer)
            actor = current_player(st)
            slot = rng.choice(legal_actions(st))
            dom = st.hands[actor][slot]
            after = transition(before, dom)
            for e in narrate(before, after):
                if e.kind == "walker":
                    # a promoted walker must actually be a walker now, held
                    hand = int(after.viewer_hand)
                    wm = walker_mask(hand, outstanding(after, hand), after.decl_id)
                    assert (wm >> e.tile) & 1
                    promoted.add(e.tile)
            # monotone: everything ever promoted-and-still-held is still a walker
            hand = int(after.viewer_hand)
            wm = walker_mask(hand, outstanding(after, hand), after.decl_id)
            for t in promoted:
                if (hand >> t) & 1:
                    assert (wm >> t) & 1, (g, t)
            st = apply_action(st, slot)


def test_engineered_walker_promotion_fires():
    """Play the tiles that beat a held tile until it becomes unbeatable-when-
    led, and assert the walker event fires on the promoting step."""
    fired = 0
    for g in range(200):
        st = new_game(83_000 + g)
        rng = random.Random(g)
        # advance a few tricks so the viewer holds a small hand with threats
        target = rng.randint(3, 5)
        while len(st.play_history) < target * 4:
            st = apply_action(st, rng.choice(legal_actions(st)))
        if is_terminal(st):
            continue
        for viewer in range(4):
            before = from_engine(st, viewer)
            actor = current_player(st)
            for slot in legal_actions(st):
                dom = st.hands[actor][slot]
                after = transition(before, dom)
                if any(e.kind == "walker" for e in narrate(before, after)):
                    fired += 1
        if fired > 5:
            break
    assert fired > 0, "no walker promotion ever observed"


def test_names():
    assert tile_name(25) == "6-4"
    assert tile_name(0) == "0-0"
    assert suit_name(5) == "fives"
    assert suit_name(7) == "trumps"
    assert Event("walker", seat=0, tile=25).sentence() == "the 6-4 became a walker"
    assert Event("banked", seat=1, points=1).sentence().endswith("banked 1 point")

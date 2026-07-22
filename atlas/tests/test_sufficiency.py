"""atlas gate C4 — the coordinate-sufficiency spot-gate.

The coordinate throws away play ORDER; the sufficiency claim (against a
non-signaling field) is that order carries no game information.  C4 tests it
rather than assuming it: engineer pairs of DIFFERENT histories that reach an
EQUAL coordinate, and require hoyt's exact best-response value (vs a fixed
lowest-legal rule sigma) to be identical to 1e-9.

Construction: play a random game; find two ADJACENT completed tricks both
led and won by the same seat (leader(i) == leader(i+1) == leader(i+2)).
Swapping their play order keeps every seat's played-set, banked points and
observed voids — hence the coordinate — while changing the history.  The
swap is replayed through the engine, so any illegal swap is simply rejected;
the coordinate address certifies equality before either root is solved.
"""
import random

import numpy as np

from forge.zeb.game import apply_action, current_player, is_terminal, new_game
from walt.contracts import EndgameRoot
import hoyt as K

from atlas.coordinate import from_engine


def _lowest(seat, hand_mask, legal_mask, decl):
    return int(legal_mask & -legal_mask).bit_length() - 1


def _play_full(seed: int):
    """Play a full random game; return (final_state, actions) where actions is
    the domino id played at each step, in order."""
    st = new_game(seed)
    rng = random.Random((seed << 5) ^ 0x2D)
    doms = []
    from forge.zeb.game import legal_actions
    while not is_terminal(st):
        actor = current_player(st)
        slot = rng.choice(legal_actions(st))
        doms.append(st.hands[actor][slot])
        st = apply_action(st, slot)
    return st, doms


def _play_doms(seed: int, doms):
    """Replay a domino-id sequence through a fresh game (same deal). Returns
    the state after the sequence, or None if any play is illegal."""
    st = new_game(seed)
    for dom in doms:
        actor = current_player(st)
        hand = st.hands[actor]
        if dom not in hand:
            return None
        slot = hand.index(dom)
        try:
            st = apply_action(st, slot)
        except ValueError:
            return None
    return st


def _root_from_state(st, me: int) -> EndgameRoot:
    rem = tuple(sorted(d for d in st.hands[me] if d not in st.played))
    return EndgameRoot(
        decl_id=st.decl_id, bidder=st.bidder, bid_value=st.bid_state.high_bid,
        bids=st.bid_state.bids, dealer=st.dealer, me=me, my_hand=rem,
        play_history=st.play_history, trick_leader=st.trick_leader,
        current_trick=st.current_trick, team_points=st.team_points)


def _solve_value(st, me: int, cap: int = 128) -> float:
    root = _root_from_state(st, me)
    from walt.worlds import enumerate_worlds
    worlds = enumerate_worlds(root)
    if len(worlds) > cap:
        idx = np.sort(np.random.default_rng(0).choice(len(worlds), cap, replace=False))
        worlds = worlds[idx]
    tab = K.compile_rule_sigma(root, worlds, _lowest)
    sub = K.build_subgame(root, worlds, np.full(len(worlds), 1.0 / len(worlds)))
    return float(K.br_solve(sub, tab, K.payoff_points(), want_strategy=False).value)


def _leaders(doms, seed: int):
    """Leader seat of each completed trick, replaying to read trick_leader."""
    st = new_game(seed)
    leaders = [st.trick_leader]
    for k, dom in enumerate(doms):
        actor = current_player(st)
        st = apply_action(st, st.hands[actor].index(dom))
        if (k + 1) % 4 == 0 and not is_terminal(st):
            leaders.append(st.trick_leader)
    return leaders


def test_c4_sufficiency_pairs():
    pairs = []
    seed = 60_000
    while len(pairs) < 20 and seed < 61_500:
        seed += 1
        final, doms = _play_full(seed)
        if final.decl_id == 8:      # doubles-suit non-game; walt/hoyt lane skips
            continue
        leaders = _leaders(doms, seed)   # leaders[t] = leader of trick t
        # want i with leader(i)==leader(i+1)==leader(i+2) and the boundary
        # after tricks i,i+1 deep enough to solve cheaply (2..4 tiles left)
        for i in range(1, 4):
            if i + 2 >= len(leaders):
                continue
            if not (leaders[i] == leaders[i + 1] == leaders[i + 2]):
                continue
            remaining = 7 - (i + 2)
            if not (2 <= remaining <= 4):
                continue

            trick_a = doms[4 * i:4 * i + 4]
            trick_b = doms[4 * (i + 1):4 * (i + 1) + 4]
            prefix = doms[:4 * i]
            orig_state = _play_doms(seed, prefix + trick_a + trick_b)
            swap_state = _play_doms(seed, prefix + trick_b + trick_a)
            if orig_state is None or swap_state is None:
                continue
            # different histories...
            if orig_state.play_history == swap_state.play_history:
                continue
            me = leaders[i + 2]     # the seat to lead the next trick
            # ...reaching an equal coordinate (all four viewers)
            if any(from_engine(orig_state, v).address()
                   != from_engine(swap_state, v).address() for v in range(4)):
                continue

            v_orig = _solve_value(orig_state, me)
            v_swap = _solve_value(swap_state, me)
            assert abs(v_orig - v_swap) <= 1e-9, (seed, i, v_orig, v_swap)
            pairs.append((seed, i, remaining, v_orig))
            break

    assert len(pairs) >= 20, f"only engineered {len(pairs)} sufficiency pairs"
    # sanity: the pairs are genuinely varied (not all the same trivial value)
    assert len({round(p[3], 6) for p in pairs}) >= 5

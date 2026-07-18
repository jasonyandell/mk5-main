"""test_worlds.py — brute-force cross-check of enumerate_worlds.

For random endgame positions, enumerate ALL unconstrained assignments of the
unknown tiles into the three non-me seats (with the correct per-seat counts)
and filter them by INDEPENDENTLY replaying the observed history under each
candidate world using the engine's own follow-suit rule (can_follow): every
observed off-suit play must have been forced (the seat held no follower of the
led suit at that moment). Assert set-equality with enumerate_worlds output.

Plain executable script: prints PASS/FAIL per gate, exits nonzero on FAIL.
"""
from __future__ import annotations

import dataclasses
import random
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "/Users/jason/code/mk5-main/.claude/worktrees/walt")

from forge.oracle.tables import can_follow, led_suit_for_lead_domino  # noqa: E402
from forge.zeb import game as zeb  # noqa: E402
from forge.zeb.types import GamePhase  # noqa: E402

from walt.tables import hand_to_mask  # noqa: E402
from walt.worlds import enumerate_worlds, seat_order  # noqa: E402

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  {detail}"
    print(line)
    if not ok:
        _failures.append(name)


def make_root(state, me):
    """Build an EndgameRoot-like object from a zeb PLAYING state for seat me."""
    my_hand = tuple(sorted(d for d in state.hands[me] if d not in state.played))
    return _Root(
        decl_id=state.decl_id,
        bidder=state.bidder,
        bid_value=state.bid_state.high_bid,
        bids=state.bid_state.bids,
        dealer=state.dealer,
        me=me,
        my_hand=my_hand,
        play_history=state.play_history,
        trick_leader=state.trick_leader,
        current_trick=state.current_trick,
        team_points=state.team_points,
    )


@dataclasses.dataclass(frozen=True)
class _Root:
    decl_id: int
    bidder: int
    bid_value: int
    bids: tuple
    dealer: int
    me: int
    my_hand: tuple
    play_history: tuple
    trick_leader: int
    current_trick: tuple
    team_points: tuple


def _position_bits(root):
    """Shared setup: seats, unknown tiles, counts, per-seat known plays."""
    seats = seat_order(root)
    played = {d for (_s, d) in root.play_history}
    mine = set(int(t) for t in root.my_hand)
    unknown = sorted(set(range(28)) - played - mine)
    made = {s: 0 for s in seats}
    for (seat, _d) in root.play_history:
        if seat in made:
            made[seat] += 1
    counts = [7 - made[s] for s in seats]
    seat_plays = {s: [d for (ps, d) in root.play_history if ps == s] for s in seats}
    return seats, unknown, counts, seat_plays


def brute_force_replay(root) -> set:
    """Literal per-ASSIGNMENT engine-rule check (slow; small positions only).

    Enumerate every unconstrained assignment and, for each, replay the observed
    history verifying each hidden seat's off-suit plays were forced under that
    candidate world (can_follow == the engine rule).
    """
    seats, unknown, counts, seat_plays = _position_bits(root)
    valid = set()
    n0, n1, n2 = counts
    U = list(unknown)
    for c0 in combinations(U, n0):
        rem1 = [t for t in U if t not in set(c0)]
        for c1 in combinations(rem1, n1):
            c2 = tuple(t for t in rem1 if t not in set(c1))
            hold = {seats[0]: set(c0), seats[1]: set(c1), seats[2]: set(c2)}
            if _history_legal(root, hold, seat_plays):
                valid.add((int(hand_to_mask(c0)), int(hand_to_mask(c1)), int(hand_to_mask(c2))))
    return valid


def brute_force_fast(root) -> set:
    """Independent forbidden-suit derivation (single replay pass) + numpy filter.

    Derives, per hidden seat, the union of led-suit followers it may not hold
    (a seat void of suit L held no follower of L, so neither does its current
    holding). This computes the SAME physics as enumerate_worlds via a wholly
    separate code path (direct can_follow / led_suit_for_lead_domino calls).
    """
    seats, unknown, counts, _ = _position_bits(root)
    forbidden = {s: 0 for s in seats}
    ph = root.play_history
    for i in range(0, len(ph), 4):
        trick = ph[i:i + 4]
        lead_tile = trick[0][1]
        led = led_suit_for_lead_domino(lead_tile, root.decl_id)
        for j, (seat, tile) in enumerate(trick):
            if j == 0 or seat not in seats:
                continue
            if not can_follow(tile, led, root.decl_id):
                fb = 0
                for d in range(28):
                    if can_follow(d, led, root.decl_id):
                        fb |= (1 << d)
                forbidden[seat] |= fb

    n0, n1, n2 = counts
    U = list(unknown)
    s0, s1, s2 = seats
    valid = set()
    for c0 in combinations(U, n0):
        m0 = int(hand_to_mask(c0))
        if m0 & forbidden[s0]:
            continue
        rem1 = [t for t in U if t not in set(c0)]
        for c1 in combinations(rem1, n1):
            m1 = int(hand_to_mask(c1))
            if m1 & forbidden[s1]:
                continue
            c2 = tuple(t for t in rem1 if t not in set(c1))
            m2 = int(hand_to_mask(c2))
            if m2 & forbidden[s2]:
                continue
            valid.add((m0, m1, m2))
    return valid


def _history_legal(root, hold, seat_plays) -> bool:
    """Replay history: every off-suit play by a hidden seat must have been forced.

    Reconstruct each hidden seat's ORIGINAL hand = current holding + its plays,
    then walk the history; at each play, the actor's hand at that moment is
    original minus its earlier plays. If the play did not follow the led suit,
    the actor must have held NO follower of the led suit at that moment.
    """
    seats = set(hold)
    original = {}
    for s in seats:
        original[s] = set(hold[s]) | set(seat_plays[s])
    # count of prior plays per seat, walking forward
    prior = {s: 0 for s in seats}
    ph = root.play_history
    for i in range(0, len(ph), 4):
        trick = ph[i:i + 4]
        lead_tile = trick[0][1]
        led = led_suit_for_lead_domino(lead_tile, root.decl_id)
        for (seat, tile) in trick:
            if seat in seats:
                if seat == trick[0][0] and tile == lead_tile:
                    pass  # leader never needs to follow
                elif not can_follow(tile, led, root.decl_id):
                    # off-suit: hand at this moment = original minus earlier plays
                    played_before = set(seat_plays[seat][: prior[seat]])
                    hand_now = original[seat] - played_before
                    if any(can_follow(h, led, root.decl_id) for h in hand_now):
                        return False
                prior[seat] += 1
    return True


def gen_endgame_root(seed: int):
    """Random position: play down to 3-5 tiles per seat, random me seat."""
    rng = random.Random(seed)
    state = zeb.new_game(zeb.game_seed(seed, seed * 7 + 1))
    decl = rng.choice((0, 1, 2, 3, 4, 5, 6, 7, 9))
    state = dataclasses.replace(state, decl_id=decl)
    target_left = rng.randint(3, 4)
    # play until the seat about to act has <= target_left tiles
    guard = 0
    while state.phase == GamePhase.PLAYING and guard < 40:
        player = zeb.current_player(state)
        left = sum(1 for d in state.hands[player] if d not in state.played)
        if left <= target_left and len(state.current_trick) == 0:
            break
        legal = zeb.legal_actions(state)
        if not legal:
            break
        state = zeb.apply_action(state, rng.choice(legal))
        guard += 1
    if state.phase != GamePhase.PLAYING:
        return None
    me = zeb.current_player(state)
    return make_root(state, me)


def test_worlds() -> None:
    n_pos = 320
    mismatches = 0
    counts = []
    checked = 0
    replay_checked = 0
    replay_mismatch = 0
    for seed in range(n_pos):
        root = gen_endgame_root(seed)
        if root is None:
            continue
        got = enumerate_worlds(root)
        got_set = {(int(r[0]), int(r[1]), int(r[2])) for r in got}
        # enumerate_worlds must not produce duplicates
        if len(got_set) != len(got):
            mismatches += 1
            continue
        bf = brute_force_fast(root)
        if got_set != bf:
            mismatches += 1
            print(f"  seed={seed} decl={root.decl_id} me={root.me} "
                  f"got={len(got_set)} bf={len(bf)} "
                  f"only_got={len(got_set - bf)} only_bf={len(bf - got_set)}")
        # literal per-assignment engine replay on small positions
        _, unknown, cnts, _ = _position_bits(root)
        from math import comb
        n0, n1, n2 = cnts
        total_assign = comb(len(unknown), n0) * comb(len(unknown) - n0, n1)
        if total_assign <= 2000:
            rep = brute_force_replay(root)
            replay_checked += 1
            if rep != got_set:
                replay_mismatch += 1
        counts.append(len(bf))
        checked += 1
    counts_arr = np.array(counts) if counts else np.array([0])
    check(
        "enumerate_worlds == independent forbidden-suit enumeration",
        mismatches == 0 and checked >= 200,
        f"positions={checked} mismatches={mismatches}",
    )
    check(
        "enumerate_worlds == literal per-assignment engine replay (small positions)",
        replay_mismatch == 0 and replay_checked >= 20,
        f"positions={replay_checked} mismatches={replay_mismatch}",
    )
    print(f"  world-count stats over {checked} positions: "
          f"min={int(counts_arr.min())} p50={int(np.percentile(counts_arr,50))} "
          f"mean={counts_arr.mean():.1f} p95={int(np.percentile(counts_arr,95))} "
          f"max={int(counts_arr.max())}")


def test_worst_case_timing() -> None:
    """A 4/4/4/4 position with no voids must enumerate 34650 worlds < 1s."""
    import time
    # build a synthetic root: me holds 4 tiles, no plays yet in the last 3
    # tricks... construct directly. Deal a game, play 3 full tricks with all
    # seats following so no voids constrain, ensuring 4 tiles left each.
    rng = random.Random(999)
    for seed in range(500):
        state = zeb.new_game(zeb.game_seed(999, seed))
        state = dataclasses.replace(state, decl_id=rng.choice((0, 1, 2, 3, 4, 5, 6, 7, 9)))
        # play exactly 3 tricks (12 plays)
        ok = True
        for _ in range(12):
            if state.phase != GamePhase.PLAYING:
                ok = False
                break
            legal = zeb.legal_actions(state)
            state = zeb.apply_action(state, rng.choice(legal))
        if not ok or state.phase != GamePhase.PLAYING or len(state.current_trick) != 0:
            continue
        me = zeb.current_player(state)
        root = make_root(state, me)
        # confirm 12 unknown tiles, 4 each
        t0 = time.perf_counter()
        w = enumerate_worlds(root)
        dt = time.perf_counter() - t0
        # only assert timing on the true worst case (no voids => 34650)
        if len(w) == 34650:
            check("worst-case 34650 worlds enumerated < 1s", dt < 1.0,
                  f"n={len(w)} t={dt*1000:.1f}ms")
            return
    check("worst-case 34650 worlds enumerated < 1s", False, "no 4/4/4 void-free position found")


if __name__ == "__main__":
    test_worlds()
    test_worst_case_timing()
    print(f"\n{len(_failures)} failure(s)" if _failures else "\nall gates green")
    sys.exit(1 if _failures else 0)

"""walt/worlds.py — exact enumeration of hidden-tile assignments (u).

Given an EndgameRoot (the walt seat's information set), enumerate every
assignment of the unknown tiles to the three non-me seats that is consistent
with physics: tile conservation, per-seat remaining counts implied by the
play history, and void constraints (a seat that failed to follow the led suit
at some moment held no follower of that suit then — hence its current holding
holds none either). Output columns are the three non-me seats in ascending
absolute seat order.
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "/Users/jason/code/mk5-main/.claude/worktrees/walt")

from walt.tables import N_DOMINOES, get_luts  # noqa: E402


def seat_order(root) -> tuple:
    """The three absolute seats (ascending) matching the world columns."""
    return tuple(s for s in range(4) if s != root.me)


def _void_forbidden(root) -> dict:
    """For each non-me seat, a uint32 bitmask of tiles it may NOT hold now.

    A seat that could not follow the led suit at some observed play held no
    follower of that suit at that moment; since its current holding is a
    subset of that hand, its current holding holds no follower either. So we
    accumulate the follow-bitmask of every suit that seat was seen void in.
    """
    luts = get_luts(root.decl_id)
    forbidden = {s: np.uint32(0) for s in range(4) if s != root.me}
    ph = root.play_history
    # play_history groups into tricks of 4 in play order; index 0 of each
    # group is the lead (the trailing group may be the partial current trick).
    for i in range(0, len(ph), 4):
        trick = ph[i:i + 4]
        lead_tile = trick[0][1]
        led = int(luts.led_suit[lead_tile])
        for (seat, tile) in trick[1:]:
            if seat == root.me:
                continue
            if not luts.can_follow[led, tile]:
                forbidden[seat] |= np.uint32(luts.can_follow_bits[led])
    return forbidden


def _combo_masks(bits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """All ways to pick k of the given single-bit masks.

    Returns (combos_idx (C,k) into `bits`, masks (C,) uint32). k may be 0.
    """
    n = len(bits)
    if k == 0:
        # one way to pick nothing (reshape(-1, 0) is ambiguous for numpy)
        return np.zeros((1, 0), dtype=np.int64), np.zeros(1, dtype=np.uint32)
    combos = np.array(list(combinations(range(n), k)), dtype=np.int64).reshape(-1, k)
    masks = np.bitwise_or.reduce(bits[combos], axis=1).astype(np.uint32)
    return combos, masks


def enumerate_worlds(root) -> np.ndarray:
    """Return (N, 3) uint32 hand masks for the three non-me seats.

    Columns are in ascending absolute seat order (== seat_order(root)).
    Physics-exact u.
    """
    seats = seat_order(root)

    played = {d for (_s, d) in root.play_history}
    mine = set(int(t) for t in root.my_hand)
    unknown = np.array(
        sorted(set(range(N_DOMINOES)) - played - mine), dtype=np.int64
    )
    U = len(unknown)

    # per-seat remaining counts implied by the history
    made = {s: 0 for s in seats}
    for (seat, _d) in root.play_history:
        if seat in made:
            made[seat] += 1
    counts = [7 - made[s] for s in seats]
    n0, n1, n2 = counts

    if sum(counts) != U:
        raise ValueError(
            f"tile conservation violated: unknown={U} but seat counts={counts}"
        )

    forbidden = _void_forbidden(root)
    s0, s1, s2 = seats
    f0, f1, f2 = forbidden[s0], forbidden[s1], forbidden[s2]

    bits = (np.uint32(1) << unknown.astype(np.uint32))  # (U,) one bit per unknown tile

    combos0, masks0 = _combo_masks(bits, n0)
    valid0 = (masks0 & f0) == np.uint32(0)

    out_cols = [[], [], []]  # collected masks per seat column
    all_bits = np.bitwise_or.reduce(bits) if U > 0 else np.uint32(0)

    for ci in np.nonzero(valid0)[0]:
        c0 = combos0[ci]
        m0 = masks0[ci]
        rem_idx = np.array(
            sorted(set(range(U)) - set(c0.tolist())), dtype=np.int64
        )
        rem_bits = bits[rem_idx] if len(rem_idx) else bits[:0]

        combos1, masks1 = _combo_masks(rem_bits, n1)
        rem_all = np.bitwise_or.reduce(rem_bits) if len(rem_idx) else np.uint32(0)
        masks2 = (rem_all & ~masks1).astype(np.uint32)  # seat2 = remainder (disjoint)

        keep = ((masks1 & f1) == np.uint32(0)) & ((masks2 & f2) == np.uint32(0))
        if not keep.any():
            continue
        k1 = masks1[keep]
        k2 = masks2[keep]
        out_cols[0].append(np.full(len(k1), m0, dtype=np.uint32))
        out_cols[1].append(k1)
        out_cols[2].append(k2)

    if not out_cols[0]:
        return np.zeros((0, 3), dtype=np.uint32)

    col0 = np.concatenate(out_cols[0])
    col1 = np.concatenate(out_cols[1])
    col2 = np.concatenate(out_cols[2])
    _ = all_bits  # (kept for clarity; masks are validated by conservation above)
    return np.stack([col0, col1, col2], axis=1).astype(np.uint32)

"""atlas/fiber.py — the exact unknowns.

The unknowns of a coordinate are not vague but delimited: an exactly
enumerable fiber of consistent hidden deals.  ``fiber(coord)`` returns every
assignment of the hidden tiles (not in the viewer's hand, not played) to the
three hidden seats that is consistent with physics — per-seat
remaining-count conservation, played-tile membership, and the voids matrix —
and nothing else.  This is exact, enumerable, no modeling: the estimated
layer (belief) is a measure a consumer hands in, never atlas's business
beyond the uniform default.

The algorithm and column convention mirror walt.worlds.enumerate_worlds
(columns = the three non-viewer seats in ascending absolute seat order); the
parity is gated in tests.  atlas does not import walt at runtime — the void
masks come from atlas.algebra's can_follow planes.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from atlas.algebra import ALL_TILES, N_DOMINOES, N_LED_SUITS, get_algebra


def hidden_seats(coord) -> tuple:
    """The three absolute seats (ascending) matching the fiber columns."""
    return tuple(s for s in range(4) if s != coord.viewer)


def _forbidden(coord, seat: int) -> np.uint32:
    """uint32 mask of tiles ``seat`` may NOT hold: the union of every led
    suit's follow set that the seat was observed void in."""
    alg = get_algebra()
    v = int(coord.voids[seat])
    m = np.uint32(0)
    for ls in range(N_LED_SUITS):
        if (v >> ls) & 1:
            m |= np.uint32(alg.can_follow[coord.decl_id, ls])
    return m


def unknown_tiles(coord) -> np.ndarray:
    """Ascending tile ids that are hidden from the viewer: not held, not
    played by anyone."""
    played_union = 0
    for m in coord.played:
        played_union |= int(m)
    hidden = int(ALL_TILES) & ~int(coord.viewer_hand) & ~played_union
    return np.flatnonzero((hidden >> np.arange(N_DOMINOES)) & 1).astype(np.int64)


def _combo_masks(bits: np.ndarray, k: int):
    """All ways to pick k of the single-bit masks in ``bits``: (idx (C,k),
    masks (C,) uint32). k may be 0."""
    n = len(bits)
    if k == 0:
        return np.zeros((1, 0), dtype=np.int64), np.zeros(1, dtype=np.uint32)
    combos = np.array(list(combinations(range(n), k)), dtype=np.int64).reshape(-1, k)
    masks = np.bitwise_or.reduce(bits[combos], axis=1).astype(np.uint32)
    return combos, masks


def fiber(coord) -> np.ndarray:
    """(N, 3) uint32 hand masks for the three hidden seats (ascending seat
    order). Physics-exact: tile conservation, per-seat remaining counts
    implied by what each seat has played, and the voids matrix."""
    seats = hidden_seats(coord)
    unknown = unknown_tiles(coord)
    U = len(unknown)

    counts = [7 - int(bin(int(coord.played[s])).count("1")) for s in seats]
    if sum(counts) != U:
        raise ValueError(
            f"tile conservation violated: unknown={U} but seat counts={counts}")

    f0, f1, f2 = (_forbidden(coord, s) for s in seats)
    n0, n1, n2 = counts

    bits = (np.uint32(1) << unknown.astype(np.uint32))
    combos0, masks0 = _combo_masks(bits, n0)
    valid0 = (masks0 & f0) == np.uint32(0)

    out0, out1, out2 = [], [], []
    for ci in np.nonzero(valid0)[0]:
        c0 = combos0[ci]
        m0 = masks0[ci]
        rem_idx = np.array(sorted(set(range(U)) - set(c0.tolist())), dtype=np.int64)
        rem_bits = bits[rem_idx] if len(rem_idx) else bits[:0]

        combos1, masks1 = _combo_masks(rem_bits, n1)
        rem_all = np.bitwise_or.reduce(rem_bits) if len(rem_idx) else np.uint32(0)
        masks2 = (rem_all & ~masks1).astype(np.uint32)  # seat2 = disjoint remainder

        keep = ((masks1 & f1) == np.uint32(0)) & ((masks2 & f2) == np.uint32(0))
        if not keep.any():
            continue
        out0.append(np.full(int(keep.sum()), m0, dtype=np.uint32))
        out1.append(masks1[keep])
        out2.append(masks2[keep])

    if not out0:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.stack([np.concatenate(out0), np.concatenate(out1),
                     np.concatenate(out2)], axis=1).astype(np.uint32)


def uniform_weights(n: int) -> np.ndarray:
    """The default measure on the fiber: uniform. Consumers replace it with
    a belief (a tilt of uniform by the field's discretionary likelihood)."""
    return np.full(int(n), 1.0 / int(n), dtype=np.float64) if n else \
        np.zeros(0, dtype=np.float64)

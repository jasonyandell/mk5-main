"""The atlas: one fully known deal, its perfect-information game tree, and
Walt's honest imperfect-information evaluation written on its positions.

The mold is the tree's shape — how many distinct positions each layer holds
(layer d = d tiles played), every legal play from every seat's real hand
followed, with transposition on the whole public record (`Record.key()`:
played, leader, open trick, banked, forbid). The terminal layer is left out
by default: it holds only scored ends, nothing to label. This plain search
is the referee for the compiled `CompiledWalt.mold`, which counts the
middle layers' hundreds of millions of positions.
"""
from __future__ import annotations

from typing import Sequence

from scenario.record import empty_record
from scenario.rules import ALL_TILES, rules_for, tiles_of

LAST_LAYER = 27  # the final play's layer; layer 28 is the finished hand


def mold(hands: Sequence[int], decl_id: int, leader: int, max_layer: int | None = None) -> list[int]:
    """Positions per layer, layers 0..=max_layer, for the deal `hands` (four
    tile masks, absolute seats) with `leader` to lead under `decl_id`."""
    dealt = len(hands) == 4 and hands[0] | hands[1] | hands[2] | hands[3] == ALL_TILES
    if not dealt or any(h.bit_count() != 7 for h in hands) or not 0 <= leader < 4:
        raise ValueError("a deal is four disjoint seven-tile hands and a leader in 0..4")
    last = LAST_LAYER if max_layer is None else max_layer
    if not 0 <= last <= 28:
        raise ValueError("a hand has layers 0..=28")
    rules = rules_for(decl_id)
    root = empty_record(leader)
    layer = {root.key(): root}
    counts = []
    for d in range(last + 1):
        counts.append(len(layer))
        if d == last:
            break
        nxt = {}
        for rec in layer.values():
            for t in tiles_of(rules.legal(hands[rec.seat] & ~rec.played, rec.led)):
                child = rec.advance(t, rules)
                nxt[child.key()] = child
        layer = nxt
    return counts


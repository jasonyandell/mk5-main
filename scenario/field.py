"""The field net: a small integer policy that plays the modeled seats.

The ladder's bottom rung is random legal play from the tape. A `FieldNet`
replaces it with a policy learned from the pour (`scenario.field_train`):
rung −1 becomes a player that plays like Walt, so a one-rung Walt plans
against a Walt-like field for the price of a forward pass per modeled move.
The net is a field only — it never scores a leaf or a cell, and the live
seat still solves.

A modeled seat's move is a pure function of what that seat knows — its
remaining hand, the public record (tiles gone, the open trick, voids shown,
points banked), the bid (read directly and as what each side still needs)
and its side — exactly the key the planner caches
modeled moves under. The features are that and nothing more, read from the
mover's chair (partner, left, right) with the tiles relabeled so the trump
pip is pip 6 and the other pips keep their order (`canon`): one trump suit
to learn instead of seven. The net is defined on the pip-trump declarations,
the only ones the `bid30` and heuristic bidders call.

Inference is integer arithmetic, so the referee (this file) and the compiled
core (`scenario/core/src/field.rs`) agree by construction: the features are
bits, the first layer sums the active rows, each hidden layer is
max(acc >> shift, 0), and the move is the highest-scoring legal tile, ties
to the lowest tile id. Weights live in a `.npz` both load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forge.oracle.declarations import PIP_TRUMP_IDS
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW, DOMINOES

from scenario.record import Record
from scenario.rules import N_TILES, Rules, tiles_of

LAYOUT = "field-v2"
STEPS = tuple(range(2, 43, 2))          # thermometer thresholds, in points
BIDS = tuple(range(30, 43))             # the contract, in points: a mark bid needs all 42

# Feature blocks, in order: (name, width).
BLOCKS = (
    ("hand", N_TILES),              # the mover's remaining tiles
    ("gone", N_TILES),              # every tile played so far, open trick included
    ("trick", 3 * N_TILES),         # the open trick's tiles, by who played them (left, partner, right)
    ("void", 3 * N_TILES),          # tiles each other seat provably cannot hold
    ("count", 2 * N_TILES),         # where the 10-count and 5-count tiles sit under this declaration
    ("side", 1),                    # the mover is on the bidding side
    ("open", 4),                    # tiles already in the open trick (0 = the mover leads)
    ("winning", 3),                 # who holds the open trick so far (left, partner, right)
    ("need", len(STEPS)),           # the bidding side still needs >= step points to make
    ("slack", len(STEPS)),          # the defenders still need >= step points to set
    ("bid", len(BIDS)),             # the contract, one-hot (bids above 42 read as 42)
)
OFFSET = {}
_at = 0
for _name, _width in BLOCKS:
    OFFSET[_name] = _at
    _at += _width
N_FEATURES = _at


def _canon(decl_id: int) -> tuple[int, ...]:
    """Tile id -> canonical tile id: the trump pip becomes 6, the others keep order."""
    pip = [q if q < decl_id else q - 1 for q in range(7)]
    pip[decl_id] = 6
    out = []
    for t in range(N_TILES):
        a, b = pip[DOMINO_HIGH[t]], pip[DOMINO_LOW[t]]
        out.append(DOMINOES.index((max(a, b), min(a, b))))
    return tuple(out)


CANON: dict[int, tuple[int, ...]] = {d: _canon(d) for d in PIP_TRUMP_IDS}


def features(rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int) -> list[int]:
    """The active feature indices of a modeled seat's decision."""
    canon = CANON.get(rules.decl_id)
    if canon is None:
        raise ValueError(f"the field net plays pip-trump declarations only, not {rules.decl_id}")
    on = [OFFSET["hand"] + canon[t] for t in tiles_of(own)]
    on += [OFFSET["gone"] + canon[t] for t in tiles_of(rec.played)]
    n = len(rec.trick)
    for i, t in enumerate(rec.trick):
        on.append(OFFSET["trick"] + ((i - n) % 4 - 1) * N_TILES + canon[t])
    for rel in (1, 2, 3):
        on += [OFFSET["void"] + (rel - 1) * N_TILES + canon[t] for t in tiles_of(rec.forbid[(seat + rel) % 4])]
    for t in range(N_TILES):
        if rules.count[t] == 10:
            on.append(OFFSET["count"] + canon[t])
        elif rules.count[t] == 5:
            on.append(OFFSET["count"] + N_TILES + canon[t])
    if seat % 2 == bidder_team:
        on.append(OFFSET["side"])
    on.append(OFFSET["open"] + n)
    if n:
        ranks = rules.rank[rules.led_suit[rec.trick[0]]]
        best = 0
        for i in range(1, n):
            if ranks[rec.trick[i]] > ranks[rec.trick[best]]:
                best = i
        on.append(OFFSET["winning"] + (best - n) % 4 - 1)
    need = bid - rec.banked[bidder_team]
    slack = (43 - bid) - rec.banked[1 - bidder_team]
    on += [OFFSET["need"] + i for i, s in enumerate(STEPS) if need >= s]
    on += [OFFSET["slack"] + i for i, s in enumerate(STEPS) if slack >= s]
    on.append(OFFSET["bid"] + min(max(bid, BIDS[0]), BIDS[-1]) - BIDS[0])
    return on


@dataclass(frozen=True, eq=False)
class FieldNet:
    """An integer MLP, features -> 28 canonical tile scores. `w1` is
    [features, h1] (a row per feature), `w2` [h2, h1], `w3` [28, h2]; all
    int64 arrays holding int16-range weights and wider biases."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    s1: int
    s2: int

    def __post_init__(self) -> None:
        h1, h2 = self.b1.shape[0], self.b2.shape[0]
        shapes = (self.w1.shape, self.w2.shape, self.w3.shape, self.b3.shape)
        if shapes != ((N_FEATURES, h1), (h2, h1), (N_TILES, h2), (N_TILES,)):
            raise ValueError(f"field net shapes {shapes} do not fit {N_FEATURES} features")
        if self.s1 < 0 or self.s2 < 0:
            raise ValueError("shifts are non-negative")
        for a in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3):
            if a.dtype != np.int64:
                raise ValueError("field net arrays are int64")

    @staticmethod
    def load(path: str | Path) -> FieldNet:
        with np.load(path) as z:
            if str(z["layout"]) != LAYOUT:
                raise ValueError(f"{path} is layout {z['layout']}, not {LAYOUT}")
            return FieldNet(*(z[k].astype(np.int64) for k in ("w1", "b1", "w2", "b2", "w3", "b3")),
                            int(z["s1"]), int(z["s2"]))

    def save(self, path: str | Path) -> None:
        np.savez(path, layout=LAYOUT, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3, s1=self.s1, s2=self.s2)

    def scores(self, active: list[int]) -> np.ndarray:
        """[28] int64 scores over canonical tiles."""
        h = np.maximum((self.b1 + self.w1[active].sum(axis=0)) >> self.s1, 0)
        h = np.maximum((self.w2 @ h + self.b2) >> self.s2, 0)
        return self.w3 @ h + self.b3

    def move(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int) -> int:
        """The highest-scoring legal tile; ties to the lowest tile id."""
        z = self.scores(features(rules, seat, own, rec, bid, bidder_team))
        canon = CANON[rules.decl_id]
        legal = tiles_of(rules.legal(own, rec.led))
        best = legal[0]
        for t in legal[1:]:
            if z[canon[t]] > z[canon[best]]:
                best = t
        return best

    def core_args(self) -> tuple:
        """The net as the compiled core takes it."""
        return (
            int(self.b1.shape[0]), int(self.b2.shape[0]),
            self.w1.ravel().tolist(), self.b1.tolist(), self.w2.ravel().tolist(), self.b2.tolist(),
            self.w3.ravel().tolist(), self.b3.tolist(), self.s1, self.s2,
            [(d, list(c)) for d, c in CANON.items()],
        )

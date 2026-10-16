# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

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
to learn instead of seven. Two feature layouts exist. `field-v2`, the
shipped net's, plays the seven pip trumps only. `field-v3` appends one
block, the declaration's kind (a pip trump, doubles trump, no trump), and
plays doubles trump (7) and no trump (9) as well, with the tiles left as
they are; every v2 feature keeps its index. A net carries its layout.

Inference is integer arithmetic, so the referee (this file) and the compiled
core (`scenario/core/src/field.rs`) agree by construction: the features are
bits, the first layer sums the active rows, each hidden layer is
max(acc >> shift, 0), and the move is the highest-scoring legal tile, ties
to the lowest tile id. The trainer saves a net as a `.npz`; the core takes it
as one little-endian blob (`to_bytes`, laid out in `field.rs`).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forge.oracle.declarations import PIP_TRUMP_IDS
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW, DOMINOES

from scenario.record import Record
from scenario.rules import N_TILES, Rules, tiles_of

LAYOUTS = ("field-v2", "field-v3")
LAYOUT = "field-v3"                     # what the trainer builds
_MAGIC = b"FNET"
_BLOB_DTYPES = ("<i2", "<i8", "<i2", "<i8", "<i2", "<i8")              # w1 b1 w2 b2 w3 b3
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
    ("kind", 3),                    # field-v3 only: a pip trump, doubles trump, no trump
)
OFFSET = {}
_at = 0
for _name, _width in BLOCKS:
    OFFSET[_name] = _at
    _at += _width
N_FEATURES = _at                                    # field-v3's width
N_FEATURES_OF = {"field-v2": OFFSET["kind"], "field-v3": N_FEATURES}
DOUBLES_TRUMP, NO_TRUMP = 7, 9
KIND = {**{d: 0 for d in PIP_TRUMP_IDS}, DOUBLES_TRUMP: 1, NO_TRUMP: 2}


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
_IDENTITY = tuple(range(N_TILES))


def canon_for(layout: str, decl_id: int) -> tuple[int, ...] | None:
    """Tile id -> canonical tile id under `layout`, or None where that layout
    does not play: pips relabel the trump to 6; doubles trump and no trump
    (`field-v3`) keep the tiles as they are and set the kind bit instead."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown field layout {layout!r}")
    if decl_id in CANON:
        return CANON[decl_id]
    if layout == "field-v3" and decl_id in (DOUBLES_TRUMP, NO_TRUMP):
        return _IDENTITY
    return None


def plays(layout: str, decl_id: int) -> bool:
    return canon_for(layout, decl_id) is not None


def refusal(layout: str, decl_id: int) -> str:
    """Why a net of `layout` will not play `decl_id` (the message every refusal carries)."""
    if layout == "field-v2":
        return f"a field-v2 field net plays pip-trump declarations only, not {decl_id}"
    return f"a {layout} field net does not play declaration {decl_id}"


def features(rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int,
             layout: str = LAYOUT) -> list[int]:
    """The active feature indices of a modeled seat's decision, in `layout`."""
    canon = canon_for(layout, rules.decl_id)
    if canon is None:
        raise ValueError(refusal(layout, rules.decl_id))
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
    if layout == "field-v3":
        on.append(OFFSET["kind"] + KIND[rules.decl_id])
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
    layout: str = LAYOUT

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown field layout {self.layout!r}")
        h1, h2 = self.b1.shape[0], self.b2.shape[0]
        n = N_FEATURES_OF[self.layout]
        shapes = (self.w1.shape, self.w2.shape, self.w3.shape, self.b3.shape)
        if shapes != ((n, h1), (h2, h1), (N_TILES, h2), (N_TILES,)):
            raise ValueError(f"field net shapes {shapes} do not fit {n} features ({self.layout})")
        if self.s1 < 0 or self.s2 < 0:
            raise ValueError("shifts are non-negative")
        for a in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3):
            if a.dtype != np.int64:
                raise ValueError("field net arrays are int64")

    @property
    def n_features(self) -> int:
        return N_FEATURES_OF[self.layout]

    def plays(self, decl_id: int) -> bool:
        """Whether the net plays this declaration (its layout's canon has it)."""
        return plays(self.layout, decl_id)

    @staticmethod
    def load(path: str | Path) -> FieldNet:
        with np.load(path) as z:
            layout = str(z["layout"])
            if layout not in LAYOUTS:
                raise ValueError(f"{path} is layout {layout}, not one of {LAYOUTS}")
            return FieldNet(*(z[k].astype(np.int64) for k in ("w1", "b1", "w2", "b2", "w3", "b3")),
                            int(z["s1"]), int(z["s2"]), layout)

    def save(self, path: str | Path) -> None:
        np.savez(path, layout=self.layout, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3, s1=self.s1, s2=self.s2)

    def to_bytes(self) -> bytes:
        """The net as one blob: `FNET`, the layout name (one length byte, then
        ASCII), h1 h2 s1 s2 as uint32, then w1 b1 w2 b2 w3 b3 in row-major
        order — weights int16 (the trainer quantizes them to fit), biases
        int64 (they carry accumulator scale, which no narrower width bounds)
        — all little-endian. The compiled core and the zeb package parse it."""
        for w in (self.w1, self.w2, self.w3):
            if w.min() < -(1 << 15) or w.max() >= 1 << 15:
                raise ValueError("field net weights must fit int16")
        name = self.layout.encode("ascii")
        head = _MAGIC + bytes([len(name)]) + name + struct.pack("<4I", self.b1.shape[0], self.b2.shape[0], self.s1, self.s2)
        arrays = zip((self.w1, self.b1, self.w2, self.b2, self.w3, self.b3), _BLOB_DTYPES)
        return head + b"".join(a.astype(dtype).tobytes() for a, dtype in arrays)

    @staticmethod
    def from_bytes(blob: bytes) -> FieldNet:
        """The net `to_bytes` wrote; refuses another magic, an unknown layout or a wrong size."""
        if blob[:4] != _MAGIC or len(blob) < 5:
            raise ValueError("not a field net blob")
        n = blob[4]
        layout = blob[5:5 + n].decode("ascii", "replace")
        if layout not in LAYOUTS:
            raise ValueError(f"field net blob names layout {layout!r}, not one of {LAYOUTS}")
        head = 5 + n
        at = head + 16
        if len(blob) < at:
            raise ValueError("field net blob is truncated")
        h1, h2, s1, s2 = struct.unpack_from("<4I", blob, head)
        shapes = ((N_FEATURES_OF[layout], h1), (h1,), (h2, h1), (h2,), (N_TILES, h2), (N_TILES,))
        sizes = [math.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in zip(shapes, _BLOB_DTYPES)]
        if len(blob) != at + sum(sizes):
            raise ValueError("field net blob is the wrong size for its widths")
        arrays = []
        for shape, dtype, size in zip(shapes, _BLOB_DTYPES, sizes):
            arrays.append(np.frombuffer(blob, dtype=dtype, count=math.prod(shape), offset=at)
                          .astype(np.int64).reshape(shape))
            at += size
        return FieldNet(*arrays, s1, s2, layout)

    def scores(self, active: list[int]) -> np.ndarray:
        """[28] int64 scores over canonical tiles."""
        h = np.maximum((self.b1 + self.w1[active].sum(axis=0)) >> self.s1, 0)
        h = np.maximum((self.w2 @ h + self.b2) >> self.s2, 0)
        return self.w3 @ h + self.b3

    def allowed(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int,
                margin: int) -> int:
        """The legal tiles scoring within `margin` of the best legal score, as
        a mask: the set a reading holds the seat to (`planner.Reading`).
        Margin 0 is the best score and its ties; a margin wider than every
        gap is every legal tile."""
        z = self.scores(features(rules, seat, own, rec, bid, bidder_team, self.layout))
        canon = canon_for(self.layout, rules.decl_id)
        legal = tiles_of(rules.legal(own, rec.led))
        best = max(int(z[canon[t]]) for t in legal)
        return sum(1 << t for t in legal if int(z[canon[t]]) >= best - margin)

    def move(self, rules: Rules, seat: int, own: int, rec: Record, bid: int, bidder_team: int) -> int:
        """The highest-scoring legal tile; ties to the lowest tile id."""
        z = self.scores(features(rules, seat, own, rec, bid, bidder_team, self.layout))
        canon = canon_for(self.layout, rules.decl_id)
        legal = tiles_of(rules.legal(own, rec.led))
        best = legal[0]
        for t in legal[1:]:
            if z[canon[t]] > z[canon[best]]:
                best = t
        return best

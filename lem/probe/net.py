# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The field net's insides: every layer's pre-activations, exactly.

`scenario.field.FieldNet` computes a move; this module computes what each
neuron saw on the way. The arithmetic is the referee's own (integer, the
same shifts), so `forward` agrees with `FieldNet.scores` bit for bit
(`lem/tests/test_probe.py` holds them equal).

A neuron has two poles: the positions that drive its pre-activation highest
(`+`) and lowest (`−`). The ReLU passes only the first forward, but the
second is what the net learned to shut the neuron off with, and both are
audited. Pole `2j` of a layer is neuron `j`'s `+` pole, `2j + 1` its `−`
pole; layer 2's poles follow layer 1's.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scenario.field import FieldNet

CHUNK = 1 << 15


@dataclass(frozen=True)
class Activations:
    """Every layer of the net over a batch of rows, int64."""

    z1: np.ndarray    # [N, h1] pre-activations
    a1: np.ndarray    # [N, h1] after shift and ReLU
    z2: np.ndarray    # [N, h2]
    a2: np.ndarray    # [N, h2]
    out: np.ndarray   # [N, 28] canonical tile scores

    @property
    def n_poles(self) -> int:
        return 2 * (self.z1.shape[1] + self.z2.shape[1])

    def pole(self, p: int) -> np.ndarray:
        """The signal of pole `p`: the pre-activation (`+`) or its negation (`−`)."""
        h1 = self.z1.shape[1]
        if p < 2 * h1:
            z = self.z1[:, p // 2]
        else:
            z = self.z2[:, (p - 2 * h1) // 2]
        return z if p % 2 == 0 else -z

    def alive(self, p: int) -> float:
        """The share of rows on which this pole's neuron is on (`+`) or off (`−`)."""
        h1 = self.z1.shape[1]
        a = self.a1[:, p // 2] if p < 2 * h1 else self.a2[:, (p - 2 * h1) // 2]
        on = float((a > 0).mean())
        return on if p % 2 == 0 else 1.0 - on


def pole_name(net: FieldNet, p: int) -> str:
    h1 = net.b1.shape[0]
    layer, j = (1, p // 2) if p < 2 * h1 else (2, (p - 2 * h1) // 2)
    return f"L{layer}{'+' if p % 2 == 0 else '-'}{j:03d}"


def load_net(path: str | Path) -> FieldNet:
    """A `.fnet` blob (the package's) or a trainer `.npz`."""
    path = Path(path)
    return FieldNet.load(path) if path.suffix == ".npz" else FieldNet.from_bytes(path.read_bytes())


def forward(net: FieldNet, x: np.ndarray) -> Activations:
    """Every layer over the feature rows `x` ([N, F] of 0/1), in chunks."""
    parts = [[], [], [], [], []]
    for at in range(0, len(x), CHUNK):
        xi = x[at:at + CHUNK].astype(np.int64)
        z1 = xi @ net.w1 + net.b1
        a1 = np.maximum(z1 >> net.s1, 0)
        z2 = a1 @ net.w2.T + net.b2
        a2 = np.maximum(z2 >> net.s2, 0)
        out = a2 @ net.w3.T + net.b3
        for part, a in zip(parts, (z1, a1, z2, a2, out)):
            part.append(a)
    return Activations(*(np.concatenate(p) if p else np.zeros((0, 0), np.int64) for p in parts))


def picks(out: np.ndarray, actions: np.ndarray, canon: np.ndarray) -> np.ndarray:
    """The net's move per row: the highest-scoring legal tile, ties to the
    lowest tile id, as `FieldNet.move` picks it. `actions` is [N, 7] real
    tile ids ascending, −1 padded; `canon` [N, 28] maps each row's real tile
    to its canonical slot."""
    n = len(out)
    best = np.full(n, -1, np.int64)
    best_score = np.full(n, np.iinfo(np.int64).min, np.int64)
    for k in range(actions.shape[1]):
        t = actions[:, k].astype(np.int64)
        ok = t >= 0
        slot = canon[np.arange(n), np.where(ok, t, 0)]
        score = out[np.arange(n), slot]
        better = ok & (score > best_score)
        best[better] = t[better]
        best_score[better] = score[better]
    return best

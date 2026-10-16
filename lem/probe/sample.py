# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The audit sample: the rows of a pour the net never trained on.

`scenario.field_train.train` holds out every deal whose index is a multiple
of ten (`deal % 10 == 0`); the pours that trained the shipped net are 50,000
deals each, so the split is the same by local or global index. `build` keeps
exactly those positions of one pour, with everything the audit needs: the
net's feature row, the legal moves, Walt's choice, and the frame (declaration,
viewer, remaining hand, bidder, bid, plays) that Silas and the position
cards are built from. `merge` concatenates the parts of several pours.

One pour is 3 GB in memory and a few minutes to walk, so a build is one
`scripts/cpu-slot` call per pour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenario.atlas import Pour, load
from scenario.field import N_FEATURES_OF, canon_for, features
from scenario.rules import N_TILES, rules_for

EVERY = 10          # the trainer's validation split: deal % EVERY == 0
MAX_PLAYS = 28
MAX_ACTIONS = 7

COLUMNS = ("x", "legal", "chosen", "actions", "deal", "index", "decl", "viewer", "own",
           "bidder", "bid", "plays", "n_plays", "pour")


def rows(pour: Pour, every: int = EVERY, layout: str = "field-v2") -> dict[str, np.ndarray]:
    keep = []
    for i, pos in enumerate(pour.positions()):
        if pos.deal % every == 0:
            keep.append((i, pos))
    n = len(keep)
    x = np.zeros((n, N_FEATURES_OF[layout]), np.uint8)
    legal = np.zeros((n, N_TILES), bool)
    chosen = np.zeros(n, np.int8)
    actions = np.full((n, MAX_ACTIONS), -1, np.int8)
    deal = np.zeros(n, np.int32)
    index = np.zeros(n, np.int32)
    decl = np.zeros(n, np.int8)
    viewer = np.zeros(n, np.int8)
    own = np.zeros(n, np.uint32)
    bidder = np.zeros(n, np.int8)
    bid = np.zeros(n, np.int16)
    plays = np.full((n, MAX_PLAYS, 2), -1, np.int8)
    n_plays = np.zeros(n, np.int8)
    for r, (i, pos) in enumerate(keep):
        rules = rules_for(pos.decl_id)
        x[r, features(rules, pos.viewer, pos.own, pos.record, pos.bid, pos.bidder % 2, layout)] = 1
        canon = canon_for(layout, pos.decl_id)
        legal[r, [canon[t] for t in pos.actions]] = True
        chosen[r] = pos.chosen
        actions[r, :len(pos.actions)] = pos.actions
        deal[r], index[r], decl[r], viewer[r] = pos.deal, i, pos.decl_id, pos.viewer
        own[r], bidder[r], bid[r] = pos.own, pos.bidder, pos.bid
        for k, (s, t) in enumerate(pos.plays):
            plays[r, k] = (s, t)
        n_plays[r] = len(pos.plays)
    pour_col = np.zeros(n, np.int8)
    return dict(x=x, legal=legal, chosen=chosen, actions=actions, deal=deal, index=index, decl=decl,
                viewer=viewer, own=own, bidder=bidder, bid=bid, plays=plays, n_plays=n_plays, pour=pour_col)


def build(pour_dir: str | Path, out: str | Path, every: int = EVERY, log=print, layout: str = "field-v2") -> int:
    pour = load(pour_dir)
    log(f"{pour_dir}: {len(pour):,} positions, {len(pour.hands):,} deals, utility {pour.utility}")
    d = rows(pour, every, layout)
    d["source"] = np.array(str(pour_dir))
    d["layout"] = np.array(layout)
    np.savez_compressed(out, **d)
    log(f"{out}: {len(d['x']):,} rows (deal % {every} == 0)")
    return len(d["x"])


def merge(out: str | Path, parts: list[str | Path], log=print) -> int:
    cols = {c: [] for c in COLUMNS}
    sources, first = [], 0
    for k, part in enumerate(parts):
        with np.load(part) as z:
            n = len(z["x"])
            for c in COLUMNS:
                a = z[c]
                if c == "deal":
                    a = a + first           # global deal ids, as the trainer numbers them
                if c == "pour":
                    a = np.full(n, k, np.int8)
                cols[c].append(a)
            sources.append(str(z["source"]))
            layout = str(z["layout"]) if "layout" in z.files else "field-v2"
            deals = int(z["deal"].max()) + 1 if n else 0
            first += max(deals, 50_000)    # the training pours are 50,000 deals each
    d = {c: np.concatenate(v) for c, v in cols.items()}
    d["source"] = np.array(sources)
    d["layout"] = np.array(layout)
    np.savez_compressed(out, **d)
    log(f"{out}: {len(d['x']):,} rows from {len(parts)} parts")
    return len(d["x"])


class Sample:
    """A merged sample, loaded lazily by column."""

    def __init__(self, path: str | Path):
        self._z = np.load(path)

    def __getattr__(self, name: str) -> np.ndarray:
        if name in COLUMNS or name == "source":
            return self._z[name]
        if name == "layout":
            return str(self._z["layout"]) if "layout" in self._z.files else "field-v2"
        raise AttributeError(name)

    def __len__(self) -> int:
        return len(self._z["deal"])

    def canon(self) -> np.ndarray:
        """[N, 28] the canonical slot of each real tile, per row."""
        table = np.array([canon_for(self.layout, d) or tuple(range(N_TILES)) for d in range(10)], np.int64)
        return table[self.decl.astype(np.int64)]

    def frame_args(self, r: int) -> tuple:
        """What `scenario_core.SilasFrame` takes for row r."""
        k = int(self.n_plays[r])
        plays = [(int(s), int(t)) for s, t in self.plays[r, :k]]
        return (int(self.decl[r]), int(self.viewer[r]), int(self.own[r]), int(self.bidder[r]),
                int(self.bid[r]), plays)

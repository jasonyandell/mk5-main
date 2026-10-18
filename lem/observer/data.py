# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""One episode per viewer. Only initial own tiles and public plays reach x.

The referee validates the recording and supplies diagnostic labels; hidden hands
never enter the token encoder. No planner values, worlds or outcomes are loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from forge.oracle.tables import DOMINOES
from scenario.record import empty_record
from scenario.rules import ALL_TILES, rules_for, tiles_of

SEAT, DECL, BOS, HAND, PLAY, BID30 = 28, 32, 42, 43, 44, 45
PREFIX = 13
QUERIES = np.arange(PREFIX, PREFIX + 56, 2)
VERSION = "initial-hand-relative-seats-v1"


def label(token):
    if token < 28:
        a, b = DOMINOES[token]
        return f"{a}-{b}"
    if token < 32:
        return ["you", "left", "partner", "right"][token - SEAT]
    if token < 42:
        return f"trump:{token - DECL}" if token < 39 else {39: "doubles trump", 40: "unsupported", 41: "no trump"}[token]
    return {BOS: "start", HAND: "your hand", PLAY: "play begins", BID30: "bid:30"}[token]


def encode(hand, viewer, bidder, decl, public_plays):
    """No other hands accepted; an unfinished prefix is also valid."""
    if not 0 <= viewer < 4 or not 0 <= bidder < 4 or decl not in (*range(8), 9):
        raise ValueError("unsupported seat or declaration")
    own = tiles_of(int(hand))
    if len(own) != 7 or int(hand) & ~ALL_TILES:
        raise ValueError("the initial hand has seven tiles")
    tokens = [BOS, BID30, SEAT + (bidder - viewer) % 4, DECL + decl, HAND, *own, PLAY]
    for seat, tile in public_plays:
        if not 0 <= int(seat) < 4 or not 0 <= int(tile) < 28:
            raise ValueError("invalid public play")
        tokens.extend((SEAT + (int(seat) - viewer) % 4, int(tile)))
    return np.array(tokens, dtype=np.int64)


def partition(hands):
    """Whole physical deals stay together, even if a caller renumbers seeds."""
    key = np.array(hands, dtype="<u4").tobytes()
    bucket = int.from_bytes(hashlib.sha256(b"observer-split-v1" + key).digest()[:4], "big") % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def load_deals(sources, limit):
    records, provenance = [], []
    for source in sources:
        source = Path(source)
        raw = (source / "manifest.json").read_bytes()
        m = json.loads(raw)
        if (m["bidder"], m["bid"]) != (0, 30):
            raise ValueError("this first observer supports the pours' seat-0 bid-30 contract")
        with np.load(source / "pour.npz", allow_pickle=False) as p:
            hands, decls, plays = (p[k] for k in ("hands", "decl", "plays"))
        if hands.shape != (len(m["deal_seeds"]), 4) or plays.shape != (len(hands), 28, 2) or decls.shape != (len(hands),):
            raise ValueError("malformed deal arrays")
        forced = set(m.get("forced_deals", []))
        # A source's (limit+1)th seed cannot enter the combined first `limit`.
        # Avoid building Python objects for the other quarter-million games.
        for i in np.argsort(m["deal_seeds"])[:limit]:
            seed = int(m["deal_seeds"][i])
            records.append(dict(seed=seed, hands=hands[i].tolist(), decl=int(decls[i]),
                                plays=plays[i].tolist(), coverage_forced=seed in forced))
        provenance.append({"source": str(source), "commit": m.get("commit"), "spec": m.get("spec"),
                           "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    records.sort(key=lambda r: r["seed"])
    if len({r["seed"] for r in records}) != len(records):
        raise ValueError("overlapping source seeds")
    records = records[:limit]
    seen = set()
    for r in records:
        key = tuple(r["hands"])
        if key in seen:
            raise ValueError("duplicate physical deal")
        seen.add(key)
        r["split"] = partition(r["hands"])
    return records, provenance


def prepare(records):
    # Allocate numeric arrays directly. Nested Python booleans for a whole pour
    # otherwise cost far more memory than the tokens or the model itself.
    n = 4 * len(records)
    rows = {"x": np.empty((n, 68), dtype=np.int64),
            "targets": np.empty((n, 28), dtype=np.int64),
            "legal": np.empty((n, 28, 28), dtype=bool),
            "possible": np.empty((n, 28, 28), dtype=bool),
            "own": np.empty((n, 28), dtype=bool),
            "coverage_forced": np.empty((n, 28), dtype=bool),
            "split": np.empty(n, dtype="<U10"),
            "seed": np.empty(n, dtype=np.int64),
            "viewer": np.empty(n, dtype=np.int64)}
    bits = 1 << np.arange(28)
    for ri, r in enumerate(records):
        hands = list(map(int, r["hands"]))
        if any(h.bit_count() != 7 for h in hands) or (hands[0] | hands[1] | hands[2] | hands[3]) != ALL_TILES:
            raise ValueError("not a complete disjoint deal")
        if len(r["plays"]) != 28:
            raise ValueError("need all 28 recorded plays")
        rules, rec = rules_for(r["decl"]), empty_record(0)
        legal, possible, actors = [], [[] for _ in range(4)], []
        for seat, tile in r["plays"]:
            available = rules.legal(hands[rec.seat] & ~rec.played, rec.led)
            if seat != rec.seat or not 0 <= tile < 28 or not available & (1 << tile):
                raise ValueError("illegal recorded play")
            legal.append(available)
            actors.append(seat)
            for viewer in range(4):
                # The actor's actual hidden hand is not used for other-seat support.
                support = (available if viewer == seat else
                           ALL_TILES & ~(rec.played | hands[viewer] | rec.forbid[seat]))
                possible[viewer].append(support)
            rec = rec.advance(tile, rules)
        for viewer in range(4):
            row = 4 * ri + viewer
            tokens = encode(hands[viewer], viewer, 0, r["decl"], r["plays"])
            rows["x"][row] = tokens[:-1]
            rows["targets"][row] = tokens[QUERIES + 1]
            rows["legal"][row] = (np.asarray(legal)[:, None] & bits) != 0
            rows["possible"][row] = (np.asarray(possible[viewer])[:, None] & bits) != 0
            rows["own"][row] = np.array(actors) == viewer
            rows["coverage_forced"][row] = r["coverage_forced"]
            for k, value in (("split", r["split"]), ("seed", r["seed"]), ("viewer", viewer)):
                rows[k][row] = value
    return rows


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

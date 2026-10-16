# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Pour A's plan is deterministic, forces only pip deals at the stated share,
declares by the book, and a tiny chunk pours both halves with their manifests."""
from __future__ import annotations

import json

import numpy as np

from arena.hand_metrics import menu_choice
from arena.walt_play import _declared_trump
from lem.pour import a as A
from scenario.atlas import deal, load
from scenario.rules import tiles_of


def test_plan_is_deterministic_and_forces_only_pip_deals():
    p1, p2 = A.plan(400001, 400), A.plan(400001, 400)
    assert p1 == p2
    assert [p.seed for p in p1] == list(range(400001, 400401))
    for p in p1:
        assert p.book == A.declare_book(deal(p.seed)[A.BIDDER])
        if p.forced:
            assert p.book <= 6 and p.decl in (A.DOUBLES_TRUMP, A.NO_TRUMP)
        else:
            assert p.decl == p.book
    forced = [p for p in p1 if p.forced]
    assert 0.06 < len(forced) / len(p1) < 0.20          # one in eight, with room for the draw
    assert abs(sum(p.decl == 7 for p in forced) - sum(p.decl == 9 for p in forced)) <= 1
    assert A.plan(400001, 100) == p1[:100]              # a prefix is the same plan


def test_the_book_declares_below_four_doubles_as_the_pours_bidder():
    for s in range(1, 300):
        hand = deal(s)[0]
        doubles = sum(1 for t in tiles_of(hand) if t in (0, 2, 5, 9, 14, 20, 27))
        book = A.declare_book(hand)
        if doubles < 4:
            assert book == _declared_trump(hand)
        else:
            e = menu_choice(tuple(tiles_of(hand)), 0)
            assert book == (_declared_trump(hand) if e is None else e.decl)
        assert book in set(range(7)) | {7, 9}


def test_a_tiny_chunk_pours_both_halves(tmp_path):
    seed = 7_000_001
    # a seed range whose plan has at least one deal in each half
    planned = A.plan(seed, 24, share=3)
    assert any(p.decl > 6 for p in planned) and any(p.decl <= 6 for p in planned)
    rec = A.chunk(seed, 24, tmp_path, pip_spec="k0,n8,F={net}", dtnt_spec="k1,n8,m8", share=3, log=lambda m: None)
    assert rec["counts"]["pip"] + rec["counts"]["dtnt"] == 24
    for half in ("pip", "dtnt"):
        m = json.loads((tmp_path / half / "manifest.json").read_text())
        p = load(tmp_path / half)
        assert m["deals"] == len(p.hands) == rec["counts"][half]
        assert m["pour"] == "A" and set(m["forced_deals"]) <= set(m["deal_seeds"])
        decls = set(int(d) for d in p.decl)
        assert decls <= (set(range(7)) if half == "pip" else {7, 9})
        assert len(p) == rec["positions"][half] > 0
    card = A.card([tmp_path])
    assert card["chunks"] == 1 and card["deals"] == 24
    assert card["forced_dt"] + card["forced_nt"] == sum(p.forced for p in planned)


def test_merge_concatenates_chunks_with_deal_offsets(tmp_path):
    for i, seed in enumerate((7_000_001, 7_000_101)):
        A.chunk(seed, 12, tmp_path / f"c{i}", pip_spec="k0,n8,F={net}", dtnt_spec="k1,n8,m8", share=2, log=lambda m: None)
    r = A.merge(tmp_path / "merged", [tmp_path / "c0", tmp_path / "c1"], log=lambda m: None)
    for half in ("pip", "dtnt"):
        p = load(tmp_path / "merged" / half)
        parts = [load(tmp_path / f"c{i}" / half) for i in range(2)]
        assert len(p.hands) == sum(len(q.hands) for q in parts) == r[half]["deals"]
        assert len(p) == sum(len(q) for q in parts)
        # the second chunk's positions point at the second chunk's deals
        n0 = len(parts[0].hands)
        assert set(p.deal[len(parts[0]):].tolist()) <= set(range(n0, len(p.hands)))
        pos = p.position(len(parts[0]))                   # the second chunk's first position, replayed
        assert pos.hands == tuple(int(h) for h in parts[1].hands[parts[1].deal[0]])
        m = json.loads((tmp_path / "merged" / half / "manifest.json").read_text())
        assert len(m["deal_seeds"]) == len(p.hands) and set(m["forced_deals"]) <= set(m["deal_seeds"])
        assert m["spec"] != "mixed" and len(m["specs"]) == 1
        assert m["declare"].startswith("declares by the book's menu") and f"{len(m['forced_deals']):,} of" in m["declare"]
        if half == "pip":
            assert m["spec"].endswith("merged/net.npz") and (tmp_path / "merged" / "net.npz").exists()


def test_the_pour_honours_the_slots_thread_count(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert A.threads() == 6 and A._walt("k1,n8,m8").threads == 6
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert A.threads() == 0 and A._walt("k1,n8,m8").threads == 0

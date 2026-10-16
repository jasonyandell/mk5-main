# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Bid policy behavior tests."""
import hashlib
import itertools
import json
import random
from pathlib import Path

from forge.oracle.declarations import DOUBLES_TRUMP, GAME_DECL_IDS, NOTRUMP
from forge.oracle.tables import DOMINO_IS_DOUBLE, DOMINOES

from arena.auction import PASS, BidContext, legal_bids
from arena.bidders import Bid30Bidder, HeuristicBidder, MenuBidder, RandomBidder
from arena.engine import ArenaConfig, run_half
from arena.hand_metrics import best_trump
from arena.play import RandomPlay

NAME_TO_ID = {f"{h}-{l}": i for i, (h, l) in enumerate(DOMINOES)}


def hand(*names: str) -> tuple[int, ...]:
    return tuple(NAME_TO_ID[n] for n in names)


def ctx(h: tuple[int, ...], high_bid: int = 0, seat: int = 1) -> BidContext:
    return BidContext(
        hand=h, seat=seat, dealer=0,
        bids=(-1, -1, -1, -1), high_bid=high_bid,
        high_seat=-1 if high_bid == 0 else 0,
        legal=legal_bids(high_bid),
    )


MONSTER = ("5-0", "5-1", "5-2", "5-3", "5-4", "5-5", "6-5")
# A 7-cycle: every pip appears exactly twice, so no suit reaches 3 tiles
GARBAGE = ("1-0", "2-1", "3-2", "4-3", "5-4", "6-5", "6-0")


def test_heuristic_opens_only_enough():
    rng = random.Random(0)
    bidder = HeuristicBidder()
    assert bidder.bid(ctx(hand(*MONSTER)), rng) == 30  # ceiling 42, bids minimum
    assert bidder.bid(ctx(hand(*MONSTER), high_bid=41), rng) == 42


def test_heuristic_passes_garbage_and_mark_bids():
    rng = random.Random(0)
    bidder = HeuristicBidder()
    assert best_trump(hand(*GARBAGE), min_trumps=3) is None
    assert bidder.bid(ctx(hand(*GARBAGE)), rng) == PASS
    # Even a monster never bids marks beyond 42
    assert bidder.bid(ctx(hand(*MONSTER), high_bid=42), rng) == PASS


def test_heuristic_never_exceeds_ceiling():
    rng = random.Random(7)
    bidder = HeuristicBidder()
    for _ in range(300):
        h = tuple(sorted(rng.sample(range(28), 7)))
        for high in (0, 30, 33, 38, 41, 42):
            value = bidder.bid(ctx(h, high_bid=high), rng)
            if value == PASS:
                continue
            assert value in legal_bids(high)
            best = best_trump(h, bidder.min_trumps)
            assert best is not None
            assert value <= best.bid_ceiling


def test_heuristic_declares_a_pip_trump():
    rng = random.Random(0)
    bidder = HeuristicBidder()
    assert bidder.declare(hand(*MONSTER), 30, rng) == 5
    assert 0 <= bidder.declare(hand(*GARBAGE), 30, rng) <= 6


def test_bid30_opens_then_passes():
    rng = random.Random(0)
    bidder = Bid30Bidder()
    assert bidder.bid(ctx(hand(*GARBAGE)), rng) == 30
    assert bidder.bid(ctx(hand(*GARBAGE), high_bid=30), rng) == PASS


def test_random_bidder_stays_legal():
    bidder = RandomBidder(p_bid=1.0)
    rng = random.Random(3)
    for high in (0, 35, 41, 42, 84):
        for _ in range(50):
            value = bidder.bid(ctx(hand(*GARBAGE), high_bid=high), rng)
            assert value == PASS or (value in legal_bids(high) and value <= 42)


# -- The book's Ch. 9 menu (#155).

BOOK_HANDS = Path(__file__).resolve().parent.parent / "silas" / "examples" / "book" / "hands.json"


def _ctx_at(h, high):
    return BidContext(hand=h, seat=1, dealer=0, bids=(-1, -1, -1, -1), high_bid=high,
                      high_seat=-1 if high == 0 else 0, legal=legal_bids(high))


def _bidder_digest(make) -> str:
    """Every bid at eight auction levels and the declaration, over 4000 hands."""
    h = hashlib.sha256()
    rng = random.Random(155)
    for _ in range(4000):
        hand_ = tuple(sorted(rng.sample(range(28), 7)))
        b = make()
        for high in (0, 30, 31, 35, 38, 41, 42, 84):
            h.update(str(b.bid(_ctx_at(hand_, high), random.Random(high))).encode())
        h.update(str(b.declare(hand_, 30, random.Random(0))).encode())
    return h.hexdigest()


def test_the_heuristic_bidder_is_unchanged_by_the_menu():
    """Byte identity: digests recorded at c07032e0, before #155 added the
    menu. Every registered study bids with HeuristicBidder; these pin its
    bids, its declarations and a whole arena run of it."""
    assert _bidder_digest(HeuristicBidder) == "aecaa8526448589cfcf233cdc9a05153dcd57e414cd7f54f2be64c8939bec6f4"
    assert _bidder_digest(lambda: HeuristicBidder(min_trumps=2, caution=5)) == \
        "81a90edf7fa3be00b8d969920aa7d03bcfab9b69d3622060f7467f4bbff937c9"
    records = run_half(n_games=6, a_team=0, cfg=ArenaConfig(marks_to_win=3, base_seed=155),
                       bid_a=HeuristicBidder(), bid_b=HeuristicBidder(caution=5),
                       play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2))
    assert hashlib.sha256(repr(records).encode()).hexdigest() == \
        "fabee9c9d834417a9fea8b8cc76c916c582a84b52916a8aeb9cb9c9651dc7f67"
    assert not hasattr(HeuristicBidder(), "declarations")  # the pip trumps, as every arena bidder


def test_the_menu_is_the_heuristic_below_four_doubles():
    rng = random.Random(9)
    seen = 0
    for _ in range(3000):
        h = tuple(sorted(rng.sample(range(28), 7)))
        if sum(DOMINO_IS_DOUBLE[d] for d in h) >= 4:
            continue
        seen += 1
        for caution in (0, 5):
            a, b = HeuristicBidder(caution=caution), MenuBidder(caution=caution)
            for high in (0, 30, 35, 41):
                assert a.bid(_ctx_at(h, high), rng) == b.bid(_ctx_at(h, high), rng)
            assert a.declare(h, 30, rng) == b.declare(h, 30, rng)
    assert seen > 2500


def _book_bidder_hands():
    names = {}
    for i, (hi, lo) in enumerate(DOMINOES):
        names[f"{hi}-{lo}"] = names[f"{lo}-{hi}"] = i
    out = {}
    for e in json.loads(BOOK_HANDS.read_text())["examples"]:
        for s in e.get("seats", []):
            if s["role"] == "bidder" and None not in s["tiles"]:
                out[e["id"]] = (tuple(names[t] for t in s["tiles"]), e.get("declaration"))
    return out


def test_the_menu_declares_the_books_ch9_hands():
    """Hands 23 and 24 at doubles trump, Hand 25 at no trump (W42-p076-a,
    W42-p078-d, W42-p081-a); every other book hand with four doubles or
    more and a stated pip trump, at that trump, or (Hands 26, 27, where the
    arena finds no suit of three) at no Ch. 9 declaration either."""
    hands = _book_bidder_hands()
    bidder, rng = MenuBidder(), random.Random(0)
    assert bidder.declare(hands["hand-23"][0], 30, rng) == DOUBLES_TRUMP
    assert bidder.declare(hands["hand-24"][0], 30, rng) == DOUBLES_TRUMP
    assert bidder.declare(hands["hand-25"][0], 30, rng) == NOTRUMP
    # Hand 25 has no suit of three: the heuristic bidder passes it, the menu bids it.
    assert HeuristicBidder().bid(ctx(hands["hand-25"][0]), rng) == PASS
    assert bidder.bid(ctx(hands["hand-25"][0]), rng) == 30
    assert bidder.bid(ctx(hands["hand-24"][0]), rng) == 30
    for name in ("hand-2", "hand-9", "hand-13", "hand-21", "hand-26", "hand-27"):
        tiles, decl = hands[name]
        assert sum(DOMINO_IS_DOUBLE[d] for d in tiles) >= 4
        got = bidder.declare(tiles, 30, rng)
        assert got not in (DOUBLES_TRUMP, NOTRUMP), name
        if name not in ("hand-26", "hand-27"):
            assert f"P{got}" == decl, name


def test_the_menu_names_its_declarations():
    assert MenuBidder().declarations == GAME_DECL_IDS
    assert "MenuBidder" in repr(MenuBidder())


def _fnv1a(data: bytes) -> int:
    h = 0xCBF2_9CE4_8422_2325
    for b in data:
        h = ((h ^ b) * 0x0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    return h


def menu_lines() -> list[str]:
    """MenuBidder on every hand of four doubles or more, in increasing order
    of its bits (tile h-l is bit h(h+1)/2 + l, Silas's order): the call
    opening the auction, over a 31 and over a 36, and the declaration."""
    doubles = [i for i in range(28) if DOMINO_IS_DOUBLE[i]]
    others = [i for i in range(28) if not DOMINO_IS_DOUBLE[i]]
    hands = sorted(
        sum(1 << t for t in ds + os)
        for k in range(4, 8)
        for ds in itertools.combinations(doubles, k)
        for os in itertools.combinations(others, 7 - k)
    )
    name = {DOUBLES_TRUMP: "DT", NOTRUMP: "NT"}
    bidder, rng, lines = MenuBidder(), random.Random(0), []
    for bits in hands:
        h = tuple(t for t in range(28) if bits >> t & 1)
        calls = [bidder.bid(_ctx_at(h, high), rng) for high in (0, 31, 36)]
        decl = bidder.declare(h, 30, rng)
        lines.append(f"{bits:07x} " + " ".join(str(c) if c else "pass" for c in calls)
                     + " " + name.get(decl, f"P{decl}"))
    return lines


# The digest of `menu_lines`; silas/tests/bidding.rs pins the same number for
# silas/examples/book/bidding/menu.silas.
MENU_DIGEST = 0x8308_F0F1_62A2_DC1B


def test_the_menu_digest_on_every_hand_of_four_doubles():
    lines = menu_lines()
    assert len(lines) == 51_108
    assert _fnv1a("\n".join(lines).encode()) == MENU_DIGEST

# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The arena's auction, written down for Silas (silas/tests/data/).

Silas's auction (`silas/src/auction.rs`) and its `bid-only-enough` policy are
held to the arena by two fixtures this file writes from the arena's own code
and checks: the legal calls after every legal prefix of an auction
(`auction.legal_bids`), and every call and declaration of `HeuristicBidder`
over 400 seeded deals at every shaker, with the forced bid where the hand
passes out. The Rust tests read the same files, so the arena and Silas agree
call for call.

To rewrite the fixtures after a deliberate change to the arena:

    python -u -c "import arena.test_silas_fixtures as t; t.write()"
"""
from __future__ import annotations

import random
from pathlib import Path

from forge.oracle.tables import DOMINOES

from arena.auction import PASS, legal_bids, run_auction
from arena.bidders import HeuristicBidder

DATA = Path(__file__).resolve().parent.parent / "silas" / "tests" / "data"
LEGAL = DATA / "auction-legal.txt"
HEURISTIC = DATA / "heuristic-bidder.txt"
DEALS, SEED = 400, 20261013


def call(value: int) -> str:
    return "pass" if value == PASS else str(value)


def legal_table() -> str:
    """Every legal prefix of fewer than four calls, and the calls legal next."""
    lines = [
        "# The calls legal after every legal prefix of an auction under the arena's",
        "# rules (arena/auction.py legal_bids: no marks cap, no forced bid). A line is",
        "# the prefix in calling order (- for none), a colon, and the legal calls.",
        "# Written and checked by arena/test_silas_fixtures.py; read by",
        "# silas/tests/auction.rs and src/tests/rules/bidding/silas-auction-fixture.test.ts.",
    ]

    def walk(prefix: list[int], high: int) -> None:
        if len(prefix) == 4:
            return
        legal = (PASS,) + legal_bids(high)
        lines.append(f"{' '.join(map(call, prefix)) or '-'}: {' '.join(map(call, legal))}")
        for value in legal:
            walk(prefix + [value], high if value == PASS else value)

    walk([], 0)
    return "\n".join(lines) + "\n"


def tile(i: int) -> str:
    high, low = DOMINOES[i]
    return f"{high}{low}"


def heuristic_table() -> str:
    """HeuristicBidder at all four seats on seeded deals, every shaker."""
    lines = [
        "# HeuristicBidder() (min_trumps 3, caution 0) at all four seats of 400 deals",
        f"# (random.Random({SEED}) shuffles), under each shaker. A line is the deal,",
        "# seat 0's seven tiles first (a tile 'hl' is high pip, low pip), then per",
        "# shaker 0..3 the four calls in calling order and the declared pip, or 'out'",
        "# for a hand passed out and then the forced shaker's declared pip.",
        "# Written and checked by arena/test_silas_fixtures.py; read by silas/tests/bidding.rs.",
    ]
    rng = random.Random(SEED)
    bidder = HeuristicBidder()
    policies = (bidder, bidder, bidder, bidder)
    for _ in range(DEALS):
        deck = list(range(28))
        rng.shuffle(deck)
        hands = tuple(tuple(sorted(deck[7 * s:7 * s + 7])) for s in range(4))
        cells = []
        for dealer in range(4):
            order = [(dealer + i) % 4 for i in (1, 2, 3, 0)]
            result = run_auction(hands, dealer, policies, random.Random(0))
            if result is None:
                forced = run_auction(hands, dealer, policies, random.Random(0), force_shaker=True)
                assert forced is not None and forced.forced and forced.winner == dealer
                cells.append(f"pass pass pass pass out {forced.decl_id}")
            else:
                calls = " ".join(call(result.bids[s]) for s in order)
                cells.append(f"{calls} {result.decl_id}")
        deal = " ".join(tile(i) for h in hands for i in h)
        lines.append(f"{deal} | {' | '.join(cells)}")
    return "\n".join(lines) + "\n"


def write() -> None:
    LEGAL.write_text(legal_table())
    HEURISTIC.write_text(heuristic_table())


def test_the_legal_calls_fixture_is_the_arenas():
    assert LEGAL.read_text() == legal_table()


def test_the_heuristic_bidder_fixture_is_the_arenas():
    assert HEURISTIC.read_text() == heuristic_table()

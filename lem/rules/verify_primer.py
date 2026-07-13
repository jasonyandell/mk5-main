#!/usr/bin/env python3
"""Verify every testable claim in lem/rules/primer.md against the game engine.

Usage:
    python -m lem.rules.verify_primer
"""

from __future__ import annotations

import sys

from forge.oracle.declarations import (
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    N_DECLS,
    NOTRUMP,
    PIP_TRUMP_IDS,
    has_trump_power,
)
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    N_DOMINOES,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
    resolve_trick,
    trick_rank,
)


def pips_to_id(high: int, low: int) -> int:
    hi, lo = max(high, low), min(high, low)
    return hi * (hi + 1) // 2 + lo


passed = 0
failed = 0


def check(description: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")


def main() -> None:
    print("=== Primer Fact Verification ===\n")

    # --- Deck facts ---
    print("[Deck]")
    check("28 dominoes in double-six set", N_DOMINOES == 28)
    check("7 dominoes per player × 4 = 28", 4 * 7 == N_DOMINOES)

    # --- Count domino facts ---
    print("\n[Count dominoes]")
    check("5-5 is worth 10 count", DOMINO_COUNT_POINTS[pips_to_id(5, 5)] == 10)
    check("6-4 is worth 10 count", DOMINO_COUNT_POINTS[pips_to_id(6, 4)] == 10)
    check("5-0 is worth 5 count", DOMINO_COUNT_POINTS[pips_to_id(5, 0)] == 5)
    check("4-1 is worth 5 count", DOMINO_COUNT_POINTS[pips_to_id(4, 1)] == 5)
    check("3-2 is worth 5 count", DOMINO_COUNT_POINTS[pips_to_id(3, 2)] == 5)

    count_ids = {pips_to_id(5, 5), pips_to_id(6, 4), pips_to_id(5, 0),
                 pips_to_id(4, 1), pips_to_id(3, 2)}
    check("Exactly 5 count dominoes",
          sum(1 for pts in DOMINO_COUNT_POINTS if pts > 0) == 5)
    check("All 23 non-count dominoes are worth 0",
          all(DOMINO_COUNT_POINTS[i] == 0 for i in range(N_DOMINOES) if i not in count_ids))
    check("Total count = 35", sum(DOMINO_COUNT_POINTS) == 35)
    check("Total hand = 42 (35 count + 7 tricks)", sum(DOMINO_COUNT_POINTS) + 7 == 42)

    # --- Trump declaration facts ---
    print("\n[Declarations]")
    check("10 total declarations", N_DECLS == 10)
    check("7 pip-suit trumps (0-6)", set(PIP_TRUMP_IDS) == set(range(7)))
    check("Doubles-as-trump exists", DOUBLES_TRUMP == 7)
    check("No-trump exists", NOTRUMP == 9)

    # --- Pip trump membership ---
    print("\n[Trump membership - pip suits]")
    # "When fives are trump, 5-5 is the highest trump"
    fives_trump = 5
    fives_trumps = [i for i in range(N_DOMINOES) if is_in_called_suit(i, fives_trump)]
    check("With fives trump, 5-5 is a trump",
          pips_to_id(5, 5) in fives_trumps)
    # 5-5 should have highest trick rank among all trumps
    five_five_rank = trick_rank(pips_to_id(5, 5), 7, fives_trump)
    check("5-5 is the highest fives-trump",
          all(trick_rank(t, 7, fives_trump) <= five_five_rank for t in fives_trumps))

    # "When a pip suit S is trump, every domino containing pip S is trump"
    for s in range(7):
        trumps = [i for i in range(N_DOMINOES) if is_in_called_suit(i, s)]
        containing_s = [i for i in range(N_DOMINOES)
                        if DOMINO_HIGH[i] == s or DOMINO_LOW[i] == s]
        check(f"Pip {s} trump: all dominoes containing {s} are trump",
              set(trumps) == set(containing_s))
        # Double is highest
        double_id = pips_to_id(s, s)
        double_rank = trick_rank(double_id, 7, s)
        check(f"Pip {s} trump: {s}-{s} is highest",
              all(trick_rank(t, 7, s) <= double_rank for t in trumps))

    # "When a pip suit is trump, 6-4 is trump iff that suit is sixes or fours"
    print("\n[6-4 trump membership]")
    id_64 = pips_to_id(6, 4)
    for s in range(7):
        is_trump = is_in_called_suit(id_64, s)
        expected = s in (4, 6)
        check(f"6-4 is trump when {s}s trump: {is_trump} == {expected}",
              is_trump == expected)

    # --- Doubles trump ---
    print("\n[Doubles trump]")
    doubles_trumps = [i for i in range(N_DOMINOES) if is_in_called_suit(i, DOUBLES_TRUMP)]
    actual_doubles = [i for i in range(N_DOMINOES) if DOMINO_IS_DOUBLE[i]]
    check("Doubles trump: exactly 7 doubles are trump",
          len(doubles_trumps) == 7)
    check("Doubles trump: trump set == double set",
          set(doubles_trumps) == set(actual_doubles))

    # "Ranked 6-6 high through 0-0 low"
    ranks = [(trick_rank(d, 7, DOUBLES_TRUMP), d) for d in doubles_trumps]
    ranks.sort(reverse=True)
    check("Doubles trump ranking: 6-6 is highest",
          ranks[0][1] == pips_to_id(6, 6))
    check("Doubles trump ranking: 0-0 is lowest",
          ranks[-1][1] == pips_to_id(0, 0))

    # "When doubles are trump, a double is not a member of its pip suit"
    # e.g., 5-5 shouldn't follow a fives lead under doubles trump
    check("5-5 can't follow fives-led under doubles trump",
          not can_follow(pips_to_id(5, 5), 5, DOUBLES_TRUMP))

    # --- Led suit ---
    print("\n[Led suit]")
    # "With fours as trump, leading 4-2 leads trump"
    check("4-2 leads trump when fours trump",
          led_suit_for_lead_domino(pips_to_id(4, 2), 4) == 7)
    # "With twos as trump, leading 5-3 leads fives"
    check("5-3 leads fives when twos trump",
          led_suit_for_lead_domino(pips_to_id(5, 3), 2) == 5)
    # Non-trump lead: higher pip determines suit
    check("6-3 leads sixes under no-trump",
          led_suit_for_lead_domino(pips_to_id(6, 3), NOTRUMP) == 6)
    # Trump lead under doubles
    check("5-5 leads trump under doubles-trump",
          led_suit_for_lead_domino(pips_to_id(5, 5), DOUBLES_TRUMP) == 7)
    check("6-3 leads sixes under doubles-trump",
          led_suit_for_lead_domino(pips_to_id(6, 3), DOUBLES_TRUMP) == 6)

    # --- Following suit ---
    print("\n[Following suit]")
    # A domino in the led suit can follow
    check("5-3 can follow fives-led (no trump)",
          can_follow(pips_to_id(5, 3), 5, NOTRUMP))
    # A domino not in led suit can't follow
    check("4-2 can't follow fives-led (no trump)",
          not can_follow(pips_to_id(4, 2), 5, NOTRUMP))
    # Trump can't follow non-trump led suit
    check("5-3 can't follow sixes-led when fives trump (5-3 is trump)",
          not can_follow(pips_to_id(5, 3), 6, 5))

    # --- Trick winning ---
    print("\n[Trick winning]")
    # Highest trump wins
    outcome = resolve_trick(
        pips_to_id(6, 6),  # lead: 6-6 (not trump, leads sixes)
        (pips_to_id(6, 6), pips_to_id(6, 3), pips_to_id(5, 1), pips_to_id(6, 2)),
        5,  # fives trump
    )
    check("5-1 (trump) beats non-trump sixes when fives trump",
          outcome.winner_offset == 2)

    # No trump: highest of led suit wins
    outcome2 = resolve_trick(
        pips_to_id(6, 3),
        (pips_to_id(6, 3), pips_to_id(6, 1), pips_to_id(4, 2), pips_to_id(6, 6)),
        NOTRUMP,
    )
    check("Under no-trump, 6-6 (highest six) wins sixes-led trick",
          outcome2.winner_offset == 3)

    # Off-suit can't win
    outcome3 = resolve_trick(
        pips_to_id(5, 3),
        (pips_to_id(5, 3), pips_to_id(6, 6), pips_to_id(5, 1), pips_to_id(5, 5)),
        NOTRUMP,
    )
    check("Under no-trump, 6-6 off-suit can't beat fives-led",
          outcome3.winner_offset == 3)  # 5-5 highest five

    # Trick points = count + 1
    check("Trick with 5-5 scores 10+1=11",
          outcome3.points == 11)

    # --- Summary ---
    print(f"\n{'=' * 40}")
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All primer facts verified against the engine.")


if __name__ == "__main__":
    main()

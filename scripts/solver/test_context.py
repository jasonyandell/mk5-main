#!/usr/bin/env python3
"""
Tests for SeedContext - verifies correctness of precomputed tables.

Tests:
1. L table matches deal_with_seed output
2. LOCAL_FOLLOW produces correct follow masks
3. TRICK_WINNER correctly identifies winners
4. TRICK_POINTS uses correct range 1-31
5. Specific trick scenarios work correctly
"""

import torch
import numpy as np

from rng import deal_with_seed
from tables import EFFECTIVE_SUIT, SUIT_MASK, HAS_POWER, DOMINO_PIPS
from context import build_context, _rank_in_trick, _get_domino_points


def test_l_table_matches_deal():
    """Test that L table matches deal_with_seed output exactly."""
    print("Testing L table matches deal_with_seed...")

    for seed in [1, 42, 12345, 999999]:
        ctx = build_context(seed, decl_id=0)
        expected = deal_with_seed(seed)

        # Check shape
        assert ctx.L.shape == (4, 7), f"L shape mismatch: {ctx.L.shape}"

        # Check values
        for p in range(4):
            for i in range(7):
                actual = int(ctx.L[p, i])
                exp = int(expected[p, i])
                assert actual == exp, f"L[{p},{i}] seed={seed}: expected {exp}, got {actual}"

    print("  PASS: L table matches deal_with_seed")


def test_local_follow_shape():
    """Test LOCAL_FOLLOW has correct shape."""
    print("Testing LOCAL_FOLLOW shape...")

    ctx = build_context(seed=42, decl_id=3)
    assert ctx.LOCAL_FOLLOW.shape == (112,), f"LOCAL_FOLLOW shape: {ctx.LOCAL_FOLLOW.shape}"

    print("  PASS: LOCAL_FOLLOW has shape (112,)")


def test_local_follow_correctness():
    """Test LOCAL_FOLLOW produces correct follow masks."""
    print("Testing LOCAL_FOLLOW correctness...")

    seed = 42
    decl_id = 3  # 3s are trump

    ctx = build_context(seed, decl_id)
    L = deal_with_seed(seed)
    absorption_id = decl_id

    errors = 0
    for leader in range(4):
        for lead_local_idx in range(7):
            # Get led suit
            global_id = L[leader, lead_local_idx]
            led_suit = EFFECTIVE_SUIT[global_id, absorption_id]

            for follower_offset in range(4):
                follower = (leader + follower_offset) % 4

                # Compute expected mask manually
                expected_mask = 0
                for local_idx in range(7):
                    follower_global_id = L[follower, local_idx]
                    suit_mask = SUIT_MASK[absorption_id, led_suit]
                    can_follow = bool((suit_mask >> follower_global_id) & 1)
                    if can_follow:
                        expected_mask |= (1 << local_idx)

                # Get actual mask
                idx = leader * 28 + lead_local_idx * 4 + follower_offset
                actual_mask = int(ctx.LOCAL_FOLLOW[idx])

                if actual_mask != expected_mask:
                    print(f"  LOCAL_FOLLOW mismatch at leader={leader}, lead_idx={lead_local_idx}, offset={follower_offset}")
                    print(f"    Expected: {expected_mask:07b}, Actual: {actual_mask:07b}")
                    errors += 1

    if errors == 0:
        print("  PASS: LOCAL_FOLLOW correctness verified")
    else:
        print(f"  FAIL: {errors} errors in LOCAL_FOLLOW")

    return errors


def test_trick_winner_shape():
    """Test TRICK_WINNER has correct shape."""
    print("Testing TRICK_WINNER shape...")

    ctx = build_context(seed=42, decl_id=5)
    assert ctx.TRICK_WINNER.shape == (9604,), f"TRICK_WINNER shape: {ctx.TRICK_WINNER.shape}"

    print("  PASS: TRICK_WINNER has shape (9604,)")


def test_trick_points_shape():
    """Test TRICK_POINTS has correct shape."""
    print("Testing TRICK_POINTS shape...")

    ctx = build_context(seed=42, decl_id=5)
    assert ctx.TRICK_POINTS.shape == (9604,), f"TRICK_POINTS shape: {ctx.TRICK_POINTS.shape}"

    print("  PASS: TRICK_POINTS has shape (9604,)")


def test_trick_points_range():
    """Test TRICK_POINTS values are in range 1-31."""
    print("Testing TRICK_POINTS range 1-31...")

    # Test multiple seeds and declarations
    for seed in [42, 12345]:
        for decl_id in [0, 3, 6]:
            ctx = build_context(seed, decl_id)
            min_pts = int(ctx.TRICK_POINTS.min())
            max_pts = int(ctx.TRICK_POINTS.max())

            # Minimum: 4 non-count dominoes + 1 trick = 1 point
            # Maximum: 5-5 (10) + 6-4 (10) + 5-0 (5) + 4-1 (5) + 1 trick = 31 points
            assert min_pts >= 1, f"seed={seed} decl={decl_id}: min points {min_pts} < 1"
            assert max_pts <= 31, f"seed={seed} decl={decl_id}: max points {max_pts} > 31"

    print("  PASS: TRICK_POINTS in range 1-31")


def test_trick_winner_range():
    """Test TRICK_WINNER values are in range 0-3."""
    print("Testing TRICK_WINNER range 0-3...")

    for seed in [42, 12345]:
        for decl_id in [0, 3, 6]:
            ctx = build_context(seed, decl_id)
            min_winner = int(ctx.TRICK_WINNER.min())
            max_winner = int(ctx.TRICK_WINNER.max())

            assert min_winner >= 0, f"seed={seed} decl={decl_id}: min winner {min_winner} < 0"
            assert max_winner <= 3, f"seed={seed} decl={decl_id}: max winner {max_winner} > 3"

    print("  PASS: TRICK_WINNER in range 0-3")


def test_rank_in_trick_tiers():
    """Test _rank_in_trick returns correct tiers."""
    print("Testing _rank_in_trick tier logic...")

    # For 3s trump (absorption_id=3, power_id=3):
    # - Dominoes with 3 have power (tier 2)
    # - Dominoes that follow suit without power (tier 1)
    # - Dominoes that don't follow (tier 0)

    # Domino IDs (triangular encoding: id = hi*(hi+1)//2 + lo):
    # 3-3 (id=9): lo=3, hi=3 -> id = 3*4//2 + 3 = 6 + 3 = 9
    # 0-3 (id=6): lo=0, hi=3 -> id = 3*4//2 + 0 = 6
    # 6-6 (id=27): lo=6, hi=6 -> id = 6*7//2 + 6 = 21 + 6 = 27
    # 0-0 (id=0): lo=0, hi=0 -> id = 0

    # 3-3 (id=9): has power, should be tier 2
    # Led suit 7 (trump), so 3-3 follows
    rank_33 = _rank_in_trick(9, led_suit=7, absorption_id=3, power_id=3)
    tier_33 = rank_33 >> 4
    assert tier_33 == 2, f"3-3 should be tier 2, got {tier_33}"

    # 0-3 (id=6): has power (contains 3), tier 2
    rank_03 = _rank_in_trick(6, led_suit=7, absorption_id=3, power_id=3)
    tier_03 = rank_03 >> 4
    assert tier_03 == 2, f"0-3 should be tier 2, got {tier_03}"

    # 6-6 (id=27): no power, led suit 6, follows suit, tier 1
    rank_66 = _rank_in_trick(27, led_suit=6, absorption_id=3, power_id=3)
    tier_66 = rank_66 >> 4
    assert tier_66 == 1, f"6-6 should be tier 1 when following suit 6, got {tier_66}"

    # 0-0 (id=0): no power, led suit 6, doesn't follow, tier 0
    rank_00 = _rank_in_trick(0, led_suit=6, absorption_id=3, power_id=3)
    assert rank_00 == 0, f"0-0 should be tier 0 when sloughing on suit 6, got {rank_00}"

    print("  PASS: _rank_in_trick tier logic verified")


def test_rank_in_trick_doubles_ranking():
    """Test doubles ranking in different scenarios."""
    print("Testing doubles ranking...")

    # For pip trump (not doubles trump), doubles rank 14 (highest)
    # 6-6 in suit 6 when 3s are trump
    rank_66_pip = _rank_in_trick(27, led_suit=6, absorption_id=3, power_id=3)
    rank_in_tier = rank_66_pip & 0x0F
    assert rank_in_tier == 14, f"6-6 should rank 14 in pip trump, got {rank_in_tier}"

    # For doubles trump (absorption_id=7), doubles rank by pip value
    # 6-6 when doubles are trump -> rank = 6 (highest double)
    rank_66_dbl = _rank_in_trick(27, led_suit=7, absorption_id=7, power_id=7)
    rank_in_tier_dbl = rank_66_dbl & 0x0F
    assert rank_in_tier_dbl == 6, f"6-6 in doubles trump should rank 6, got {rank_in_tier_dbl}"

    # 0-0 when doubles are trump -> rank = 0 (lowest double)
    rank_00_dbl = _rank_in_trick(0, led_suit=7, absorption_id=7, power_id=7)
    rank_in_tier_00 = rank_00_dbl & 0x0F
    assert rank_in_tier_00 == 0, f"0-0 in doubles trump should rank 0, got {rank_in_tier_00}"

    print("  PASS: Doubles ranking verified")


def test_get_domino_points():
    """Test _get_domino_points returns correct values."""
    print("Testing _get_domino_points...")

    # 5-5 (id=25) = 10 points
    assert _get_domino_points(25) == 10, "5-5 should be 10 points"

    # 6-4 (id=25 is wrong, let me find it)
    # Triangular: id = hi*(hi+1)//2 + lo
    # 6-4: hi=6, lo=4 -> id = 6*7//2 + 4 = 21 + 4 = 25... wait that's 5-5
    # Let me recalculate: 5-5 is hi=5, lo=5 -> id = 5*6//2 + 5 = 15 + 5 = 20? No wait
    # Actually: (0,0)=0, (0,1)=1, (1,1)=2, (0,2)=3, (1,2)=4, (2,2)=5...
    # hi=5, lo=5: id = 5*(5+1)//2 + 5 = 15 + 5 = 20

    # Verify from DOMINO_PIPS
    for d in range(28):
        lo, hi = DOMINO_PIPS[d]
        if lo == 5 and hi == 5:
            assert _get_domino_points(d) == 10, f"5-5 (id={d}) should be 10"
        if lo == 4 and hi == 6:
            assert _get_domino_points(d) == 10, f"6-4 (id={d}) should be 10"
        if lo + hi == 5 and lo != hi:  # 5-0, 4-1, 3-2
            assert _get_domino_points(d) == 5, f"{lo}-{hi} (id={d}) should be 5"

    print("  PASS: _get_domino_points verified")


def test_specific_trick_scenario():
    """Test a specific trick to verify winner and points calculation."""
    print("Testing specific trick scenario...")

    # Use a known seed
    seed = 42
    decl_id = 3  # 3s are trump

    ctx = build_context(seed, decl_id)
    L = deal_with_seed(seed)

    # Pick a specific trick: leader=0, all players play their first domino
    leader = 0
    p0, p1, p2, p3 = 0, 0, 0, 0

    # Get global IDs
    g0 = L[0, p0]
    g1 = L[1, p1]
    g2 = L[2, p2]
    g3 = L[3, p3]

    # Compute expected winner manually
    led_suit = EFFECTIVE_SUIT[g0, decl_id]
    ranks = [
        _rank_in_trick(g0, led_suit, decl_id, decl_id),
        _rank_in_trick(g1, led_suit, decl_id, decl_id),
        _rank_in_trick(g2, led_suit, decl_id, decl_id),
        _rank_in_trick(g3, led_suit, decl_id, decl_id),
    ]
    expected_winner = ranks.index(max(ranks))

    # Compute expected points manually
    expected_points = (
        _get_domino_points(g0) +
        _get_domino_points(g1) +
        _get_domino_points(g2) +
        _get_domino_points(g3) +
        1  # trick point
    )

    # Get actual from tables
    idx = leader * 2401 + p0 * 343 + p1 * 49 + p2 * 7 + p3
    actual_winner = int(ctx.TRICK_WINNER[idx])
    actual_points = int(ctx.TRICK_POINTS[idx])

    assert actual_winner == expected_winner, f"Winner mismatch: expected {expected_winner}, got {actual_winner}"
    assert actual_points == expected_points, f"Points mismatch: expected {expected_points}, got {actual_points}"

    print(f"  Trick: {DOMINO_PIPS[g0]} leads, {DOMINO_PIPS[g1]}, {DOMINO_PIPS[g2]}, {DOMINO_PIPS[g3]}")
    print(f"  Winner offset: {actual_winner}, Points: {actual_points}")
    print("  PASS: Specific trick scenario verified")


def test_trump_beats_suit():
    """Test that trump dominoes beat non-trump even if non-trump has higher rank."""
    print("Testing trump beats suit...")

    # Find a deal where we can test trump vs suit
    seed = 100
    decl_id = 0  # 0s are trump

    ctx = build_context(seed, decl_id)
    L = deal_with_seed(seed)

    # We need:
    # - A non-trump domino leading
    # - A trump domino following

    # Let's manually construct a scenario
    # Find indices where player 0 has non-trump and player 1 has trump
    absorption_id = decl_id

    for p0 in range(7):
        g0 = L[0, p0]
        led_suit = EFFECTIVE_SUIT[g0, absorption_id]

        # Skip if g0 is trump (led suit = 7)
        if led_suit == 7:
            continue

        for p1 in range(7):
            g1 = L[1, p1]
            # Check if g1 is trump
            if HAS_POWER[g1, decl_id]:
                # Found: g0 is non-trump, g1 is trump

                # g1 should win (tier 2 beats tier 1)
                rank0 = _rank_in_trick(g0, led_suit, absorption_id, decl_id)
                rank1 = _rank_in_trick(g1, led_suit, absorption_id, decl_id)

                assert rank1 > rank0, f"Trump should beat non-trump: rank0={rank0}, rank1={rank1}"

                # Check the table
                # Use same p2, p3 for simplicity
                for p2 in range(7):
                    for p3 in range(7):
                        idx = 0 * 2401 + p0 * 343 + p1 * 49 + p2 * 7 + p3

                        # Winner should be 1 (player 1 with trump) unless p2 or p3 have higher trump
                        actual_winner = int(ctx.TRICK_WINNER[idx])

                        # Verify p2, p3 aren't higher
                        g2 = L[2, p2]
                        g3 = L[3, p3]
                        rank2 = _rank_in_trick(g2, led_suit, absorption_id, decl_id)
                        rank3 = _rank_in_trick(g3, led_suit, absorption_id, decl_id)

                        expected_winner = [rank0, rank1, rank2, rank3].index(max([rank0, rank1, rank2, rank3]))
                        assert actual_winner == expected_winner, f"Winner mismatch at idx={idx}"

                print(f"  Verified: {DOMINO_PIPS[g0]} (rank {rank0}) vs {DOMINO_PIPS[g1]} trump (rank {rank1})")
                print("  PASS: Trump beats suit verified")
                return

    print("  SKIP: Could not find suitable test case (no non-trump vs trump in deal)")


def test_decl_id_validation():
    """Test that invalid decl_id raises error."""
    print("Testing decl_id validation...")

    try:
        build_context(seed=42, decl_id=7)
        assert False, "Should have raised ValueError for decl_id=7"
    except ValueError as e:
        assert "0-6" in str(e)

    try:
        build_context(seed=42, decl_id=-1)
        assert False, "Should have raised ValueError for decl_id=-1"
    except ValueError as e:
        assert "0-6" in str(e)

    print("  PASS: Invalid decl_id raises ValueError")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SeedContext Tests")
    print("=" * 60)
    print()

    errors = 0

    test_l_table_matches_deal()
    test_local_follow_shape()
    errors += test_local_follow_correctness()
    test_trick_winner_shape()
    test_trick_points_shape()
    test_trick_points_range()
    test_trick_winner_range()
    test_rank_in_trick_tiers()
    test_rank_in_trick_doubles_ranking()
    test_get_domino_points()
    test_specific_trick_scenario()
    test_trump_beats_suit()
    test_decl_id_validation()

    print()
    print("=" * 60)
    if errors == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"TESTS COMPLETED WITH {errors} ERRORS")
    print("=" * 60)

    return errors


if __name__ == '__main__':
    import sys
    sys.exit(main())

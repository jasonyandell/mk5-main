"""Hand-level feature extraction for bidding analysis.

This module provides unified feature extraction from a player's 7-domino hand.
Previously duplicated across run_11{b,f,g,j,q,s,u}.py - now consolidated here.

Usage:
    from forge.analysis.utils.hand_features import extract_hand_features

    features = extract_hand_features(hand, trump_suit)
"""

from __future__ import annotations

from forge.oracle import tables


def extract_hand_features(hand: list[int], trump_suit: int) -> dict:
    """Extract bidding-relevant features from a 7-domino hand.

    Args:
        hand: List of 7 domino IDs (0-27)
        trump_suit: Trump pip (0-6) from decl_id, or >6 for no trumps

    Returns:
        Dictionary with features:
        - n_doubles: Count of double dominoes (0-7)
        - trump_count: Count of dominoes containing trump pip (0-7)
        - max_suit_length: Maximum count for any single pip (1-7)
        - n_6_high, n_5_high, n_4_high: Counts of dominoes by high pip
        - count_points: Total count points held (0-35)
        - n_count_dominoes: Number of count dominoes held (0-5)
        - total_pips: Sum of all pips on all dominoes
        - has_trump_double: 1 if holds the trump double, 0 otherwise
        - n_voids: Number of pips with 0 dominoes (0-7)
        - n_singletons: Number of pips with exactly 1 domino (0-7)
    """
    # Basic counts
    n_doubles = sum(1 for d in hand if tables.DOMINO_IS_DOUBLE[d])

    # Trump suit length
    if trump_suit <= 6:
        trump_count = sum(1 for d in hand if tables.domino_contains_pip(d, trump_suit))
    else:
        trump_count = 0

    # Pip counts for suit analysis
    pip_counts = [
        sum(1 for d in hand if tables.domino_contains_pip(d, pip))
        for pip in range(7)
    ]
    max_suit_length = max(pip_counts)

    # High dominoes (by highest pip on domino)
    n_6_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 6)
    n_5_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 5)
    n_4_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 4)

    # Count holdings (5-count: 5-0, 4-1, 3-2; 10-count: 5-5, 6-4)
    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand)
    n_count_dominoes = sum(1 for d in hand if tables.DOMINO_COUNT_POINTS[d] > 0)

    # Total pips (crude hand strength measure)
    total_pips = sum(tables.DOMINO_HIGH[d] + tables.DOMINO_LOW[d] for d in hand)

    # Trump-specific features
    has_trump_double = 0
    if trump_suit <= 6:
        has_trump_double = int(
            any(
                tables.DOMINO_IS_DOUBLE[d] and tables.DOMINO_HIGH[d] == trump_suit
                for d in hand
            )
        )

    # Void suits (suits with 0 dominoes)
    n_voids = sum(1 for c in pip_counts if c == 0)

    # Singleton suits
    n_singletons = sum(1 for c in pip_counts if c == 1)

    return {
        'n_doubles': n_doubles,
        'trump_count': trump_count,
        'max_suit_length': max_suit_length,
        'n_6_high': n_6_high,
        'n_5_high': n_5_high,
        'n_4_high': n_4_high,
        'count_points': count_points,
        'n_count_dominoes': n_count_dominoes,
        'total_pips': total_pips,
        'has_trump_double': has_trump_double,
        'n_voids': n_voids,
        'n_singletons': n_singletons,
    }


# Feature names for consistent ordering in DataFrames
HAND_FEATURE_NAMES = [
    'n_doubles',
    'trump_count',
    'max_suit_length',
    'n_6_high',
    'n_5_high',
    'n_4_high',
    'count_points',
    'n_count_dominoes',
    'total_pips',
    'has_trump_double',
    'n_voids',
    'n_singletons',
]


# Regression features (subset used in 11f, 11s)
REGRESSION_FEATURES = [
    'n_doubles',
    'trump_count',
    'n_6_high',
    'n_5_high',
    'count_points',
    'total_pips',
    'has_trump_double',
    'max_suit_length',
    'n_voids',
    'n_singletons',
]

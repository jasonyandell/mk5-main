"""Information-theoretic and compression metrics.

This module provides functions for analyzing the structure of V values:
- entropy_bits(): Shannon entropy of a distribution
- conditional_entropy(): H(V|feature)
- mutual_information(): I(V; feature)
- lzma_ratio(): Compressibility via LZMA
"""

from __future__ import annotations

import lzma
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def entropy_bits(values: np.ndarray, base: float = 2.0) -> float:
    """
    Compute Shannon entropy of a discrete distribution.

    Args:
        values: (N,) array of discrete values
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy in specified base (bits if base=2)
    """
    # Count unique values
    unique, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)

    # H = -sum(p * log(p))
    # Handle p=0 case (0 * log(0) = 0)
    probs = probs[probs > 0]
    if base == 2.0:
        return -np.sum(probs * np.log2(probs))
    else:
        return -np.sum(probs * np.log(probs)) / np.log(base)


def conditional_entropy(
    values: np.ndarray,
    feature: np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Compute conditional entropy H(V|feature).

    This is the expected entropy of V given the feature value:
    H(V|F) = sum_f P(F=f) * H(V|F=f)

    Args:
        values: (N,) target values (e.g., V)
        feature: (N,) conditioning feature
        base: Logarithm base

    Returns:
        Conditional entropy H(V|feature)
    """
    n = len(values)
    unique_features = np.unique(feature)

    total_entropy = 0.0
    for f in unique_features:
        mask = feature == f
        p_f = mask.sum() / n
        if p_f > 0:
            h_given_f = entropy_bits(values[mask], base)
            total_entropy += p_f * h_given_f

    return total_entropy


def mutual_information(
    values: np.ndarray,
    feature: np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Compute mutual information I(V; feature).

    I(V; F) = H(V) - H(V|F)

    This measures how much knowing the feature reduces uncertainty about V.

    Args:
        values: (N,) target values
        feature: (N,) feature
        base: Logarithm base

    Returns:
        Mutual information I(V; feature)
    """
    h_v = entropy_bits(values, base)
    h_v_given_f = conditional_entropy(values, feature, base)
    return h_v - h_v_given_f


def information_gain_ranking(
    values: np.ndarray,
    features: dict[str, np.ndarray],
) -> list[tuple[str, float, float]]:
    """
    Rank features by information gain (mutual information with V).

    Args:
        values: (N,) target values (e.g., V)
        features: Dict mapping feature name to (N,) feature array

    Returns:
        List of (feature_name, mutual_info, conditional_entropy)
        sorted by mutual_info descending
    """
    h_v = entropy_bits(values)
    results = []

    for name, feature in features.items():
        h_cond = conditional_entropy(values, feature)
        mi = h_v - h_cond
        results.append((name, mi, h_cond))

    # Sort by mutual information descending
    results.sort(key=lambda x: -x[1])
    return results


def lzma_ratio(data: bytes, preset: int = 6) -> float:
    """
    Compute LZMA compression ratio.

    Lower ratio = more compressible = more structure.

    Args:
        data: Bytes to compress
        preset: LZMA compression preset (0-9, higher = slower but better)

    Returns:
        Ratio of compressed to uncompressed size (0-1)
    """
    compressed = lzma.compress(data, preset=preset)
    return len(compressed) / len(data)


def serialize_v_by_depth(
    states: np.ndarray,
    values: np.ndarray,
) -> bytes:
    """
    Serialize V values ordered by depth (remaining dominoes).

    Args:
        states: (N,) int64 packed states
        values: (N,) int8 V values

    Returns:
        Bytes representation of V in depth order
    """
    from forge.analysis.utils.features import depth

    depths = depth(states)
    order = np.argsort(depths)
    return values[order].tobytes()


def serialize_v_by_state(values: np.ndarray) -> bytes:
    """
    Serialize V values in state index order (as stored).

    Args:
        values: (N,) int8 V values

    Returns:
        Bytes representation
    """
    return values.astype(np.int8).tobytes()


def serialize_v_random(
    values: np.ndarray,
    seed: int = 42,
) -> bytes:
    """
    Serialize V values in random order.

    Args:
        values: (N,) int8 V values
        seed: Random seed for reproducibility

    Returns:
        Bytes representation in random order
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    return values[order].tobytes()


def compression_analysis(
    states: np.ndarray,
    values: np.ndarray,
    preset: int = 6,
) -> dict[str, float]:
    """
    Compare LZMA compression under different orderings.

    Args:
        states: (N,) int64 packed states
        values: (N,) int8 V values
        preset: LZMA preset

    Returns:
        Dict with compression ratios:
        - depth_order: Ordered by remaining dominoes
        - state_order: Original order
        - random_order: Random order (baseline)
    """
    return {
        "depth_order": lzma_ratio(serialize_v_by_depth(states, values), preset),
        "state_order": lzma_ratio(serialize_v_by_state(values), preset),
        "random_order": lzma_ratio(serialize_v_random(values), preset),
    }


def effective_alphabet_size(values: np.ndarray) -> int:
    """
    Count unique values in array.

    Args:
        values: (N,) discrete values

    Returns:
        Number of unique values
    """
    return len(np.unique(values))


def entropy_rate(
    values: np.ndarray,
    context_length: int = 1,
) -> float:
    """
    Estimate entropy rate using n-gram model.

    For context_length=1, this is H(V_i | V_{i-1}).
    Higher context captures longer-range dependencies.

    Args:
        values: (N,) sequence of values
        context_length: Number of previous values to condition on

    Returns:
        Estimated entropy rate in bits
    """
    n = len(values)
    if n <= context_length:
        return entropy_bits(values)

    # Build context -> next value mapping
    from collections import defaultdict
    context_counts: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for i in range(context_length, n):
        context = tuple(values[i - context_length:i])
        next_val = values[i]
        context_counts[context][next_val] += 1

    # Compute weighted average entropy
    total_entropy = 0.0
    total_count = 0

    for context, next_counts in context_counts.items():
        context_total = sum(next_counts.values())
        probs = np.array(list(next_counts.values())) / context_total

        # Entropy for this context
        h = -np.sum(probs * np.log2(probs + 1e-10))
        total_entropy += context_total * h
        total_count += context_total

    return total_entropy / total_count if total_count > 0 else 0.0

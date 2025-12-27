"""
Tests for rng.py - verify Python RNG matches TypeScript implementation.

Run with: python -m pytest scripts/solver/test_rng.py -v
Or: python scripts/solver/test_rng.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rng import (
    lcg_next,
    to_float,
    next_int,
    shuffle_with_seed,
    deal_with_seed,
    domino_to_global_id,
    global_id_to_domino,
    get_domino_values_order,
    SeededRandom,
    DOMINO_VALUES,
)


def test_lcg_next_basic():
    """Test basic LCG next value computation."""
    # seed=1 should give predictable results
    seed1 = lcg_next(1)
    assert seed1 == 16807, f"lcg_next(1) = {seed1}, expected 16807"

    seed2 = lcg_next(seed1)
    assert seed2 == 282475249, f"lcg_next(16807) = {seed2}, expected 282475249"


def test_domino_global_id_roundtrip():
    """Test that domino_to_global_id and global_id_to_domino are inverses."""
    for global_id in range(28):
        lo, hi = global_id_to_domino(global_id)
        reconstructed = domino_to_global_id(lo, hi)
        assert reconstructed == global_id, f"Roundtrip failed for {global_id}"
        assert lo <= hi, f"Invalid domino: lo={lo} > hi={hi}"


def test_domino_values_order():
    """Test that DOMINO_VALUES matches expected order."""
    # First 7: blanks (0,0) to (0,6)
    for i in range(7):
        assert DOMINO_VALUES[i] == (0, i), f"Index {i}: expected (0,{i}), got {DOMINO_VALUES[i]}"

    # Total should be 28
    assert len(DOMINO_VALUES) == 28


def test_domino_values_global_ids():
    """Test the global ID conversion for DOMINO_VALUES order."""
    ids = get_domino_values_order()

    # First in DOMINO_VALUES order: (0,0) -> global_id = 0
    assert ids[0] == 0

    # Second: (0,1) -> global_id = 1
    assert ids[1] == 1

    # (0,6) -> global_id = 6*(6+1)//2 + 0 = 21
    assert ids[6] == 21

    # (1,1) -> global_id = 1*(1+1)//2 + 1 = 2
    assert ids[7] == 2

    # (6,6) -> global_id = 6*(6+1)//2 + 6 = 27
    assert ids[27] == 27


def test_seeded_random_class():
    """Test SeededRandom class matches functional implementation."""
    rng = SeededRandom(42)

    # Get a sequence of values
    values = [rng.next() for _ in range(5)]

    # Verify they're in expected range
    for v in values:
        assert 0.0 <= v < 1.0

    # Verify determinism - same seed gives same sequence
    rng2 = SeededRandom(42)
    values2 = [rng2.next() for _ in range(5)]
    assert values == values2


def test_shuffle_determinism():
    """Test that shuffle is deterministic with same seed."""
    items = list(range(28))

    shuffled1, _ = shuffle_with_seed(items, 42)
    shuffled2, _ = shuffle_with_seed(items, 42)

    assert shuffled1 == shuffled2


def test_shuffle_different_seeds():
    """Test that different seeds give different shuffles."""
    items = list(range(28))

    shuffled1, _ = shuffle_with_seed(items, 1)
    shuffled2, _ = shuffle_with_seed(items, 2)

    assert shuffled1 != shuffled2


def test_deal_shape():
    """Test that deal_with_seed returns correct shape."""
    hands = deal_with_seed(42)

    assert hands.shape == (4, 7)
    assert hands.dtype == np.int32


def test_deal_completeness():
    """Test that deal contains all 28 unique dominoes."""
    hands = deal_with_seed(42)

    all_dominoes = set(hands.flatten())
    assert len(all_dominoes) == 28
    assert all_dominoes == set(range(28))


def run_ts_export() -> dict:
    """Run the TypeScript export script and return the JSON output."""
    project_root = Path(__file__).parent.parent.parent

    # Run the export script
    result = subprocess.run(
        ["npx", "tsx", "scripts/export-deals.ts"],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"TypeScript export failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")
        raise RuntimeError("Failed to run export-deals.ts")

    return json.loads(result.stdout)


def test_against_typescript():
    """Compare Python output against TypeScript for test seeds."""
    ts_output = run_ts_export()

    for test_case in ts_output["deals"]:
        seed = test_case["seed"]
        ts_hands = test_case["hands"]  # List of 4 lists of 7 global IDs

        py_hands = deal_with_seed(seed)

        for player in range(4):
            py_hand = list(py_hands[player])
            ts_hand = ts_hands[player]

            assert py_hand == ts_hand, (
                f"Mismatch for seed={seed}, player={player}:\n"
                f"  Python: {py_hand}\n"
                f"  TypeScript: {ts_hand}"
            )

    print(f"Verified {len(ts_output['deals'])} seeds match TypeScript")


def test_shuffle_against_typescript():
    """Verify shuffle sequence matches TypeScript exactly."""
    ts_output = run_ts_export()

    for test_case in ts_output["shuffles"]:
        seed = test_case["seed"]
        ts_shuffled = test_case["shuffled"]

        # Start with DOMINO_VALUES order global IDs
        domino_ids = get_domino_values_order()
        py_shuffled, _ = shuffle_with_seed(domino_ids, seed)

        assert py_shuffled == ts_shuffled, (
            f"Shuffle mismatch for seed={seed}:\n"
            f"  Python: {py_shuffled}\n"
            f"  TypeScript: {ts_shuffled}"
        )

    print(f"Verified shuffle sequences match TypeScript")


def test_rng_sequence_against_typescript():
    """Verify RNG sequence matches TypeScript exactly."""
    ts_output = run_ts_export()

    for test_case in ts_output["rng_sequences"]:
        seed = test_case["seed"]
        ts_seeds = test_case["seeds"]
        ts_floats = test_case["floats"]

        rng = SeededRandom(seed)

        for i, (expected_seed, expected_float) in enumerate(zip(ts_seeds, ts_floats)):
            actual_float = rng.next()
            actual_seed = rng.get_seed()

            assert actual_seed == expected_seed, (
                f"Seed mismatch at step {i} for initial seed={seed}:\n"
                f"  Python: {actual_seed}\n"
                f"  TypeScript: {expected_seed}"
            )

            # Float comparison with tolerance for floating point
            assert abs(actual_float - expected_float) < 1e-15, (
                f"Float mismatch at step {i} for initial seed={seed}:\n"
                f"  Python: {actual_float}\n"
                f"  TypeScript: {expected_float}"
            )

    print(f"Verified RNG sequences match TypeScript")


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    test_lcg_next_basic()
    print("  lcg_next: PASS")

    test_domino_global_id_roundtrip()
    print("  domino_global_id_roundtrip: PASS")

    test_domino_values_order()
    print("  domino_values_order: PASS")

    test_domino_values_global_ids()
    print("  domino_values_global_ids: PASS")

    test_seeded_random_class()
    print("  seeded_random_class: PASS")

    test_shuffle_determinism()
    print("  shuffle_determinism: PASS")

    test_shuffle_different_seeds()
    print("  shuffle_different_seeds: PASS")

    test_deal_shape()
    print("  deal_shape: PASS")

    test_deal_completeness()
    print("  deal_completeness: PASS")

    print("\nRunning TypeScript comparison tests...")
    try:
        test_rng_sequence_against_typescript()
        print("  rng_sequence_against_typescript: PASS")

        test_shuffle_against_typescript()
        print("  shuffle_against_typescript: PASS")

        test_against_typescript()
        print("  deal_against_typescript: PASS")

        print("\nAll tests PASSED!")
    except Exception as e:
        print(f"\nTypeScript comparison failed: {e}")
        print("Make sure to run: npx tsx scripts/export-deals.ts")
        sys.exit(1)

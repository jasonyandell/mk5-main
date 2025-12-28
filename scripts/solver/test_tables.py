#!/usr/bin/env python3
"""
Test that Python tables match TypeScript tables exactly.

This script:
1. Exports TS tables to JSON via export-tables.ts
2. Compares every entry in Python tables to TS tables
3. Verifies: 252 EFFECTIVE_SUIT, 72 SUIT_MASK, 252 HAS_POWER, 252 RANK entries

Usage: python scripts/solver/test_tables.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tables import (
    DOMINO_PIPS,
    EFFECTIVE_SUIT,
    SUIT_MASK,
    HAS_POWER,
    RANK,
)


def export_ts_tables() -> dict:
    """Run the TypeScript export script and parse its JSON output."""
    scripts_dir = Path(__file__).parent.parent
    project_dir = scripts_dir.parent

    result = subprocess.run(
        ['npx', 'tsx', 'scripts/export-tables.ts'],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error running export-tables.ts:\n{result.stderr}")
        sys.exit(1)

    return json.loads(result.stdout)


def test_domino_pips(ts_tables: dict) -> int:
    """Test DOMINO_PIPS matches."""
    ts_pips = ts_tables['DOMINO_PIPS']
    errors = 0

    assert DOMINO_PIPS.shape == (28, 2), f"DOMINO_PIPS shape mismatch: {DOMINO_PIPS.shape}"

    for d in range(28):
        ts_lo, ts_hi = ts_pips[d]
        py_lo, py_hi = DOMINO_PIPS[d]

        if ts_lo != py_lo or ts_hi != py_hi:
            print(f"DOMINO_PIPS[{d}] mismatch: TS=[{ts_lo},{ts_hi}] PY=[{py_lo},{py_hi}]")
            errors += 1

    return errors


def test_effective_suit(ts_tables: dict) -> int:
    """Test EFFECTIVE_SUIT matches - 252 entries."""
    ts_suit = ts_tables['EFFECTIVE_SUIT']
    errors = 0
    count = 0

    assert EFFECTIVE_SUIT.shape == (28, 9), f"EFFECTIVE_SUIT shape mismatch: {EFFECTIVE_SUIT.shape}"

    for d in range(28):
        for abs_id in range(9):
            ts_val = ts_suit[d][abs_id]
            py_val = int(EFFECTIVE_SUIT[d, abs_id])
            count += 1

            if ts_val != py_val:
                print(f"EFFECTIVE_SUIT[{d}][{abs_id}] mismatch: TS={ts_val} PY={py_val}")
                errors += 1

    assert count == 252, f"EFFECTIVE_SUIT entry count: {count}"
    return errors


def test_suit_mask(ts_tables: dict) -> int:
    """Test SUIT_MASK matches - 72 entries."""
    ts_mask = ts_tables['SUIT_MASK']
    errors = 0
    count = 0

    assert SUIT_MASK.shape == (9, 8), f"SUIT_MASK shape mismatch: {SUIT_MASK.shape}"

    for abs_id in range(9):
        for suit in range(8):
            ts_val = ts_mask[abs_id][suit]
            py_val = int(SUIT_MASK[abs_id, suit])
            count += 1

            if ts_val != py_val:
                print(f"SUIT_MASK[{abs_id}][{suit}] mismatch: TS={ts_val} (0b{ts_val:028b}) PY={py_val} (0b{py_val:028b})")
                errors += 1

    assert count == 72, f"SUIT_MASK entry count: {count}"
    return errors


def test_has_power(ts_tables: dict) -> int:
    """Test HAS_POWER matches - 252 entries."""
    ts_power = ts_tables['HAS_POWER']
    errors = 0
    count = 0

    assert HAS_POWER.shape == (28, 9), f"HAS_POWER shape mismatch: {HAS_POWER.shape}"

    for d in range(28):
        for power_id in range(9):
            ts_val = ts_power[d][power_id]
            py_val = bool(HAS_POWER[d, power_id])
            count += 1

            if ts_val != py_val:
                print(f"HAS_POWER[{d}][{power_id}] mismatch: TS={ts_val} PY={py_val}")
                errors += 1

    assert count == 252, f"HAS_POWER entry count: {count}"
    return errors


def test_rank(ts_tables: dict) -> int:
    """Test RANK matches - 252 entries."""
    ts_rank = ts_tables['RANK']
    errors = 0
    count = 0

    assert RANK.shape == (28, 9), f"RANK shape mismatch: {RANK.shape}"

    for d in range(28):
        for power_id in range(9):
            ts_val = ts_rank[d][power_id]
            py_val = int(RANK[d, power_id])
            count += 1

            if ts_val != py_val:
                print(f"RANK[{d}][{power_id}] mismatch: TS={ts_val} PY={py_val}")
                errors += 1

    assert count == 252, f"RANK entry count: {count}"
    return errors


def main():
    print("Exporting TypeScript tables...")
    ts_tables = export_ts_tables()
    print("TypeScript tables exported successfully.\n")

    total_errors = 0

    print("Testing DOMINO_PIPS (28 x 2 = 56 entries)...")
    errors = test_domino_pips(ts_tables)
    print(f"  {'PASS' if errors == 0 else 'FAIL'}: {errors} errors\n")
    total_errors += errors

    print("Testing EFFECTIVE_SUIT (28 x 9 = 252 entries)...")
    errors = test_effective_suit(ts_tables)
    print(f"  {'PASS' if errors == 0 else 'FAIL'}: {errors} errors\n")
    total_errors += errors

    print("Testing SUIT_MASK (9 x 8 = 72 entries)...")
    errors = test_suit_mask(ts_tables)
    print(f"  {'PASS' if errors == 0 else 'FAIL'}: {errors} errors\n")
    total_errors += errors

    print("Testing HAS_POWER (28 x 9 = 252 entries)...")
    errors = test_has_power(ts_tables)
    print(f"  {'PASS' if errors == 0 else 'FAIL'}: {errors} errors\n")
    total_errors += errors

    print("Testing RANK (28 x 9 = 252 entries)...")
    errors = test_rank(ts_tables)
    print(f"  {'PASS' if errors == 0 else 'FAIL'}: {errors} errors\n")
    total_errors += errors

    print("=" * 50)
    if total_errors == 0:
        print("ALL TESTS PASSED!")
        print("  - DOMINO_PIPS: 56 entries verified")
        print("  - EFFECTIVE_SUIT: 252 entries verified")
        print("  - SUIT_MASK: 72 entries verified")
        print("  - HAS_POWER: 252 entries verified")
        print("  - RANK: 252 entries verified")
        sys.exit(0)
    else:
        print(f"TESTS FAILED: {total_errors} total errors")
        sys.exit(1)


if __name__ == '__main__':
    main()

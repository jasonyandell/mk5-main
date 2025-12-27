"""
Texas 42 Solver - Output Format

Save solved data to JSON for loading in JavaScript.
"""

import json
import gzip
from typing import Dict, List
from pathlib import Path

from .solve import SolveResult
from .context import SeedContext


def save_json(result: SolveResult, output_path: str, compress: bool = True):
    """
    Save solve result to JSON file.

    Format:
    {
        "seed": 12345,
        "decl_id": 3,
        "root_value": 14,
        "stats": {...},
        "states": {
            "packed_hex": {"v": 14, "m": [-128, 8, 14, -128, 2, -128, -128]},
            ...
        }
    }
    """
    data = {
        "seed": result.seed,
        "decl_id": result.decl_id,
        "root_value": result.root_value,
        "stats": result.stats,
        "states": {}
    }

    # Convert packed states to hex strings for JSON
    for packed, value in result.V.items():
        hex_key = hex(packed)
        moves = result.MoveValues.get(packed, [-128] * 7)
        data["states"][hex_key] = {"v": value, "m": moves}

    # Write to file
    path = Path(output_path)
    if compress or output_path.endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    return path


def load_json(input_path: str) -> dict:
    """Load solve result from JSON file."""
    path = Path(input_path)
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def estimate_size(result: SolveResult) -> dict:
    """Estimate storage sizes."""
    num_states = len(result.V)
    # Each state: ~20 bytes (hex key) + ~30 bytes (value + moves)
    estimated_json_bytes = num_states * 50
    # Compressed typically 10-20x smaller
    estimated_gz_bytes = estimated_json_bytes // 15

    return {
        "num_states": num_states,
        "estimated_json_mb": estimated_json_bytes / 1e6,
        "estimated_gz_mb": estimated_gz_bytes / 1e6,
    }


if __name__ == "__main__":
    # Test with small data
    from .solve import SolveResult

    mock_result = SolveResult(
        seed=12345,
        decl_id=3,
        root_value=14,
        V={0x1234: 14, 0x5678: -2},
        MoveValues={0x1234: [-128, 8, 14, -128, 2, -128, -128], 0x5678: [0, 1, 2, 3, -128, -128, -128]},
        stats={"total_time": 10.5}
    )

    print("Saving mock result...")
    path = save_json(mock_result, "/tmp/test_solve.json.gz")
    print(f"Saved to {path}")

    print("\nLoading back...")
    loaded = load_json(str(path))
    print(f"Loaded {len(loaded['states'])} states")
    print(f"Root value: {loaded['root_value']}")

"""
Output module for GPU solver - writes results to Parquet files.

One file per (seed, decl_id) combination with atomic writes for crash recovery.
"""

import time
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np

# Try to import pyarrow for Parquet writing
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


@dataclass
class SeedTimer:
    """Track timing for each phase of solving a seed."""
    seed: int
    decl_id: int
    start_time: float = 0.0
    phase_start: float = 0.0

    def __post_init__(self):
        self.start_time = time.time()
        self.phase_start = self.start_time

    def phase(self, name: str, extra: str = "") -> None:
        """Log completion of a phase."""
        now = time.time()
        elapsed = now - self.phase_start
        ts = time.strftime("%H:%M:%S")
        extra_str = f" | {extra}" if extra else ""
        print(f"{ts} | seed={self.seed} decl={self.decl_id} | {name} | {elapsed:.2f}s{extra_str}")
        self.phase_start = now

    def done(self, root_value: int) -> None:
        """Log completion of entire solve."""
        total = time.time() - self.start_time
        ts = time.strftime("%H:%M:%S")
        print(f"{ts} | seed={self.seed} decl={self.decl_id} | DONE | {total:.2f}s | root={root_value:+d}")


def get_output_path(output_dir: Path, seed: int, decl_id: int) -> Path:
    """Get the output path for a solved seed."""
    return output_dir / f"seed_{seed:08d}_decl_{decl_id}.parquet"


def write_parquet(
    output_path: Path,
    seed: int,
    decl_id: int,
    all_states: torch.Tensor,
    V: torch.Tensor,
    move_values: torch.Tensor,
) -> None:
    """
    Write solver results to Parquet file with atomic write.

    Args:
        output_path: Destination file path
        seed: RNG seed used for dealing
        decl_id: Declaration ID (0-6 for pip trump)
        all_states: (N,) int64 tensor of packed states
        V: (N,) int8 tensor of minimax values
        move_values: (N, 7) int8 tensor of move values
    """
    if not HAS_PYARROW:
        raise ImportError("pyarrow is required for Parquet output. Install with: pip install pyarrow")

    # Convert to numpy for pyarrow
    states_np = all_states.cpu().numpy()
    v_np = V.cpu().numpy()
    mv_np = move_values.cpu().numpy()

    # Create table with metadata
    table = pa.table({
        'state': states_np,
        'value': v_np,
        'move0': mv_np[:, 0],
        'move1': mv_np[:, 1],
        'move2': mv_np[:, 2],
        'move3': mv_np[:, 3],
        'move4': mv_np[:, 4],
        'move5': mv_np[:, 5],
        'move6': mv_np[:, 6],
    })

    # Add metadata
    metadata = {
        b'seed': str(seed).encode(),
        b'decl_id': str(decl_id).encode(),
        b'state_count': str(len(states_np)).encode(),
        b'root_value': str(int(V[0])).encode(),
    }
    table = table.replace_schema_metadata(metadata)

    # Atomic write: write to temp file, then rename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix('.tmp')

    try:
        pq.write_table(table, temp_path, compression='snappy')
        temp_path.rename(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_json(
    output_path: Path,
    seed: int,
    decl_id: int,
    all_states: torch.Tensor,
    V: torch.Tensor,
    move_values: torch.Tensor,
) -> None:
    """
    Write solver results to JSON file (fallback if pyarrow not available).

    This is less efficient but works without additional dependencies.
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'seed': seed,
        'decl_id': decl_id,
        'state_count': len(all_states),
        'root_value': int(V[0]),
        'states': all_states.cpu().tolist(),
        'values': V.cpu().tolist(),
        'move_values': move_values.cpu().tolist(),
    }

    temp_path = output_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f)
        temp_path.rename(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def solve_and_save(
    seed: int,
    decl_id: int,
    output_dir: Path,
    device: torch.device = None,
    use_json: bool = False,
) -> bool:
    """
    Solve one seed and write immediately.

    Args:
        seed: RNG seed for dealing
        decl_id: Declaration ID (0-6 for pip trump)
        output_dir: Directory to write output files
        device: Target device (defaults to CPU)
        use_json: Use JSON output instead of Parquet

    Returns:
        True if newly solved, False if already exists
    """
    from solve import solve_seed

    ext = '.json' if use_json else '.parquet'
    output_path = output_dir / f"seed_{seed:08d}_decl_{decl_id}{ext}"

    if output_path.exists():
        return False

    timer = SeedTimer(seed, decl_id)

    # Solve
    all_states, V, move_values, root_value = solve_seed(seed, decl_id, device)
    timer.phase("solve", f"states={len(all_states):,}")

    # Write
    if use_json:
        write_json(output_path, seed, decl_id, all_states, V, move_values)
    else:
        write_parquet(output_path, seed, decl_id, all_states, V, move_values)

    timer.done(root_value)
    return True

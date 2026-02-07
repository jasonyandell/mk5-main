from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch


try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


OutputFormat = Literal["parquet", "pt"]


def output_path_for(
    output_dir: Path,
    seed: int,
    decl_id: int,
    fmt: OutputFormat,
    *,
    opp_seed: int | None = None,
) -> Path:
    """Generate output path for a shard.

    Args:
        output_dir: Directory for output files
        seed: Base seed (determines P0 hand in marginalized mode)
        decl_id: Declaration ID (0-9)
        fmt: Output format ('parquet' or 'pt')
        opp_seed: If provided, creates marginalized naming: seed_X_oppY_decl_Z

    Returns:
        Path for the output file
    """
    ext = "parquet" if fmt == "parquet" else "pt"
    if opp_seed is not None:
        return output_dir / f"seed_{seed:08d}_opp{opp_seed}_decl_{decl_id}.{ext}"
    return output_dir / f"seed_{seed:08d}_decl_{decl_id}.{ext}"


def write_result(
    output_path: Path,
    seed: int,
    decl_id: int,
    all_states: torch.Tensor,
    v: torch.Tensor,
    move_values: torch.Tensor,
    *,
    fmt: OutputFormat | None = None,
    non_blocking: bool = False,
) -> None:
    """Write solver results to disk.

    Args:
        non_blocking: If True, uses non_blocking GPU→CPU transfers. The caller
            must ensure the CUDA stream is synchronized before the next seed
            reuses GPU memory.
    """
    fmt = fmt or ("parquet" if output_path.suffix == ".parquet" else "pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if fmt == "pt":
        torch.save(
            {
                "seed": int(seed),
                "decl_id": int(decl_id),
                "all_states": all_states.detach().to("cpu", non_blocking=non_blocking),
                "V": v.detach().to("cpu", non_blocking=non_blocking),
                "move_values": move_values.detach().to("cpu", non_blocking=non_blocking),
            },
            tmp_path,
        )
        tmp_path.replace(output_path)
        return

    if fmt != "parquet":
        raise ValueError(f"unknown fmt: {fmt}")

    if not (HAS_PYARROW and HAS_NUMPY):
        raise RuntimeError("Parquet output requires `pyarrow` and `numpy` (or use --format pt).")

    # Non-blocking transfers: GPU can continue while data moves to CPU.
    # The .numpy() call will implicitly synchronize when accessing the data.
    states_cpu = all_states.detach().to("cpu", non_blocking=non_blocking)
    v_cpu = v.detach().to("cpu", non_blocking=non_blocking)
    mv_cpu = move_values.detach().to("cpu", non_blocking=non_blocking)

    table = pa.table(
        {
            "state": pa.array(states_cpu.numpy().astype("int64", copy=False)),
            "V": pa.array(v_cpu.numpy().astype("int8", copy=False)),
            "q0": pa.array(mv_cpu[:, 0].numpy().astype("int8", copy=False)),
            "q1": pa.array(mv_cpu[:, 1].numpy().astype("int8", copy=False)),
            "q2": pa.array(mv_cpu[:, 2].numpy().astype("int8", copy=False)),
            "q3": pa.array(mv_cpu[:, 3].numpy().astype("int8", copy=False)),
            "q4": pa.array(mv_cpu[:, 4].numpy().astype("int8", copy=False)),
            "q5": pa.array(mv_cpu[:, 5].numpy().astype("int8", copy=False)),
            "q6": pa.array(mv_cpu[:, 6].numpy().astype("int8", copy=False)),
        }
    )
    table = table.replace_schema_metadata(
        {
            b"seed": str(int(seed)).encode(),
            b"decl_id": str(int(decl_id)).encode(),
        }
    )

    pq.write_table(table, tmp_path, compression="snappy")
    tmp_path.replace(output_path)

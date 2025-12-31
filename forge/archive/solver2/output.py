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


def output_path_for(output_dir: Path, seed: int, decl_id: int, fmt: OutputFormat) -> Path:
    return output_dir / f"seed_{seed:08d}_decl_{decl_id}.{'parquet' if fmt == 'parquet' else 'pt'}"


def write_result(
    output_path: Path,
    seed: int,
    decl_id: int,
    all_states: torch.Tensor,
    v: torch.Tensor,
    move_values: torch.Tensor,
    *,
    fmt: OutputFormat | None = None,
) -> None:
    fmt = fmt or ("parquet" if output_path.suffix == ".parquet" else "pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if fmt == "pt":
        torch.save(
            {
                "seed": int(seed),
                "decl_id": int(decl_id),
                "all_states": all_states.detach().cpu(),
                "V": v.detach().cpu(),
                "move_values": move_values.detach().cpu(),
            },
            tmp_path,
        )
        tmp_path.replace(output_path)
        return

    if fmt != "parquet":
        raise ValueError(f"unknown fmt: {fmt}")

    if not (HAS_PYARROW and HAS_NUMPY):
        raise RuntimeError("Parquet output requires `pyarrow` and `numpy` (or use --format pt).")

    states_cpu = all_states.detach().cpu()
    v_cpu = v.detach().cpu()
    mv_cpu = move_values.detach().cpu()

    table = pa.table(
        {
            "state": pa.array(states_cpu.numpy().astype("int64", copy=False)),
            "V": pa.array(v_cpu.numpy().astype("int8", copy=False)),
            "mv0": pa.array(mv_cpu[:, 0].numpy().astype("int8", copy=False)),
            "mv1": pa.array(mv_cpu[:, 1].numpy().astype("int8", copy=False)),
            "mv2": pa.array(mv_cpu[:, 2].numpy().astype("int8", copy=False)),
            "mv3": pa.array(mv_cpu[:, 3].numpy().astype("int8", copy=False)),
            "mv4": pa.array(mv_cpu[:, 4].numpy().astype("int8", copy=False)),
            "mv5": pa.array(mv_cpu[:, 5].numpy().astype("int8", copy=False)),
            "mv6": pa.array(mv_cpu[:, 6].numpy().astype("int8", copy=False)),
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

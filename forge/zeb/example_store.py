"""Training example serialization for worker-to-learner data transfer.

Workers save ExampleBatch .pt files; the learner scans and loads them.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch
from torch import Tensor


@dataclass
class ExampleBatch:
    """CPU tensors, ready to .to(device) on load."""

    observations: Tensor    # [N, 36, 8] int32
    masks: Tensor           # [N, 36] bool
    hand_indices: Tensor    # [N, 7] int64
    hand_masks: Tensor      # [N, 7] bool
    policy_targets: Tensor  # [N, 7] float32
    value_targets: Tensor   # [N] float32
    metadata: dict          # {worker_id, model_step, n_games, timestamp}

    @property
    def n_examples(self) -> int:
        return self.observations.shape[0]


def save_examples(batch: ExampleBatch, dir: Path, worker_id: str) -> Path:
    """Atomic write: save to .tmp, rename to .pt."""
    dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    short_id = uuid4().hex[:8]
    name = f"{worker_id}_{ts}_{short_id}"
    tmp_path = dir / f"{name}.tmp"
    final_path = dir / f"{name}.pt"
    torch.save({
        'observations': batch.observations,
        'masks': batch.masks,
        'hand_indices': batch.hand_indices,
        'hand_masks': batch.hand_masks,
        'policy_targets': batch.policy_targets,
        'value_targets': batch.value_targets,
        'metadata': batch.metadata,
    }, tmp_path)
    os.rename(tmp_path, final_path)
    return final_path


def load_examples(path: Path) -> ExampleBatch:
    """Load and return CPU tensors."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return ExampleBatch(
        observations=data['observations'],
        masks=data['masks'],
        hand_indices=data['hand_indices'],
        hand_masks=data['hand_masks'],
        policy_targets=data['policy_targets'],
        value_targets=data['value_targets'],
        metadata=data['metadata'],
    )


def scan_pending(dir: Path) -> list[Path]:
    """Return .pt files sorted by mtime (oldest first)."""
    if not dir.exists():
        return []
    files = list(dir.glob('*.pt'))
    files.sort(key=lambda p: p.stat().st_mtime)
    return files

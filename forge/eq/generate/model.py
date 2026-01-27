"""Model inference functions for GPU E[Q] pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor


def query_model(
    model,
    tokens: Tensor,
    masks: Tensor,
    states: GameStateTensor,
    n_samples: int,
    device: str,
    chunk_size: int | None = None,
) -> Tensor:
    """Run model forward pass with automatic chunking.

    Args:
        model: Stage 1 model
        tokens: [batch, 32, 12] tokenized input
        masks: [batch, 32] attention masks
        states: GameStateTensor (for current_player)
        n_samples: Number of samples per game
        device: Device
        chunk_size: Max batch per forward pass (auto-calibrated if None)

    Returns:
        [batch, 7] Q-values
    """
    n_games = states.n_games
    batch_size = n_games * n_samples

    # Expand current_player to all samples
    current_players = states.current_player.unsqueeze(1).expand(-1, n_samples).reshape(batch_size).long()

    # Determine model device
    model_device = next(model.parameters()).device

    # Ensure tokens are on model device and correct dtype
    if tokens.device != model_device:
        tokens = tokens.to(model_device)
    if masks.device != model_device:
        masks = masks.to(model_device)
    if current_players.device != model_device:
        current_players = current_players.to(model_device)

    # Convert int8 to int32 for model input (Embedding layer)
    tokens = tokens.to(torch.int32)

    # Get optimal chunk size if not provided
    if chunk_size is None and model_device.type == 'cuda':
        from forge.eq.calibration import get_optimal_chunk
        chunk_size = get_optimal_chunk(model, device)

    # Model forward with mixed precision (float16 on CUDA for ~2-3x speedup)
    with torch.inference_mode():
        use_amp = (model_device.type == 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            if chunk_size is None or batch_size <= chunk_size:
                # Single forward pass
                q_values, _ = model(tokens, masks, current_players)
            else:
                # Chunked forward passes
                q_chunks = []
                for i in range(0, batch_size, chunk_size):
                    end = min(i + chunk_size, batch_size)
                    q_chunk, _ = model(
                        tokens[i:end],
                        masks[i:end],
                        current_players[i:end],
                    )
                    q_chunks.append(q_chunk)
                q_values = torch.cat(q_chunks, dim=0)

    # Clone to prevent CUDA graph buffer reuse when model is called multiple times
    # (e.g., posterior weighting calls model again for past steps)
    return q_values.clone()

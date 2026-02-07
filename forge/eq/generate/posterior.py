"""Posterior weighting functions for GPU E[Q] pipeline."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.posterior_gpu import (
    compute_legal_masks_gpu,
    compute_posterior_weights_gpu,
    reconstruct_historical_hands,
    reconstruct_past_states_gpu,
)
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.eq.types import PosteriorDiagnostics

from .deals import build_hypothetical_deals
from .eq_compute import compute_eq_pdf
from .types import PosteriorConfig


def compute_posterior_weighted_eq(
    states: GameStateTensor,
    worlds: Tensor,
    q_values: Tensor,
    n_samples: int,
    model,
    tokenizer: GPUTokenizer,
    posterior_config: PosteriorConfig,
    device: str,
) -> tuple[Tensor, Tensor, Tensor, PosteriorDiagnostics | None]:
    """Compute posterior-weighted E[Q] using past K steps.

    Args:
        states: Current game states [N games]
        worlds: Sampled opponent hands [N, M, 3, max_hand_size]
        q_values: Q-values for current step [N*M, 7]
        n_samples: Number of samples per game (M)
        model: Stage 1 oracle model
        tokenizer: GPUTokenizer instance
        posterior_config: Posterior weighting configuration
        device: Device for computation

    Returns:
        Tuple of (e_q_mean, e_q_var, e_q_pdf, diagnostics):
            - e_q_mean: [N, 7] posterior-weighted E[Q] mean values
            - e_q_var: [N, 7] posterior-weighted E[Q] variance values
            - e_q_pdf: [N, 7, 85] posterior-weighted PDF
            - diagnostics: PosteriorDiagnostics (aggregated across games, or None if fallback)
    """
    n_games = states.n_games

    # 1. Compute history lengths (count non-empty entries in history)
    # history: [N, 28, 3] with -1 for unused entries
    # An entry is valid if player (first field) >= 0
    history_len = (states.history[:, :, 0] >= 0).sum(dim=1)  # [N]

    # Note: Games with insufficient history (< window_k) are handled per-game
    # by the valid_mask in reconstruct_past_states_gpu. Invalid steps get
    # zero legal actions -> uniform weights for that game only.

    # 2. Reconstruct past states
    past_states = reconstruct_past_states_gpu(
        history=states.history,
        history_len=history_len,
        window_k=posterior_config.window_k,
        device=states.device,
    )

    # 3. Build hypothetical deals from worlds (same format as main pipeline)
    hypothetical = build_hypothetical_deals(states, worlds)  # [N, M, 4, 7]

    # 4. Reconstruct historical hands at each past step k
    # This fixes the bug where we were using CURRENT hands instead of HISTORICAL hands
    # Historical hands have dominoes that were played AFTER step k added back
    historical_hands = reconstruct_historical_hands(
        hypothetical=hypothetical,
        history=states.history,
        history_len=history_len,
        step_indices=past_states.step_indices,
        valid_mask=past_states.valid_mask,
    )  # [N, M, K, 4, 7]

    # 5. Tokenize past steps for all games/samples/steps
    # Need larger tokenizer for N*M*K batch (past steps tokenization)
    K = posterior_config.window_k
    past_batch_size = n_games * n_samples * K

    # Create a separate tokenizer for past steps if needed
    if past_batch_size > tokenizer.max_batch:
        past_tokenizer = GPUTokenizer(max_batch=past_batch_size, device=states.device)
    else:
        past_tokenizer = tokenizer

    past_tokens, past_masks = past_tokenizer.tokenize_past_steps_batched(
        worlds=hypothetical,
        past_states=past_states,
        decl_ids=states.decl_ids,  # Per-game declaration IDs
        historical_hands=historical_hands,  # Reconstructed hands at each past step
    )

    # 5. Query oracle for past steps
    # past_tokens: [N*K*M, 32, 12] with layout g*K*M + k*M + m
    total_batch = n_games * n_samples * K

    # Expand actors to match tokenizer's [N, K, M] layout: actors[n, k] -> [n, k, m] -> [n*k*m]
    # Clamp actors to [0, 3] range - invalid entries (-1) will be clamped to 0
    # Their outputs will be masked out by valid_mask anyway
    actors_clamped = past_states.actors.clamp(0, 3)
    actors_expanded = actors_clamped.unsqueeze(-1).expand(n_games, K, n_samples)  # [N, K, M]
    actors_flat = actors_expanded.reshape(total_batch).long()  # [N*K*M]

    # Determine model device
    model_device = next(model.parameters()).device

    # Move tensors to model device
    if past_tokens.device != model_device:
        past_tokens = past_tokens.to(model_device)
    if past_masks.device != model_device:
        past_masks = past_masks.to(model_device)
    if actors_flat.device != model_device:
        actors_flat = actors_flat.to(model_device)

    # Convert to int32 for model
    past_tokens = past_tokens.to(torch.int32)

    # Check if ANY valid steps exist - if all masks are 0, skip model and use uniform weights
    # This happens early in the game when history_len < window_k for all games
    has_any_valid = past_masks.any()
    if not has_any_valid:
        # No valid history - fall back to uniform weighting
        q_reshaped = q_values.view(n_games, n_samples, 7)
        e_q = q_reshaped.mean(dim=1)
        e_q_var = q_reshaped.var(dim=1, unbiased=False)
        e_q_pdf = compute_eq_pdf(q_reshaped)
        # Return proper diagnostics: ESS = n_samples (uniform weights)
        uniform_diagnostics = PosteriorDiagnostics(
            ess=float(n_samples),  # Uniform weights -> ESS = n_samples
            max_w=1.0 / n_samples,  # Uniform weight
            entropy=math.log(n_samples),  # Uniform entropy
            k_eff=float(n_samples),  # k_eff = exp(entropy) = n_samples
            n_invalid=0,
            n_illegal=0,
            window_nll=0.0,
            window_k_used=0,  # No history used
        )
        return e_q, e_q_var, e_q_pdf, uniform_diagnostics

    # Model forward pass
    with torch.inference_mode():
        use_amp = (model_device.type == 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            q_past, _ = model(past_tokens, past_masks, actors_flat)

    # Reshape: [N*K*M, 7] -> [N, K, M, 7] to match tokenizer layout
    q_past = q_past.view(n_games, K, n_samples, 7)
    # Transpose to [N, M, K, 7] for compatibility with posterior weighting
    q_past = q_past.permute(0, 2, 1, 3).contiguous()

    # 6. Compute legal masks for past steps
    legal_masks = compute_legal_masks_gpu(
        worlds=hypothetical,
        past_states=past_states,
        decl_ids=states.decl_ids,  # Per-game declaration IDs
        device=states.device,
        historical_hands=historical_hands,  # Reconstructed hands at each past step
    )

    # 7. Compute posterior weights
    weights, diagnostics = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=past_states.observed_actions,
        worlds=hypothetical,
        actors=past_states.actors,
        tau=posterior_config.tau,
        uniform_mix=posterior_config.uniform_mix,
        historical_hands=historical_hands,  # Reconstructed hands at each past step
    )

    # 8. Apply weights to current Q-values
    # q_values: [N*M, 7] -> [N, M, 7]
    q_values_reshaped = q_values.view(n_games, n_samples, 7)

    # weights: [N, M] -> [N, M, 1] for broadcasting
    weights_expanded = weights.unsqueeze(-1)  # [N, M, 1]

    # Weighted mean: e_q[n] = sum_m(weights[n, m] * q_values[n, m, :])
    e_q_mean = (weights_expanded * q_values_reshaped).sum(dim=1)  # [N, 7]

    # Weighted variance: Var = E[X^2] - E[X]^2
    e_q_sq = (weights_expanded * q_values_reshaped**2).sum(dim=1)  # [N, 7]
    e_q_var = e_q_sq - e_q_mean**2  # [N, 7]

    # Weighted PDF
    e_q_pdf = compute_eq_pdf(q_values_reshaped, weights=weights)  # [N, 7, 85]

    # 9. Aggregate diagnostics across games (average)
    if diagnostics is not None:
        # diagnostics is a dict with tensors of shape [N] for each field
        # We aggregate by taking the mean across games
        aggregated_diagnostics = PosteriorDiagnostics(
            ess=diagnostics['ess'].mean().item() if 'ess' in diagnostics else 0.0,
            max_w=diagnostics['max_w'].mean().item() if 'max_w' in diagnostics else 0.0,
            entropy=diagnostics['entropy'].mean().item() if 'entropy' in diagnostics else 0.0,
            k_eff=diagnostics['k_eff'].mean().item() if 'k_eff' in diagnostics else 0.0,
            n_invalid=0,  # Not provided by compute_posterior_weights_gpu
            n_illegal=0,  # Not provided by compute_posterior_weights_gpu
            window_nll=0.0,  # Not provided by compute_posterior_weights_gpu
            window_k_used=posterior_config.window_k,  # Use config value
        )
    else:
        aggregated_diagnostics = None

    return e_q_mean, e_q_var, e_q_pdf, aggregated_diagnostics


def compute_posterior_weights_batch(
    states: GameStateTensor,
    worlds: Tensor,
    hypothetical: Tensor,
    model,
    tokenizer: GPUTokenizer,
    posterior_config: PosteriorConfig,
    device: str,
) -> tuple[Tensor, PosteriorDiagnostics | None]:
    """Compute posterior weights for a batch of sampled worlds.

    Extracts the weight computation from compute_posterior_weighted_eq
    for use in adaptive sampling.

    Args:
        states: Current game states [N games]
        worlds: Sampled opponent hands [N, M, 3, 7]
        hypothetical: Full deals [N, M, 4, 7]
        model: Stage 1 oracle
        tokenizer: GPU tokenizer
        posterior_config: Posterior config
        device: Device

    Returns:
        Tuple of (weights, diagnostics):
            - weights: [N, M] normalized weights summing to 1 per game
            - diagnostics: PosteriorDiagnostics or None
    """
    n_games = states.n_games
    n_samples = worlds.shape[1]
    K = posterior_config.window_k

    # 1. Compute history lengths
    history_len = (states.history[:, :, 0] >= 0).sum(dim=1)

    # 2. Check if any valid history exists
    has_history = (history_len >= 1).any()
    if not has_history:
        # No history - return uniform weights
        weights = torch.ones(n_games, n_samples, device=device) / n_samples
        return weights, None

    # 3. Reconstruct past states
    past_states = reconstruct_past_states_gpu(
        history=states.history,
        history_len=history_len,
        window_k=K,
        device=device,
    )

    # 4. Reconstruct historical hands
    historical_hands = reconstruct_historical_hands(
        hypothetical=hypothetical,
        history=states.history,
        history_len=history_len,
        step_indices=past_states.step_indices,
        valid_mask=past_states.valid_mask,
    )

    # 5. Check if ANY valid steps exist
    if not past_states.valid_mask.any():
        weights = torch.ones(n_games, n_samples, device=device) / n_samples
        return weights, None

    # 6. Tokenize past steps
    past_batch_size = n_games * n_samples * K
    if past_batch_size > tokenizer.max_batch:
        past_tokenizer = GPUTokenizer(max_batch=past_batch_size, device=device)
    else:
        past_tokenizer = tokenizer

    past_tokens, past_masks = past_tokenizer.tokenize_past_steps_batched(
        worlds=hypothetical,
        past_states=past_states,
        decl_ids=states.decl_ids,
        historical_hands=historical_hands,
    )

    # 7. Query oracle for past steps
    actors_clamped = past_states.actors.clamp(0, 3)
    actors_expanded = actors_clamped.unsqueeze(-1).expand(n_games, K, n_samples)
    actors_flat = actors_expanded.reshape(-1).long()

    model_device = next(model.parameters()).device
    if past_tokens.device != model_device:
        past_tokens = past_tokens.to(model_device)
    if past_masks.device != model_device:
        past_masks = past_masks.to(model_device)
    if actors_flat.device != model_device:
        actors_flat = actors_flat.to(model_device)

    past_tokens = past_tokens.to(torch.int32)

    # Process in chunks to limit peak VRAM usage
    # Without chunking: N*K*M forward passes allocated at once (e.g., 5*4*5000 = 100k)
    # With chunking: only chunk_size activations at once
    past_batch_size = past_tokens.shape[0]
    chunk_size = 8000

    with torch.inference_mode():
        use_amp = (model_device.type == 'cuda')
        dtype = torch.float16 if use_amp else torch.float32
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Pre-allocate output on GPU
            q_past = torch.empty(past_batch_size, 7, device=model_device, dtype=dtype)

            # Fill in chunks - all GPU operations, no CPU sync
            for i in range(0, past_batch_size, chunk_size):
                end = min(i + chunk_size, past_batch_size)
                q_past[i:end], _ = model(
                    past_tokens[i:end], past_masks[i:end], actors_flat[i:end]
                )

    # Reshape: [N*K*M, 7] -> [N, K, M, 7] -> [N, M, K, 7]
    q_past = q_past.view(n_games, K, n_samples, 7)
    q_past = q_past.permute(0, 2, 1, 3).contiguous()

    # 8. Compute legal masks for past steps
    legal_masks = compute_legal_masks_gpu(
        worlds=hypothetical,
        past_states=past_states,
        decl_ids=states.decl_ids,
        device=device,
        historical_hands=historical_hands,
    )

    # 9. Compute posterior weights
    weights, diagnostics = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=past_states.observed_actions,
        worlds=hypothetical,
        actors=past_states.actors,
        tau=posterior_config.tau,
        uniform_mix=posterior_config.uniform_mix,
        historical_hands=historical_hands,
    )

    return weights, diagnostics

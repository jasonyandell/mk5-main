"""Adaptive convergence-based sampling for GPU E[Q] pipeline."""

from __future__ import annotations

from typing import TextIO

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.eq.types import PosteriorDiagnostics

from .deals import build_hypothetical_deals
from .model import query_model
from .posterior import compute_posterior_weights_batch
from .sampling import sample_worlds_batched
from .tokenization import tokenize_batched
from .types import AdaptiveConfig, PosteriorConfig


# Module-level log file for adaptive sampling convergence tracking
_ADAPTIVE_LOG: TextIO | None = None


def set_adaptive_log(log_file: TextIO | None) -> None:
    """Set the log file for adaptive sampling convergence tracking."""
    global _ADAPTIVE_LOG
    _ADAPTIVE_LOG = log_file


def sample_until_convergence(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    tokenizer: GPUTokenizer,
    model,
    adaptive_config: AdaptiveConfig,
    device: str,
    decision_idx: int = 0,
    seeds: list[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, None, int, bool]:
    """Sample worlds until E[Q] estimates converge.

    Uses iterative batch sampling with running statistics to detect convergence.
    Convergence criterion: max(SEM) over legal actions < sem_threshold
    where SEM = Standard Error of Mean = sigma / sqrt(n)

    Args:
        states: GameStateTensor with N games
        sampler: Pre-allocated world sampler
        tokenizer: Pre-allocated GPU tokenizer
        model: Stage 1 oracle model
        adaptive_config: Adaptive sampling configuration
        device: Device for computation

    Returns:
        Tuple of (e_q, e_q_var, e_q_pdf, diagnostics, n_samples_used, did_converge):
            - e_q: [N, 7] mean Q-values
            - e_q_var: [N, 7] variance of Q-values
            - e_q_pdf: [N, 7, 85] full PDF
            - diagnostics: None (posterior not supported with adaptive)
            - n_samples_used: Number of samples actually used
            - did_converge: True if SEM < threshold, False if hit max_samples
    """
    n_games = states.n_games
    n_actions = 7
    n_bins = 85
    batch_size = adaptive_config.batch_size
    min_samples = adaptive_config.min_samples
    max_samples = adaptive_config.max_samples
    sem_threshold = adaptive_config.sem_threshold

    # Get legal actions mask for convergence checking
    legal_mask = states.legal_actions()  # [N, 7]

    # Running statistics for mean/variance (sum-based for numerical stability)
    n_total = torch.zeros(n_games, device=device, dtype=torch.int64)
    q_sum = torch.zeros(n_games, n_actions, device=device, dtype=torch.float32)
    q_sq_sum = torch.zeros(n_games, n_actions, device=device, dtype=torch.float32)

    # Online PDF accumulation: histogram counts (unnormalized, normalize at end)
    # Layout: [N * 85 * 7] flat for efficient scatter_add
    pdf_counts = torch.zeros(n_games * n_bins * n_actions, device=device, dtype=torch.float32)

    # Pre-compute game offset for PDF accumulation (constant across iterations)
    game_idx = torch.arange(n_games, device=device).view(n_games, 1, 1)  # [N, 1, 1]
    game_offset = game_idx * (n_bins * n_actions)  # [N, 1, 1]
    action_idx = torch.arange(n_actions, device=device).view(1, 1, n_actions)  # [1, 1, 7]

    # Per-game convergence tracking
    converged = torch.zeros(n_games, dtype=torch.bool, device=device)

    iteration = 0
    while True:
        # Sample a batch of worlds
        worlds = sample_worlds_batched(states, sampler, batch_size)  # [N, batch, 3, 7]

        # Build hypothetical deals
        hypothetical = build_hypothetical_deals(states, worlds)  # [N, batch, 4, 7]

        # Tokenize (recreate tokenizer if needed)
        n_worlds = worlds.shape[1]
        total_batch = n_games * n_worlds
        if total_batch > tokenizer.max_batch:
            tokenizer = GPUTokenizer(max_batch=total_batch, device=device)
        tokens, masks = tokenize_batched(states, hypothetical, tokenizer)

        # Query model
        q_values = query_model(model, tokens, masks, states, n_worlds, device)  # [N*batch, 7]
        q_batch = q_values.view(n_games, n_worlds, n_actions)  # [N, batch, 7]

        # Accumulate mean/variance statistics (convert to float32 for numerical stability)
        q_batch_f32 = q_batch.float()  # Ensure float32 for accumulation
        q_sum += q_batch_f32.sum(dim=1)  # [N, 7]
        q_sq_sum += (q_batch_f32 ** 2).sum(dim=1)  # [N, 7]
        n_total += n_worlds  # All games get same batch

        # Online PDF accumulation: bin Q-values and scatter-add counts
        # Bin Q-values: q in [-42, 42] -> bin in [0, 84]
        bins = (q_batch + 42).round().clamp(0, n_bins - 1).long()  # [N, batch, 7]

        # Compute flat indices: flat_idx = game_offset + bin * 7 + action
        bin_offset = bins * n_actions  # [N, batch, 7]
        flat_indices = (game_offset + bin_offset + action_idx).view(-1)  # [N * batch * 7]

        # Scatter-add 1.0 for each sample (unnormalized counts)
        ones = torch.ones(n_games * n_worlds * n_actions, device=device, dtype=torch.float32)
        pdf_counts.scatter_add_(0, flat_indices, ones)

        iteration += 1
        current_n = n_total[0].item()  # All games have same count

        # Check convergence (only after min_samples)
        if current_n >= min_samples:
            # Compute current mean and variance
            n_float = n_total.unsqueeze(1).float()  # [N, 1]
            e_q = q_sum / n_float  # [N, 7]
            e_q_var = (q_sq_sum / n_float) - e_q ** 2  # [N, 7]
            e_q_var = e_q_var.clamp(min=0)  # Numerical stability

            # Compute SEM = sqrt(var / n)
            sem = torch.sqrt(e_q_var / n_float)  # [N, 7]

            # Max SEM over legal actions per game
            # Set illegal actions to 0 so they don't affect max
            sem_legal = sem.clone()
            sem_legal[~legal_mask] = 0.0
            max_sem_per_game = sem_legal.max(dim=1).values  # [N]

            # Check convergence
            converged = max_sem_per_game < sem_threshold

            # Log convergence stats if log file is set
            if _ADAPTIVE_LOG is not None:
                n_converged = converged.sum().item()
                # Debug: check for NaN sources
                n_legal = legal_mask.sum(dim=1)  # [N] - legal actions per game
                sem_g0 = sem[0, legal_mask[0]].tolist() if legal_mask[0].any() else []
                mean_max_sem = max_sem_per_game.mean().item()
                max_max_sem = max_sem_per_game.max().item()
                seed_str = f"seeds={seeds[0]}-{seeds[-1]}, " if seeds else ""
                _ADAPTIVE_LOG.write(
                    f"{seed_str}decision={decision_idx}, samples={current_n}, "
                    f"converged={n_converged}/{n_games}, "
                    f"mean_max_sem={mean_max_sem:.4f}, max_max_sem={max_max_sem:.4f}\n"
                )
                _ADAPTIVE_LOG.flush()

            if converged.all():
                did_converge = True
                break

        # Check max samples limit
        if current_n >= max_samples:
            did_converge = False
            break
    else:
        # Loop completed without break (shouldn't happen, but handle it)
        did_converge = False

    # Final statistics
    n_float = n_total.unsqueeze(1).float()  # [N, 1]
    e_q = q_sum / n_float  # [N, 7]
    e_q_var = (q_sq_sum / n_float) - e_q ** 2  # [N, 7]
    e_q_var = e_q_var.clamp(min=0)

    # Normalize PDF counts to probabilities
    # Reshape: [N * 85 * 7] -> [N, 85, 7] -> [N, 7, 85]
    pdf_unnorm = pdf_counts.view(n_games, n_bins, n_actions)
    e_q_pdf = (pdf_unnorm / n_total.view(n_games, 1, 1).float()).permute(0, 2, 1).contiguous()

    # Ensure tensors are on correct device
    if e_q.device != states.hands.device:
        e_q = e_q.to(states.hands.device)
    if e_q_var.device != states.hands.device:
        e_q_var = e_q_var.to(states.hands.device)
    if e_q_pdf.device != states.hands.device:
        e_q_pdf = e_q_pdf.to(states.hands.device)

    # Return number of samples used
    n_samples_used = int(n_total[0].item())  # All games have same count

    return e_q, e_q_var, e_q_pdf, None, n_samples_used, did_converge


def sample_until_convergence_posterior(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    tokenizer: GPUTokenizer,
    model,
    adaptive_config: AdaptiveConfig,
    posterior_config: PosteriorConfig,
    device: str,
    decision_idx: int = 0,
    seeds: list[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, PosteriorDiagnostics | None, int, bool]:
    """Sample worlds until posterior-weighted E[Q] estimates converge.

    Like sample_until_convergence but uses posterior weighting. Tracks weighted
    running statistics and uses ESS (effective sample size) for convergence.

    Args:
        states: GameStateTensor with N games
        sampler: Pre-allocated world sampler
        tokenizer: Pre-allocated GPU tokenizer
        model: Stage 1 oracle model
        adaptive_config: Adaptive sampling configuration
        posterior_config: Posterior weighting configuration
        device: Device for computation
        decision_idx: Current decision index (for logging)
        seeds: Optional seed list (for logging)

    Returns:
        Tuple of (e_q, e_q_var, e_q_pdf, diagnostics, n_samples_used, did_converge)
    """
    n_games = states.n_games
    n_actions = 7
    n_bins = 85
    batch_size = adaptive_config.batch_size
    min_samples = adaptive_config.min_samples
    max_samples = adaptive_config.max_samples
    sem_threshold = adaptive_config.sem_threshold

    # Get legal actions mask for convergence checking
    legal_mask = states.legal_actions()  # [N, 7]

    # Weighted running statistics (all on GPU)
    # Track per-game: sum_w, sum_w_sq, sum_wq[7], sum_wq_sq[7]
    sum_w = torch.zeros(n_games, device=device, dtype=torch.float32)
    sum_w_sq = torch.zeros(n_games, device=device, dtype=torch.float32)
    sum_wq = torch.zeros(n_games, n_actions, device=device, dtype=torch.float32)
    sum_wq_sq = torch.zeros(n_games, n_actions, device=device, dtype=torch.float32)

    # Online PDF accumulation with weights
    pdf_weighted = torch.zeros(n_games * n_bins * n_actions, device=device, dtype=torch.float32)
    game_offset = torch.arange(n_games, device=device).view(n_games, 1, 1) * (n_bins * n_actions)
    action_idx = torch.arange(n_actions, device=device).view(1, 1, n_actions)

    # Track total samples and last diagnostics
    n_total = 0
    last_diagnostics = None

    iteration = 0
    did_converge = False

    while True:
        # 1. Sample a batch of worlds
        worlds = sample_worlds_batched(states, sampler, batch_size)  # [N, batch, 3, 7]
        n_worlds = worlds.shape[1]

        # 2. Build hypothetical deals
        hypothetical = build_hypothetical_deals(states, worlds)  # [N, batch, 4, 7]

        # 3. Tokenize current step
        total_batch = n_games * n_worlds
        if total_batch > tokenizer.max_batch:
            tokenizer = GPUTokenizer(max_batch=total_batch, device=device)
        tokens, masks = tokenize_batched(states, hypothetical, tokenizer)

        # 4. Query model for current step Q-values
        q_values = query_model(model, tokens, masks, states, n_worlds, device)  # [N*batch, 7]

        # 5. Compute posterior weights for this batch
        weights, diagnostics = compute_posterior_weights_batch(
            states=states,
            worlds=worlds,
            hypothetical=hypothetical,
            model=model,
            tokenizer=tokenizer,
            posterior_config=posterior_config,
            device=device,
        )  # weights: [N, batch], diagnostics: PosteriorDiagnostics

        last_diagnostics = diagnostics

        # 6. Accumulate weighted statistics (all GPU tensors)
        q_batch = q_values.view(n_games, n_worlds, n_actions).float()  # [N, batch, 7]
        # weights: [N, batch] -> [N, batch, 1] for broadcasting
        w = weights.unsqueeze(-1)  # [N, batch, 1]

        sum_w += weights.sum(dim=1)  # [N]
        sum_w_sq += (weights ** 2).sum(dim=1)  # [N]
        sum_wq += (w * q_batch).sum(dim=1)  # [N, 7]
        sum_wq_sq += (w * q_batch ** 2).sum(dim=1)  # [N, 7]

        # 7. Accumulate weighted PDF
        bins = (q_batch + 42).round().clamp(0, n_bins - 1).long()  # [N, batch, 7]
        bin_offset = bins * n_actions
        flat_indices = (game_offset + bin_offset + action_idx).view(-1)
        # Weight each sample's contribution to PDF
        weights_flat = weights.unsqueeze(-1).expand(n_games, n_worlds, n_actions).reshape(-1)
        pdf_weighted.scatter_add_(0, flat_indices, weights_flat)

        n_total += n_worlds
        iteration += 1

        # 8. Check convergence (after min_samples)
        if n_total >= min_samples:
            # Weighted mean and variance
            # Avoid division by zero
            sum_w_safe = sum_w.clamp(min=1e-10).unsqueeze(1)  # [N, 1]
            e_q = sum_wq / sum_w_safe  # [N, 7]
            e_q_var = (sum_wq_sq / sum_w_safe) - e_q ** 2  # [N, 7]
            e_q_var = e_q_var.clamp(min=0)

            # ESS = (sum_w)^2 / sum_w_sq per game
            ess = (sum_w ** 2) / sum_w_sq.clamp(min=1e-10)  # [N]

            # SEM = sqrt(var / ESS)
            sem = torch.sqrt(e_q_var / ess.unsqueeze(1).clamp(min=1))  # [N, 7]

            # Max SEM over legal actions
            sem_legal = sem.clone()
            sem_legal[~legal_mask] = 0.0
            max_sem_per_game = sem_legal.max(dim=1).values  # [N]

            # Check convergence
            converged = max_sem_per_game < sem_threshold

            # Log if enabled
            if _ADAPTIVE_LOG is not None:
                n_converged = converged.sum().item()
                mean_ess = ess.mean().item()
                mean_max_sem = max_sem_per_game.mean().item()
                max_max_sem = max_sem_per_game.max().item()
                seed_str = f"seeds={seeds[0]}-{seeds[-1]}, " if seeds else ""
                _ADAPTIVE_LOG.write(
                    f"{seed_str}decision={decision_idx}, samples={n_total}, "
                    f"converged={n_converged}/{n_games}, ESS={mean_ess:.0f}, "
                    f"mean_max_sem={mean_max_sem:.4f}, max_max_sem={max_max_sem:.4f}\n"
                )
                _ADAPTIVE_LOG.flush()

            if converged.all():
                did_converge = True
                break

        # Check max samples limit
        if n_total >= max_samples:
            did_converge = False
            break

    # Final weighted statistics
    sum_w_safe = sum_w.clamp(min=1e-10).unsqueeze(1)
    e_q = sum_wq / sum_w_safe
    e_q_var = (sum_wq_sq / sum_w_safe) - e_q ** 2
    e_q_var = e_q_var.clamp(min=0)

    # Normalize weighted PDF
    pdf_unnorm = pdf_weighted.view(n_games, n_bins, n_actions)
    e_q_pdf = (pdf_unnorm / sum_w.view(n_games, 1, 1).clamp(min=1e-10)).permute(0, 2, 1).contiguous()

    # Ensure tensors on correct device
    if e_q.device != states.hands.device:
        e_q = e_q.to(states.hands.device)
    if e_q_var.device != states.hands.device:
        e_q_var = e_q_var.to(states.hands.device)
    if e_q_pdf.device != states.hands.device:
        e_q_pdf = e_q_pdf.to(states.hands.device)

    # Convert dict diagnostics to PosteriorDiagnostics dataclass
    if last_diagnostics is not None:
        aggregated_diagnostics = PosteriorDiagnostics(
            ess=last_diagnostics['ess'].mean().item() if 'ess' in last_diagnostics else 0.0,
            max_w=last_diagnostics['max_w'].mean().item() if 'max_w' in last_diagnostics else 0.0,
            entropy=last_diagnostics['entropy'].mean().item() if 'entropy' in last_diagnostics else 0.0,
            k_eff=last_diagnostics['k_eff'].mean().item() if 'k_eff' in last_diagnostics else 0.0,
            n_invalid=0,
            n_illegal=0,
            window_nll=0.0,
            window_k_used=posterior_config.window_k,
        )
    else:
        aggregated_diagnostics = None

    return e_q, e_q_var, e_q_pdf, aggregated_diagnostics, n_total, did_converge

"""Metrics for domino model evaluation: Q-gap, blunder rate, regret stats."""

from typing import Dict

import torch
from torch import Tensor


def compute_qgap(logits: Tensor, qvals: Tensor, legal: Tensor, teams: Tensor) -> Tensor:
    """
    Compute Q-gap: oracle_best_q - oracle_q[pred_action] (after team sign).

    This measures how suboptimal the model choice is according to the oracle.

    Args:
        logits: Model output logits, shape (batch, 7)
        qvals: Oracle Q-values for each action, shape (batch, 7)
        legal: Legal action mask (1=legal, 0=illegal), shape (batch, 7)
        teams: Team assignment (0 or 1), shape (batch,)

    Returns:
        Mean Q-gap across the batch (scalar tensor)
    """
    # Mask illegal moves
    logits_masked = logits.masked_fill(legal == 0, float('-inf'))
    preds = logits_masked.argmax(dim=-1)

    # Team sign (Team 0 maximizes, Team 1 minimizes)
    team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(-1)
    q_signed = qvals * team_sign

    # Find optimal Q after masking illegal
    q_masked = torch.where(legal > 0, q_signed, torch.tensor(float('-inf'), device=logits.device))
    optimal_q = q_masked.max(dim=-1).values

    # Get predicted Q
    pred_q = q_signed.gather(1, preds.unsqueeze(-1)).squeeze(-1)

    # Gap (always positive for team 0 perspective)
    gap = optimal_q - pred_q
    return gap.mean()


def compute_qgaps_per_sample(
    logits: Tensor, qvals: Tensor, legal: Tensor, teams: Tensor
) -> Tensor:
    """
    Compute per-sample Q-gaps (not reduced to mean).

    Args:
        logits: Model output logits, shape (batch, 7)
        qvals: Oracle Q-values for each action, shape (batch, 7)
        legal: Legal action mask (1=legal, 0=illegal), shape (batch, 7)
        teams: Team assignment (0 or 1), shape (batch,)

    Returns:
        Q-gap for each sample, shape (batch,)
    """
    # Mask illegal moves
    logits_masked = logits.masked_fill(legal == 0, float('-inf'))
    preds = logits_masked.argmax(dim=-1)

    # Team sign (Team 0 maximizes, Team 1 minimizes)
    team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(-1)
    q_signed = qvals * team_sign

    # Find optimal Q after masking illegal
    q_masked = torch.where(legal > 0, q_signed, torch.tensor(float('-inf'), device=logits.device))
    optimal_q = q_masked.max(dim=-1).values

    # Get predicted Q
    pred_q = q_signed.gather(1, preds.unsqueeze(-1)).squeeze(-1)

    # Gap (always positive for team 0 perspective)
    return optimal_q - pred_q


def compute_blunder_rate(gaps: Tensor, threshold: float = 10.0) -> Tensor:
    """
    Fraction of moves with Q-gap > threshold.

    A "blunder" is a move that is significantly worse than optimal.

    Args:
        gaps: Q-gap values, shape (batch,) or scalar
        threshold: Q-gap threshold for blunder classification (default: 10.0)

    Returns:
        Fraction of blunders (scalar tensor)
    """
    return (gaps > threshold).float().mean()


def compute_regret_stats(gaps: Tensor) -> Dict[str, Tensor]:
    """
    Compute statistics on Q-gap (regret) distribution.

    Args:
        gaps: Q-gap values for each sample, shape (batch,)

    Returns:
        Dictionary with:
            mean: Mean regret
            p99: 99th percentile regret
            max: Maximum regret
            zero_rate: Fraction of samples with zero regret (optimal play)
    """
    return {
        'mean': gaps.mean(),
        'p99': torch.quantile(gaps.float(), 0.99),
        'max': gaps.max(),
        'zero_rate': (gaps == 0).float().mean(),
    }


def compute_per_slot_accuracy(
    logits: Tensor, targets: Tensor, legal: Tensor
) -> Tensor:
    """
    Compute accuracy stratified by which slot the oracle chose.

    Note: Accuracy is misleading when ties exist (multiple optimal actions).
    Use per-slot regret for a more meaningful metric.

    Args:
        logits: Model output logits, shape (batch, 7)
        targets: Oracle target actions, shape (batch,)
        legal: Legal action mask (1=legal, 0=illegal), shape (batch, 7)

    Returns:
        Accuracy for each slot 0-6, shape (7,). NaN for slots with no samples.
    """
    logits_masked = logits.masked_fill(legal == 0, float('-inf'))
    preds = logits_masked.argmax(dim=-1)
    correct = preds == targets

    accs = torch.zeros(7, device=logits.device)
    for slot in range(7):
        mask = targets == slot
        if mask.any():
            accs[slot] = correct[mask].float().mean()
        else:
            accs[slot] = float('nan')
    return accs


def compute_per_declaration_regret(gaps: Tensor, decl_ids: Tensor) -> Dict[str, Tensor]:
    """
    Compute mean regret for each declaration type (0-9).

    Args:
        gaps: Q-gap values, shape (batch,)
        decl_ids: Declaration ID for each sample, shape (batch,)

    Returns:
        Dictionary mapping 'decl_N' to mean regret for declaration N.
    """
    results = {}
    for decl in range(10):
        mask = decl_ids == decl
        if mask.any():
            results[f'decl_{decl}'] = gaps[mask].mean()
    return results


def compute_per_slot_tie_rate(
    logits: Tensor, qvals: Tensor, legal: Tensor, teams: Tensor
) -> Tensor:
    """
    For tied-Q situations, compute how often each slot is selected.

    When multiple slots share the max oracle Q-value, this measures
    which slot the model prefers. Uniform distribution (~14.3% each)
    indicates no slot bias in tie-breaking.

    Args:
        logits: Model output logits, shape (batch, 7)
        qvals: Oracle Q-values, shape (batch, 7)
        legal: Legal action mask, shape (batch, 7)
        teams: Team assignment (0 or 1), shape (batch,)

    Returns:
        Tie selection rate for each slot 0-6, shape (7,).
        Returns zeros if no tied samples exist.
    """
    # Apply team sign to get Q from current player's perspective
    team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(-1)
    q_signed = qvals * team_sign
    q_masked = torch.where(legal > 0, q_signed, torch.tensor(float('-inf'), device=logits.device))
    max_q = q_masked.max(dim=-1, keepdim=True).values

    # Identify ties: more than one slot has max_q
    is_max = (q_masked == max_q) & (legal > 0)
    n_ties = is_max.sum(dim=-1)
    has_tie = n_ties > 1

    if not has_tie.any():
        return torch.zeros(7, device=logits.device)

    # For tied samples, which slot did model pick?
    logits_masked = logits.masked_fill(legal == 0, float('-inf'))
    preds = logits_masked.argmax(dim=-1)

    # Count selections per slot among tied samples
    tie_preds = preds[has_tie]
    slot_counts = torch.bincount(tie_preds, minlength=7).float()
    return slot_counts / slot_counts.sum()  # Normalize to rate


def compute_per_slot_regret(
    gaps: Tensor, targets: Tensor
) -> Tensor:
    """
    Compute mean regret stratified by which slot the oracle chose.

    Args:
        gaps: Q-gap values, shape (batch,)
        targets: Oracle target actions (slot indices), shape (batch,)

    Returns:
        Mean regret for each slot 0-6, shape (7,). NaN for slots with no samples.
    """
    regrets = torch.zeros(7, device=gaps.device)
    for slot in range(7):
        mask = targets == slot
        if mask.any():
            regrets[slot] = gaps[mask].mean()
        else:
            regrets[slot] = float('nan')
    return regrets

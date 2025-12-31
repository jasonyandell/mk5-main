"""Metrics for domino model evaluation: Q-gap, blunder rate, accuracy."""

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


def compute_accuracy(logits: Tensor, targets: Tensor, legal: Tensor) -> Tensor:
    """
    Fraction of exact matches with oracle best move.

    Args:
        logits: Model output logits, shape (batch, 7)
        targets: Oracle target actions, shape (batch,)
        legal: Legal action mask (1=legal, 0=illegal), shape (batch, 7)

    Returns:
        Accuracy (scalar tensor)
    """
    logits_masked = logits.masked_fill(legal == 0, float('-inf'))
    preds = logits_masked.argmax(dim=-1)
    return (preds == targets).float().mean()

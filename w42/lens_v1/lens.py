"""Lens v1 — utility-conditioned Q-greedy player.

A "Lens" is a Q-greedy player parameterized by a utility function. The forge
oracle produces a per-action Q-distribution (PDF over 85 bins covering Q in
[-42, +42]); the utility function summarizes that distribution to a scalar
per legal action; the action is the argmax of those scalars.

Five utilities, matched to Wave 4.0 conventions
(`w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/analyze.py`):

  - ev          = mean of Q samples (forge `e_q`)
  - p_make      = P(Q >= contract_threshold) using forge bid-aware bins
  - mark_ev     = mm * (2*p_make - 1) where mm = max(1, bid // 42)
  - cvar_10     = mean of Q in lower-tail (cum mass <= 0.10)
  - robust_q25  = smallest Q with cum mass >= 0.25

For all five, "higher is better" because the forge Q tensor is already
POV-corrected by `query_model` (Q is from the acting player's seat — defender
wants high Q because it means setting the contract).

Tie-break: lowest legal slot index.
"""
from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.generate.actions import contract_threshold_bins, EQ_BIN_COUNT


UTILITIES = ("ev", "p_make", "mark_ev", "cvar_10", "robust_q25", "disaster")


def _q_values_tensor(device) -> Tensor:
    """Return the Q-bin-center tensor [85] of values -42..+42."""
    return torch.arange(-42, 43, dtype=torch.float32, device=device)


def utility_scores(
    utility: str,
    e_q: Tensor,            # [n_games, 7]
    e_q_pdf: Tensor,        # [n_games, 7, 85]
    bidder: Tensor,         # [n_games] int
    current_players: Tensor, # [n_games] int
    bid_values: list[int] | Tensor | None = None,
) -> Tensor:
    """Compute scalar utility values per (game, action). Higher is better.

    Returns: [n_games, 7] float32 tensor.
    """
    if utility not in UTILITIES:
        raise ValueError(f"Unknown utility: {utility!r}. Known: {UTILITIES}")

    n_games = e_q_pdf.shape[0]
    device = e_q_pdf.device

    if utility == "ev":
        return e_q.float()

    if utility in ("p_make", "mark_ev"):
        is_offense = ((current_players % 2) == (bidder % 2)).unsqueeze(1)  # [n,1]
        offense_bins, defense_bins = contract_threshold_bins(
            bid_values, n_games=n_games, device=device,
        )
        bins = torch.arange(EQ_BIN_COUNT, device=device).view(1, 1, EQ_BIN_COUNT)
        p_off = (e_q_pdf * (bins >= offense_bins.view(n_games, 1, 1))).sum(dim=2)
        p_def = (e_q_pdf * (bins >= defense_bins.view(n_games, 1, 1))).sum(dim=2)
        p_make = torch.where(is_offense, p_off, p_def)
        if utility == "p_make":
            return p_make.float()
        # mark_ev = mm * (2*p_make - 1); mm = max(1, bid // 42)
        if bid_values is None:
            bv = torch.full((n_games,), 30, dtype=torch.long, device=device)
        elif isinstance(bid_values, Tensor):
            bv = bid_values.to(device=device, dtype=torch.long).flatten()
        else:
            bv = torch.tensor(bid_values, dtype=torch.long, device=device).flatten()
        mm = torch.clamp(bv // 42, min=1).float().unsqueeze(1)  # [n,1]
        return (mm * (2.0 * p_make - 1.0)).float()

    # cvar_10 and robust_q25 both need cumulative pdf and Q-bin centers
    qvals = _q_values_tensor(device).view(1, 1, EQ_BIN_COUNT)  # [1,1,85]
    pdf = e_q_pdf / e_q_pdf.sum(dim=2, keepdim=True).clamp(min=1e-12)
    cum = torch.cumsum(pdf, dim=2)  # [n,7,85]

    if utility == "cvar_10":
        # mean of Q where cum <= 0.10. Use mass-weighted mean within mask.
        # Note: the mask may include many empty bins (cum still 0); what
        # matters is the mass within the mask (`wsum`). When the smallest
        # mass-bearing bin already exceeds 10% mass, wsum within mask is 0
        # and we fall back to the smallest mass-bearing bin (worst-case
        # tail location), matching Wave 4.0 analyze.py.
        mask = (cum <= 0.10).float()  # [n,7,85]
        w = pdf * mask
        wsum = w.sum(dim=2, keepdim=True)  # [n,7,1]
        cvar_w = (qvals * w).sum(dim=2, keepdim=True) / wsum.clamp(min=1e-12)
        # Fallback: first mass-bearing bin
        first_idx = (pdf > 0).float().argmax(dim=2, keepdim=True)  # [n,7,1]
        first_q = first_idx.float() - 42.0
        empty = (wsum <= 1e-12)
        cvar = torch.where(empty, first_q, cvar_w).squeeze(2)
        return cvar.float()

    if utility == "disaster":
        # "Anything below the make threshold is a disaster, treat it as Q=-42."
        # Per-bin Q values; for bins below the seat's make threshold, replace
        # the Q value with -42 BEFORE taking the expectation under the pdf.
        # Above-threshold bins keep their continuous Q value (so we still
        # distinguish "made by 1" from "made by 12"). The result is a clipped
        # EV that overweights the worst case.
        is_offense = ((current_players % 2) == (bidder % 2))  # [n]
        offense_bins, defense_bins = contract_threshold_bins(
            bid_values, n_games=n_games, device=device,
        )
        threshold_bin = torch.where(is_offense, offense_bins, defense_bins)  # [n]
        bins = torch.arange(EQ_BIN_COUNT, device=device).view(1, 1, EQ_BIN_COUNT)  # [1,1,85]
        # broadcast threshold to [n, 1, 1] so we can compare
        above_thresh = (bins >= threshold_bin.view(n_games, 1, 1))  # [n,1,85]
        qvals = torch.arange(-42, 43, dtype=torch.float32, device=device).view(1, 1, EQ_BIN_COUNT)
        # For above-threshold bins, use the bin's Q value; below, use -42.
        clipped_q = torch.where(above_thresh, qvals, torch.full_like(qvals, -42.0))
        # Expectation under pdf, broadcasting across the 7 action slots
        return (e_q_pdf * clipped_q).sum(dim=2).float()

    if utility == "robust_q25":
        # smallest Q with cum >= 0.25 -> argmax over (cum >= 0.25) returns
        # first True bin (earliest threshold-cross). PyTorch argmax on bool
        # treats float; convert.
        mask = (cum >= 0.25).float()
        # Add a tiny epsilon decreasing across bins so argmax of mask returns
        # the FIRST True (lowest) index.
        order = torch.arange(EQ_BIN_COUNT, device=device, dtype=torch.float32)
        score = mask - 1e-3 * order.view(1, 1, EQ_BIN_COUNT)
        idx = score.argmax(dim=2)  # [n,7]
        # If no bin reaches 0.25 (shouldn't happen with normalized pdf), idx=0.
        q25 = idx.float() - 42.0
        return q25.float()

    raise AssertionError("unreachable")


def argmax_under_utility(
    utility: str,
    e_q: Tensor,
    e_q_pdf: Tensor,
    bidder: Tensor,
    current_players: Tensor,
    legal_mask: Tensor,
    bid_values: list[int] | Tensor | None = None,
) -> Tensor:
    """Pick best legal slot per game under the named utility.

    Tie-break: lowest legal slot index (PyTorch argmax returns the first
    occurrence of the max, so we add a tiny -slot_idx perturbation to break
    ties downward).

    Returns: [n_games] long tensor of action indices.
    """
    scores = utility_scores(
        utility, e_q, e_q_pdf, bidder, current_players, bid_values,
    )  # [n_games, 7]

    # Mask illegal slots to -inf
    masked = torch.where(
        legal_mask, scores, torch.full_like(scores, float("-inf")),
    )

    # Tie-break: torch.argmax returns the FIRST occurrence of the max, which
    # is the lowest slot index — exactly the Wave 4.0 convention.
    actions = masked.argmax(dim=1)
    return actions.long()

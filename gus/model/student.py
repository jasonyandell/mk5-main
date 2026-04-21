"""Student network: shared state encoder + heads.

v0: encoder + belief_head only. v1 will add V_head, π_me_head, and the
world-conditioned Q_head. Scaffolding is set up to extend cleanly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import FEATURE_DIM, N_DOMINOES

DEFAULT_HIDDEN_DIM = 256


class StateEncoder(nn.Module):
    """Simple MLP state encoder. 2 hidden layers, GELU, layer-norm."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM, out_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BeliefHead(nn.Module):
    """Per-domino 3-way classifier: P(domino ∈ {left_opp, partner, right_opp}).

    Output shape: [B, 28, 3] logits. Downstream cross-entropy loss masked to
    unseen dominoes.
    """

    def __init__(self, in_dim: int = DEFAULT_HIDDEN_DIM, n_dominoes: int = N_DOMINOES, n_seats: int = 3):
        super().__init__()
        self.n_dominoes = n_dominoes
        self.n_seats = n_seats
        self.proj = nn.Linear(in_dim, n_dominoes * n_seats)

    def forward(self, z: Tensor) -> Tensor:
        logits = self.proj(z)  # [B, 28*3]
        return logits.view(-1, self.n_dominoes, self.n_seats)  # [B, 28, 3]


class StudentV0(nn.Module):
    """State encoder + belief head. v0 sanity-check student."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.encoder = StateEncoder(hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.belief = BeliefHead(in_dim=hidden_dim)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        z = self.encoder(features)
        belief_logits = self.belief(z)
        return {"belief_logits": belief_logits, "z": z}


def belief_loss(
    logits: Tensor,    # [B, 28, 3]
    target: Tensor,    # [B, 28] long in {0,1,2}
    mask: Tensor,      # [B, 28] bool
) -> Tensor:
    """Cross-entropy over unseen-domino slots only. Returns scalar mean."""
    # Flatten to [B*28, 3] and [B*28] / [B*28]
    B, D, S = logits.shape
    flat_logits = logits.reshape(B * D, S)
    flat_target = target.reshape(B * D)
    flat_mask = mask.reshape(B * D)
    if flat_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    # Select only masked positions
    sel_logits = flat_logits[flat_mask]
    sel_target = flat_target[flat_mask]
    return nn.functional.cross_entropy(sel_logits, sel_target)


def belief_accuracy(
    logits: Tensor,    # [B, 28, 3]
    target: Tensor,    # [B, 28]
    mask: Tensor,      # [B, 28]
) -> tuple[int, int]:
    """Return (n_correct, n_total) over unseen-domino slots."""
    preds = logits.argmax(dim=-1)  # [B, 28]
    sel = mask
    n_total = int(sel.sum().item())
    if n_total == 0:
        return 0, 0
    n_correct = int(((preds == target) & sel).sum().item())
    return n_correct, n_total

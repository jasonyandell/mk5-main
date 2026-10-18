# SPDX-FileCopyrightText: 2022 Andrej Karpathy
# SPDX-FileCopyrightText: 2026 Jason Yandell
# SPDX-License-Identifier: MIT
"""Same nanoGPT backbone and inputs; candidate identity is an output axis."""
import torch
from torch import nn
from torch.nn import functional as F

from ..model import Config, Observer

TASK = "next-tile-and-bid30-splat-v1"


class SplatObserver(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.backbone = Observer(config)
        self.outcomes = nn.Linear(config.width, 28 * 43)
        self.outcomes.apply(Observer._init)

    def forward(self, tokens, positions=None, inspect=False):
        """positions selects output locations only, never changes the backbone."""
        b = self.backbone
        if tokens.shape[1] > self.config.context:
            raise ValueError("episode exceeds context")
        x = b.token(tokens) + b.position(torch.arange(tokens.shape[1], device=tokens.device))
        traces = []
        for block in b.blocks:
            x, trace = block(x)
            if inspect:
                traces.append({k: v.detach() for k, v in trace.items()})
        x = b.norm(x)
        tile = F.linear(x, b.token.weight[:28])
        if positions is not None:
            x = x[torch.arange(len(x), device=x.device)[:, None], positions]
        splat = self.outcomes(x).reshape(*x.shape[:-1], 28, 43)
        return (tile, splat, traces) if inspect else (tile, splat)


def joint_loss(tile_logits, tile_targets, splat_logits, histograms, candidates):
    """Equal objective weights; equal positions, then equal legal candidates.

    Zero rows mean unavailable teacher labels, not a zero-valued outcome.
    The candidate labels select OUTPUT rows; no candidate tokens enter the net.
    """
    tile_loss = F.cross_entropy(tile_logits.reshape(-1, 28), tile_targets.reshape(-1))
    chosen = splat_logits.gather(-2, candidates[..., None].expand(*candidates.shape, 43))
    mask = histograms.sum(-1) > 0
    ce = -(histograms * chosen.log_softmax(-1)).sum(-1)
    count = mask.sum(-1)
    position_loss = (ce * mask).sum(-1) / count.clamp_min(1)
    splat_loss = position_loss.sum() / (count > 0).sum().clamp_min(1)
    return tile_loss + splat_loss, tile_loss, splat_loss

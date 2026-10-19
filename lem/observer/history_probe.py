# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Temporary attention-edge interventions, never a player or input transform."""
from contextlib import contextmanager
import math

import torch

from .data import PREFIX, SEAT


def completed_tile_masks(tokens):
    """One unpadded acting-seat prefix -> four equally sized tile-key masks.

    Incomplete tricks, actor tokens, own-hand metadata and the query are kept.
    Relative seats are self=0, left=1, partner=2, right=3.
    """
    if tokens.ndim != 1 or len(tokens) < PREFIX + 1 or (len(tokens) - PREFIX - 1) % 2:
        raise ValueError('expected one complete acting-seat prefix')
    if int(tokens[-1]) != SEAT:
        raise ValueError('query must belong to viewer')
    plays = (len(tokens) - PREFIX - 1) // 2
    masks = torch.zeros((4, len(tokens)), dtype=torch.bool, device=tokens.device)
    for start in range(0, plays - plays % 4, 4):
        actors = [int(tokens[PREFIX + 2 * j]) - SEAT for j in range(start, start + 4)]
        if sorted(actors) != list(range(4)):
            raise ValueError('completed trick must contain each seat once')
        for j, actor in zip(range(start, start + 4), actors):
            pos = PREFIX + 2 * j + 1
            if not 0 <= int(tokens[pos]) < 28:
                raise ValueError('expected tile token')
            masks[actor, pos] = True
    return masks


@contextmanager
def attenuate_keys(model, mask, strength=1.0, last_only=False):
    """Multiply selected keys' unnormalized attention by 1-strength.

    Applied at every query and head in all layers (or the last layer only).
    strength=1 is a hard knockout, .5 halves weights before renormalization.
    No parameter changes. The caller supplies a boolean [batch,time] key mask.
    Use only on eval models; concurrent forwards on the same instance are unsafe.
    """
    if model.training or not 0 <= strength <= 1:
        raise ValueError('eval mode and strength in [0,1] required')
    if mask.ndim != 2 or mask.dtype != torch.bool or mask[:, 0].any():
        raise ValueError('boolean [batch,time] mask must preserve first key')
    backbone = getattr(model, 'backbone', model)

    def intervene(module, arguments, output):
        x = arguments[0]
        b, t, w = x.shape
        if mask.shape != (b, t) or mask.device != x.device:
            raise ValueError('mask must match current batch, time and device')
        if strength == 0 or not mask.any():
            return output
        q, k, v = [a.reshape(b, t, module.heads, w // module.heads).transpose(1, 2)
                   for a in module.qkv(x).chunk(3, -1)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(w // module.heads)
        scores = scores.masked_fill(~module.causal[:t, :t], -torch.inf)
        if strength == 1:
            scores = scores.masked_fill(mask[:, None, None, :], -torch.inf)
        else:
            scores = scores + mask[:, None, None, :] * math.log1p(-strength)
        weights = scores.softmax(-1)
        attended = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, w)
        return module.projection(attended), weights

    handles = []
    try:
        for block in (backbone.blocks[-1:] if last_only else backbone.blocks):
            handles.append(block.attention.register_forward_hook(intervene))
        yield
    finally:
        for handle in handles:
            handle.remove()

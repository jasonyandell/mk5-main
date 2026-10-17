# SPDX-FileCopyrightText: 2022 Andrej Karpathy
# SPDX-FileCopyrightText: 2026 Jason Yandell
# SPDX-License-Identifier: MIT
"""Small adaptation of nanoGPT model.py (3adf61e154c3fe3fca428ad6bc3818b27a3b8291).

Pre-LN residual blocks, learned positions, causal attention, GELU MLP and tied
embeddings. Context includes metadata; the prediction vocabulary is 28 tiles.
See LICENSE.nanoGPT. No dropout, so an inspection follows the training forward
path exactly. Attention is explicit: these sequences are only 68 tokens long.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    width: int = 64
    layers: int = 2
    heads: int = 2
    context: int = 68
    vocabulary: int = 46

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.context) < 1 or self.width % self.heads:
            raise ValueError("positive sizes; width must divide into heads")


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.projection = nn.Linear(c.width, c.width)
        self.register_buffer("causal", torch.tril(torch.ones(c.context, c.context, dtype=torch.bool)),
                             persistent=False)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = [a.reshape(b, t, self.heads, w // self.heads).transpose(1, 2)
                   for a in self.qkv(x).chunk(3, dim=-1)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(w // self.heads)
        weights = scores.masked_fill(~self.causal[:t, :t], float("-inf")).softmax(-1)
        y = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, w)
        return self.projection(y), weights


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_attention = nn.LayerNorm(c.width)
        self.attention = Attention(c)
        self.ln_mlp = nn.LayerNorm(c.width)
        self.expand = nn.Linear(c.width, 4 * c.width)
        self.projection = nn.Linear(4 * c.width, c.width)

    def forward(self, x):
        attended, weights = self.attention(self.ln_attention(x))
        x = x + attended
        pre = self.expand(self.ln_mlp(x))
        post = F.gelu(pre)
        return x + self.projection(post), {"attention": weights, "pre": pre, "post": post}


class Observer(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocabulary, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._init)
        for name, parameter in self.named_parameters():
            if name.endswith("projection.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, tokens, inspect=False):
        if tokens.shape[1] > self.config.context:
            raise ValueError("episode exceeds context")
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        traces = []
        for block in self.blocks:
            x, trace = block(x)
            if inspect:
                traces.append({k: v.detach() for k, v in trace.items()})
        logits = F.linear(self.norm(x), self.token.weight[:28])
        return (logits, traces) if inspect else logits

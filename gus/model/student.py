"""Student network: shared state encoder + heads.

v0: MLP encoder + belief_head only — FAILED to generalize (overfits to
~34% eval acc ≈ chance). Kept for reference.

v1: transformer encoder over tokenized play sequence + belief_head. The
transformer sees each play as a positional token, enabling it to learn
void inference attentionally — the signal v0's bag-of-features couldn't
express.

Future: V_head, π_me_head, world-conditioned Q_head on top of the
transformer state encoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import FEATURE_DIM, N_DOMINOES
from .tokenize import (
    N_PLAYER_REL,
    N_TRICK_POS,
    N_TRICK_SLOTS,
    N_TYPES,
    SEQ_LEN,
    TOKEN_VOCAB_SIZE,
)

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


class PlayEmbedding(nn.Module):
    """Sum of embeddings for the five channels of a play token.

    Input: [B, L, 5] long
    Output: [B, L, d_model]

    Note: attribute names avoid shadowing nn.Module methods (`type`, `float`,
    etc.) — hence the `_emb` suffix.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.tok_emb = nn.Embedding(TOKEN_VOCAB_SIZE, d_model)
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        self.trick_emb = nn.Embedding(N_TRICK_SLOTS, d_model)
        self.pos_emb = nn.Embedding(N_TRICK_POS, d_model)
        self.player_rel_emb = nn.Embedding(N_PLAYER_REL, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, L, 5] where channels are (token, type, trick, pos, player_rel)
        t = tokens[..., 0]
        ty = tokens[..., 1]
        tr = tokens[..., 2]
        po = tokens[..., 3]
        pr = tokens[..., 4]
        emb = (
            self.tok_emb(t)
            + self.type_emb(ty)
            + self.trick_emb(tr)
            + self.pos_emb(po)
            + self.player_rel_emb(pr)
        )
        return self.norm(emb)


class TransformerEncoder(nn.Module):
    """Small transformer over the [L, 5] play-sequence tokens.

    Attention mask: PAD positions (token==PAD) are excluded from attention.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = PlayEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> Tensor:
        """
        tokens: [B, L, 5] long
        attention_mask: [B, L] bool — True where real
        Returns: [B, L, d_model]
        """
        x = self.embed(tokens)  # [B, L, d_model]
        # TransformerEncoder expects src_key_padding_mask as True where PAD.
        padding_mask = ~attention_mask  # True where pad
        x = self.blocks(x, src_key_padding_mask=padding_mask)
        return x


class StudentTransformer(nn.Module):
    """Transformer-encoder student. v1 — belief head only for now; heads for
    V/π_me/world-Q will be added after belief is proven to learn."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        # Use the CLS position (index 0) as the pooled state embedding.
        self.belief = BeliefHead(in_dim=d_model)

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        h = self.encoder(tokens, attention_mask)  # [B, L, d_model]
        cls_h = h[:, 0, :]  # [B, d_model] — CLS token's encoding
        belief_logits = self.belief(cls_h)  # [B, 28, 3]
        return {"belief_logits": belief_logits, "z": cls_h, "hidden": h}


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

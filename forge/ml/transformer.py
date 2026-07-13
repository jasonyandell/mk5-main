"""DominoTransformer — pure PyTorch model architecture.

No Lightning dependency.  Imported by both the Lightning training wrapper
(forge.ml.module) and lightweight inference paths (forge.zeb.eval.loading).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DominoTransformer(nn.Module):
    """
    The actual model architecture for domino play prediction.

    Extracted from train_pretokenized.py - keeps architecture code
    separate from Lightning wrapper.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature embeddings for the token representation
        self.high_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.low_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.is_double_embed = nn.Embedding(2, embed_dim // 12)
        self.count_value_embed = nn.Embedding(3, embed_dim // 12)
        self.trump_rank_embed = nn.Embedding(8, embed_dim // 6)
        self.player_id_embed = nn.Embedding(4, embed_dim // 12)
        self.is_current_embed = nn.Embedding(2, embed_dim // 12)
        self.is_partner_embed = nn.Embedding(2, embed_dim // 12)
        self.is_remaining_embed = nn.Embedding(2, embed_dim // 12)
        self.token_type_embed = nn.Embedding(8, embed_dim // 6)
        self.decl_embed = nn.Embedding(10, embed_dim // 6)
        self.leader_embed = nn.Embedding(4, embed_dim // 12)

        self.input_proj = nn.Linear(self._calc_input_dim(), embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        # Disable nested tensor fast-paths for CUDA graph capture stability.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
            mask_check=False,
        )
        self.output_proj = nn.Linear(embed_dim, 1)  # Action logits
        self.value_head = nn.Linear(embed_dim, 1)   # State value prediction

        # Constant offsets [0..6] for gathering the current player's hand tokens.
        # Non-persistent so it does not affect checkpoint compatibility.
        self.register_buffer("_hand_offsets", torch.arange(7, dtype=torch.int64), persistent=False)

    def _calc_input_dim(self) -> int:
        """Calculate total dimension of concatenated embeddings."""
        return (
            self.embed_dim // 6 * 4 +
            self.embed_dim // 12 * 6 +
            self.embed_dim // 6 +
            self.embed_dim // 12
        )

    def forward(self, tokens: Tensor, mask: Tensor, current_player: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            tokens: Input token tensor, shape (batch, seq_len, 12)
            mask: Attention mask, shape (batch, seq_len)
            current_player: Current player index, shape (batch,)

        Returns:
            Tuple of:
                logits: Action logits, shape (batch, 7)
                value: State value prediction, shape (batch,)
        """
        device = tokens.device

        # Build embeddings from token features
        embeds = [
            self.high_pip_embed(tokens[:, :, 0]),
            self.low_pip_embed(tokens[:, :, 1]),
            self.is_double_embed(tokens[:, :, 2]),
            self.count_value_embed(tokens[:, :, 3]),
            self.trump_rank_embed(tokens[:, :, 4]),
            self.player_id_embed(tokens[:, :, 5]),
            self.is_current_embed(tokens[:, :, 6]),
            self.is_partner_embed(tokens[:, :, 7]),
            self.is_remaining_embed(tokens[:, :, 8]),
            self.token_type_embed(tokens[:, :, 9]),
            self.decl_embed(tokens[:, :, 10]),
            self.leader_embed(tokens[:, :, 11]),
        ]

        x = torch.cat(embeds, dim=-1)
        x = self.input_proj(x)

        # Apply transformer with attention mask
        attn_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract hand representations for current player
        # Player's 7 dominoes start at index 1 + player_id * 7
        start_indices = 1 + current_player * 7
        gather_indices = start_indices.unsqueeze(1) + self._hand_offsets.unsqueeze(0)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        hand_repr = torch.gather(x, dim=1, index=gather_indices)
        logits = self.output_proj(hand_repr).squeeze(-1)

        # Value prediction: mean pool over valid tokens
        # Use context token (always present at position 0) for value
        value = self.value_head(x[:, 0, :]).squeeze(-1)

        return logits, value

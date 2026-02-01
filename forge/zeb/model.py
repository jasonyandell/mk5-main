"""Zeb neural network architecture for self-play."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ZebEmbeddings(nn.Module):
    """Embeddings for 8-feature token representation."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Compute per-feature embedding dims (should sum to embed_dim)
        base = embed_dim // 8

        self.high_pip_embed = nn.Embedding(7, base)       # 0-6
        self.low_pip_embed = nn.Embedding(7, base)        # 0-6
        self.is_double_embed = nn.Embedding(2, base // 2)  # 0-1
        self.count_embed = nn.Embedding(3, base // 2)     # 0-2
        self.player_embed = nn.Embedding(4, base)         # 0-3 relative
        self.is_in_hand_embed = nn.Embedding(2, base // 2)  # 0-1
        self.decl_embed = nn.Embedding(10, base)          # 0-9
        self.token_type_embed = nn.Embedding(3, base)     # 0-2

        # Project concatenated embeddings to embed_dim
        total_dim = base * 5 + (base // 2) * 3
        self.proj = nn.Linear(total_dim, embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [batch, seq_len, 8] -> [batch, seq_len, embed_dim]"""
        embeds = [
            self.high_pip_embed(tokens[:, :, 0]),
            self.low_pip_embed(tokens[:, :, 1]),
            self.is_double_embed(tokens[:, :, 2]),
            self.count_embed(tokens[:, :, 3]),
            self.player_embed(tokens[:, :, 4]),
            self.is_in_hand_embed(tokens[:, :, 5]),
            self.decl_embed(tokens[:, :, 6]),
            self.token_type_embed(tokens[:, :, 7]),
        ]
        x = torch.cat(embeds, dim=-1)
        return self.proj(x)


class ZebModel(nn.Module):
    """Policy + Value network for self-play."""

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_tokens: int = 36,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

        self.embeddings = ZebEmbeddings(embed_dim)
        self.pos_embed = nn.Embedding(max_tokens, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy head: projects hand slot tokens to action logits
        self.policy_proj = nn.Linear(embed_dim, 1)

        # Value head: uses CLS-like aggregation
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(
        self,
        tokens: Tensor,       # [batch, seq_len, 8]
        mask: Tensor,         # [batch, seq_len] bool, True = valid
        hand_indices: Tensor,  # [batch, 7] indices of hand slots in sequence
        hand_mask: Tensor,    # [batch, 7] bool, True = valid hand slot
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            policy: [batch, 7] unnormalized logits for each hand slot
            value: [batch] scalar value predictions
        """
        batch_size, seq_len, _ = tokens.shape

        # Embed tokens
        x = self.embeddings(tokens)  # [batch, seq, embed_dim]

        # Add positional embeddings
        pos = torch.arange(seq_len, device=tokens.device)
        x = x + self.pos_embed(pos)

        # Transformer with attention mask (True = masked/ignored)
        attn_mask = ~mask  # Invert for PyTorch convention
        x = self.encoder(x, src_key_padding_mask=attn_mask)

        # Policy: gather hand slot embeddings and project
        # hand_indices: [batch, 7], need to gather from x: [batch, seq, embed]
        hand_embeds = torch.gather(
            x, 1,
            hand_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # [batch, 7, embed_dim]
        policy = self.policy_proj(hand_embeds).squeeze(-1)  # [batch, 7]

        # Value: mean pool over valid tokens
        x_masked = x * mask.unsqueeze(-1)
        pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        value = self.value_head(pooled).squeeze(-1)  # [batch]

        return policy, value

    def get_action(
        self,
        tokens: Tensor,
        mask: Tensor,
        hand_indices: Tensor,
        hand_mask: Tensor,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample action and return log_prob + value.

        Returns:
            action: [batch] sampled slot indices
            log_prob: [batch] log probability of sampled action
            value: [batch] value estimates
        """
        policy, value = self(tokens, mask, hand_indices, hand_mask)

        # Mask illegal actions
        policy = policy.masked_fill(~hand_mask, float('-inf'))

        # Sample with temperature
        probs = F.softmax(policy / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value


def get_model_config(size: str = 'small') -> dict:
    """Get model hyperparameters by size."""
    configs = {
        'small': dict(embed_dim=64, n_heads=2, n_layers=2, ff_dim=128),
        'medium': dict(embed_dim=128, n_heads=4, n_layers=4, ff_dim=256),
        'large': dict(embed_dim=256, n_heads=8, n_layers=6, ff_dim=512),
    }
    return configs[size]

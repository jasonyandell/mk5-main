#!/usr/bin/env python3
"""
Transformer Move Prediction Diagnostic.

Tests if transformer attention can learn cross-seed generalizable patterns
for Texas 42 move selection. The hypothesis: self-attention computes pairwise
token interactions - exactly what's needed for "does my trump beat their trump?"

Architecture (from bead t42-1d1g):
- 29-32 tokens: context + 4 player hands + current trick
- Embedding dim: 64
- Layers: 2
- Attention heads: 4
- Output: 7-way classification (which domino to play)

Success metric: Cross-seed accuracy close to within-seed accuracy.
The MLP baseline showed 1.8x gap; target is < 1.3x.

Usage:
    python scripts/solver2/train_transformer.py --max-samples 10000  # Quick test
    python scripts/solver2/train_transformer.py                      # Full run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.declarations import (
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    N_DECLS,
    NOTRUMP,
    PIP_TRUMP_IDS,
)
from scripts.solver2.rng import deal_from_seed
from scripts.solver2.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    is_in_called_suit,
    trick_rank,
)


def log(msg: str) -> None:
    """Print with timestamp and flush."""
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


LOG_START_TIME = time.time()


# =============================================================================
# Trump Rank Computation
# =============================================================================

def get_trump_rank(domino_id: int, decl_id: int) -> int:
    """Return trump rank 0-6 (0=boss) or 7 if not trump."""
    if decl_id == NOTRUMP:
        return 7
    if not is_in_called_suit(domino_id, decl_id):
        return 7

    # Get all trumps and their trick_rank values
    trumps = []
    for d in range(28):
        if is_in_called_suit(d, decl_id):
            tau = trick_rank(d, 7, decl_id)  # led_suit=7 means trump suit
            trumps.append((d, tau))

    # Sort by tau descending (highest = boss)
    trumps.sort(key=lambda x: -x[1])

    # Find rank of this domino
    for rank, (d, _) in enumerate(trumps):
        if d == domino_id:
            return rank
    return 7


# Precompute trump ranks for all (domino, decl) pairs
TRUMP_RANK_TABLE = {}
for _decl in range(N_DECLS):
    for _dom in range(28):
        TRUMP_RANK_TABLE[(_dom, _decl)] = get_trump_rank(_dom, _decl)


# =============================================================================
# Tokenization
# =============================================================================

# Token type IDs for segment embedding
TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_PLAYER1 = 2
TOKEN_TYPE_PLAYER2 = 3
TOKEN_TYPE_PLAYER3 = 4
TOKEN_TYPE_TRICK_P0 = 5
TOKEN_TYPE_TRICK_P1 = 6
TOKEN_TYPE_TRICK_P2 = 7

# Feature dimensions for each embedding table
EMBED_HIGH_PIP = 7      # 0-6
EMBED_LOW_PIP = 7       # 0-6
EMBED_IS_DOUBLE = 2     # bool
EMBED_COUNT_VALUE = 3   # 0, 5, 10 -> 0, 1, 2
EMBED_TRUMP_RANK = 8    # 0-6 for ranks, 7 for non-trump
EMBED_PLAYER_ID = 4     # 0-3
EMBED_IS_CURRENT = 2    # bool
EMBED_IS_PARTNER = 2    # bool
EMBED_IS_REMAINING = 2  # bool (for hand tokens)
EMBED_DECL = 10         # declaration type
EMBED_LEADER = 4        # who leads

# Count value mapping
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


def tokenize_sample(
    state: int,
    seed: int,
    decl_id: int,
    hands: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Tokenize a game state into transformer input.

    Returns:
        tokens: (max_tokens, n_features) - padded token features
        mask: (max_tokens,) - 1 for real tokens, 0 for padding
        current_player: int - who is to move
    """
    # Extract state fields
    remaining = [(state >> (p * 7)) & 0x7F for p in range(4)]
    leader = (state >> 28) & 0x3
    trick_len = (state >> 30) & 0x3
    p0_local = (state >> 32) & 0x7
    p1_local = (state >> 35) & 0x7
    p2_local = (state >> 38) & 0x7

    current_player = (leader + trick_len) % 4
    partner = (current_player + 2) % 4

    # Feature structure for each token:
    # [high_pip, low_pip, is_double, count_value, trump_rank,
    #  player_id, is_current, is_partner, is_remaining, token_type,
    #  decl_id, leader_id]
    N_FEATURES = 12
    MAX_TOKENS = 32  # 1 context + 28 hand + 3 trick

    tokens = np.zeros((MAX_TOKENS, N_FEATURES), dtype=np.int64)
    mask = np.zeros(MAX_TOKENS, dtype=np.float32)

    token_idx = 0

    # Token 0: Context token (declaration + leader)
    tokens[token_idx] = [
        0, 0, 0, 0, 0,  # no domino features
        0, 0, 0, 0,     # no player segment
        TOKEN_TYPE_CONTEXT,
        decl_id,
        leader,
    ]
    mask[token_idx] = 1.0
    token_idx += 1

    # Tokens 1-28: All 28 hand positions (4 players × 7 dominoes)
    for player in range(4):
        is_current = 1 if player == current_player else 0
        is_partner_flag = 1 if player == partner else 0

        for local_idx in range(7):
            global_id = hands[player][local_idx]
            has_domino = (remaining[player] >> local_idx) & 1

            tokens[token_idx] = [
                DOMINO_HIGH[global_id],
                DOMINO_LOW[global_id],
                1 if DOMINO_IS_DOUBLE[global_id] else 0,
                COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]],
                TRUMP_RANK_TABLE[(global_id, decl_id)],
                player,
                is_current,
                is_partner_flag,
                has_domino,
                TOKEN_TYPE_PLAYER0 + player,
                decl_id,
                leader,
            ]
            mask[token_idx] = 1.0
            token_idx += 1

    # Tokens 29-31: Current trick plays (if any)
    trick_plays = [p0_local, p1_local, p2_local]
    trick_token_types = [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P1, TOKEN_TYPE_TRICK_P2]

    for i in range(trick_len):
        local_idx = trick_plays[i]
        if local_idx >= 7:
            continue  # No play at this position

        play_player = (leader + i) % 4
        global_id = hands[play_player][local_idx]

        is_current = 1 if play_player == current_player else 0
        is_partner_flag = 1 if play_player == partner else 0

        tokens[token_idx] = [
            DOMINO_HIGH[global_id],
            DOMINO_LOW[global_id],
            1 if DOMINO_IS_DOUBLE[global_id] else 0,
            COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]],
            TRUMP_RANK_TABLE[(global_id, decl_id)],
            play_player,
            is_current,
            is_partner_flag,
            0,  # not in hand anymore
            trick_token_types[i],
            decl_id,
            leader,
        ]
        mask[token_idx] = 1.0
        token_idx += 1

    return tokens, mask, current_player


# =============================================================================
# Transformer Model
# =============================================================================

class DominoTransformer(nn.Module):
    """
    Transformer for move prediction in Texas 42.

    Input: Tokenized game state (29-32 tokens)
    Output: 7-way classification over current player's legal moves
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

        # Feature embeddings (compositional)
        self.high_pip_embed = nn.Embedding(EMBED_HIGH_PIP, embed_dim // 6)
        self.low_pip_embed = nn.Embedding(EMBED_LOW_PIP, embed_dim // 6)
        self.is_double_embed = nn.Embedding(EMBED_IS_DOUBLE, embed_dim // 12)
        self.count_value_embed = nn.Embedding(EMBED_COUNT_VALUE, embed_dim // 12)
        self.trump_rank_embed = nn.Embedding(EMBED_TRUMP_RANK, embed_dim // 6)
        self.player_id_embed = nn.Embedding(EMBED_PLAYER_ID, embed_dim // 12)
        self.is_current_embed = nn.Embedding(EMBED_IS_CURRENT, embed_dim // 12)
        self.is_partner_embed = nn.Embedding(EMBED_IS_PARTNER, embed_dim // 12)
        self.is_remaining_embed = nn.Embedding(EMBED_IS_REMAINING, embed_dim // 12)
        self.token_type_embed = nn.Embedding(8, embed_dim // 6)  # 8 token types
        self.decl_embed = nn.Embedding(EMBED_DECL, embed_dim // 6)
        self.leader_embed = nn.Embedding(EMBED_LEADER, embed_dim // 12)

        # Projection to full embed_dim
        # Sum of fractional dims: 64/6*4 + 64/12*6 = ~42 + 32 = ~74
        # We'll project to embed_dim
        self.input_proj = nn.Linear(self._calc_input_dim(), embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: takes the 7 hand tokens for current player
        # and produces 7 logits
        self.output_proj = nn.Linear(embed_dim, 1)

    def _calc_input_dim(self) -> int:
        """Calculate input dimension from embedding sizes."""
        return (
            self.embed_dim // 6 +   # high_pip
            self.embed_dim // 6 +   # low_pip
            self.embed_dim // 12 +  # is_double
            self.embed_dim // 12 +  # count_value
            self.embed_dim // 6 +   # trump_rank
            self.embed_dim // 12 +  # player_id
            self.embed_dim // 12 +  # is_current
            self.embed_dim // 12 +  # is_partner
            self.embed_dim // 12 +  # is_remaining
            self.embed_dim // 6 +   # token_type
            self.embed_dim // 6 +   # decl
            self.embed_dim // 12    # leader
        )

    def forward(
        self,
        tokens: torch.Tensor,      # (batch, max_tokens, n_features)
        mask: torch.Tensor,        # (batch, max_tokens)
        current_player: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            logits: (batch, 7) - logits for each of current player's dominoes
        """
        batch_size = tokens.size(0)
        device = tokens.device

        # Embed each feature
        embeds = []
        embeds.append(self.high_pip_embed(tokens[:, :, 0]))
        embeds.append(self.low_pip_embed(tokens[:, :, 1]))
        embeds.append(self.is_double_embed(tokens[:, :, 2]))
        embeds.append(self.count_value_embed(tokens[:, :, 3]))
        embeds.append(self.trump_rank_embed(tokens[:, :, 4]))
        embeds.append(self.player_id_embed(tokens[:, :, 5]))
        embeds.append(self.is_current_embed(tokens[:, :, 6]))
        embeds.append(self.is_partner_embed(tokens[:, :, 7]))
        embeds.append(self.is_remaining_embed(tokens[:, :, 8]))
        embeds.append(self.token_type_embed(tokens[:, :, 9]))
        embeds.append(self.decl_embed(tokens[:, :, 10]))
        embeds.append(self.leader_embed(tokens[:, :, 11]))

        # Concatenate and project
        x = torch.cat(embeds, dim=-1)  # (batch, max_tokens, input_dim)
        x = self.input_proj(x)         # (batch, max_tokens, embed_dim)

        # Create attention mask (True = ignore)
        attn_mask = (mask == 0)  # (batch, max_tokens)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract current player's 7 hand token representations (vectorized)
        # Hand tokens are at indices 1 + player*7 through 1 + player*7 + 6
        # (index 0 is context token)

        # Create index tensor for gathering: (batch, 7)
        # start_idx = 1 + current_player * 7
        start_indices = 1 + current_player * 7  # (batch,)
        offsets = torch.arange(7, device=device)  # (7,)
        gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)  # (batch, 7)

        # Expand for gathering along embed_dim: (batch, 7, embed_dim)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        # Gather the hand representations
        hand_repr = torch.gather(x, dim=1, index=gather_indices)  # (batch, 7, embed_dim)

        # Project to logits
        logits = self.output_proj(hand_repr).squeeze(-1)  # (batch, 7)

        return logits


# =============================================================================
# Data Processing (Vectorized)
# =============================================================================

def process_file_vectorized(
    file_path: Path,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Process one parquet file with vectorized operations.

    Returns:
        tokens: (N, 32, 12)
        masks: (N, 32)
        current_players: (N,)
        targets: (N,) - optimal local move index
        legal_masks: (N, 7) - 1 if legal, 0 if not
    """
    try:
        pf = pq.ParquetFile(file_path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(file_path)
        states = df["state"].values.astype(np.int64)

        # Get Q-values for finding optimal move
        mv_cols = [f"mv{i}" for i in range(7)]
        q_values_all = np.stack([df[c].values for c in mv_cols], axis=1)  # (N, 7)

        hands = deal_from_seed(seed)
        hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])  # (28,)

        n_states = len(states)

        # Sample if needed
        if max_samples and n_states > max_samples:
            indices = rng.choice(n_states, size=max_samples, replace=False)
            states = states[indices]
            q_values_all = q_values_all[indices]

        n_samples = len(states)

        # Find legal masks and optimal moves (vectorized)
        legal_masks = (q_values_all != -128).astype(np.float32)  # (N, 7)

        # Filter samples with at least one legal move
        has_legal = legal_masks.any(axis=1)
        if not has_legal.any():
            return None

        states = states[has_legal]
        q_values_all = q_values_all[has_legal]
        legal_masks = legal_masks[has_legal]
        n_samples = len(states)

        # Find optimal move: argmax over legal Q-values
        # Convert to int32 to avoid overflow when masking with -129
        q_int32 = q_values_all.astype(np.int32)
        q_masked = np.where(legal_masks > 0, q_int32, -129)
        targets = q_masked.argmax(axis=1).astype(np.int64)  # (N,)

        # ===== Vectorized tokenization =====

        # Extract state fields (vectorized bit operations)
        remaining = np.zeros((n_samples, 4), dtype=np.int64)
        for p in range(4):
            remaining[:, p] = (states >> (p * 7)) & 0x7F

        leader = ((states >> 28) & 0x3).astype(np.int64)  # (N,)
        trick_len = ((states >> 30) & 0x3).astype(np.int64)  # (N,)
        p0_local = ((states >> 32) & 0x7).astype(np.int64)  # (N,)
        p1_local = ((states >> 35) & 0x7).astype(np.int64)  # (N,)
        p2_local = ((states >> 38) & 0x7).astype(np.int64)  # (N,)

        current_player = ((leader + trick_len) % 4).astype(np.int64)  # (N,)
        partner = ((current_player + 2) % 4).astype(np.int64)  # (N,)

        # Precompute static domino features (same for all samples in this file)
        # For each of 28 hand positions: [high, low, is_double, count_val, trump_rank]
        domino_features = np.zeros((28, 5), dtype=np.int64)
        for i, global_id in enumerate(hands_flat):
            domino_features[i, 0] = DOMINO_HIGH[global_id]
            domino_features[i, 1] = DOMINO_LOW[global_id]
            domino_features[i, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            domino_features[i, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            domino_features[i, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]

        # Build tokens array: (N, 32, 12)
        MAX_TOKENS = 32
        N_FEATURES = 12
        tokens = np.zeros((n_samples, MAX_TOKENS, N_FEATURES), dtype=np.int64)
        masks = np.zeros((n_samples, MAX_TOKENS), dtype=np.float32)

        # Token 0: Context
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = leader
        masks[:, 0] = 1.0

        # Tokens 1-28: Hand positions
        for p in range(4):
            for local_idx in range(7):
                flat_idx = p * 7 + local_idx
                token_idx = 1 + flat_idx
                global_id = hands_flat[flat_idx]

                # Static domino features
                tokens[:, token_idx, 0] = domino_features[flat_idx, 0]  # high_pip
                tokens[:, token_idx, 1] = domino_features[flat_idx, 1]  # low_pip
                tokens[:, token_idx, 2] = domino_features[flat_idx, 2]  # is_double
                tokens[:, token_idx, 3] = domino_features[flat_idx, 3]  # count_value
                tokens[:, token_idx, 4] = domino_features[flat_idx, 4]  # trump_rank
                tokens[:, token_idx, 5] = p  # player_id

                # Dynamic per-sample features
                tokens[:, token_idx, 6] = (current_player == p).astype(np.int64)  # is_current
                tokens[:, token_idx, 7] = (partner == p).astype(np.int64)  # is_partner
                tokens[:, token_idx, 8] = ((remaining[:, p] >> local_idx) & 1).astype(np.int64)  # is_remaining
                tokens[:, token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
                tokens[:, token_idx, 10] = decl_id
                tokens[:, token_idx, 11] = leader

                masks[:, token_idx] = 1.0

        # Tokens 29-31: Trick plays (conditional)
        # This is trickier because different samples have different trick lengths
        # We'll process each possible trick position

        trick_plays = [p0_local, p1_local, p2_local]
        trick_token_types = [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P1, TOKEN_TYPE_TRICK_P2]

        for trick_pos in range(3):
            # Mask for samples that have a play at this position
            has_play = (trick_len > trick_pos) & (trick_plays[trick_pos] < 7)

            if not has_play.any():
                continue

            token_idx = 29 + trick_pos
            local_idx = trick_plays[trick_pos]
            play_player = (leader + trick_pos) % 4

            # For each sample with a play at this position, fill in the token
            for sample_idx in np.where(has_play)[0]:
                pp = int(play_player[sample_idx])
                li = int(local_idx[sample_idx])
                global_id = hands[pp][li]

                tokens[sample_idx, token_idx, 0] = DOMINO_HIGH[global_id]
                tokens[sample_idx, token_idx, 1] = DOMINO_LOW[global_id]
                tokens[sample_idx, token_idx, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
                tokens[sample_idx, token_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
                tokens[sample_idx, token_idx, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]
                tokens[sample_idx, token_idx, 5] = pp
                tokens[sample_idx, token_idx, 6] = 1 if pp == current_player[sample_idx] else 0
                tokens[sample_idx, token_idx, 7] = 1 if pp == partner[sample_idx] else 0
                tokens[sample_idx, token_idx, 8] = 0  # not in hand anymore
                tokens[sample_idx, token_idx, 9] = trick_token_types[trick_pos]
                tokens[sample_idx, token_idx, 10] = decl_id
                tokens[sample_idx, token_idx, 11] = leader[sample_idx]

                masks[sample_idx, token_idx] = 1.0

        return (tokens, masks, current_player, targets, legal_masks)

    except Exception as e:
        log(f"  ERROR processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_data(
    data_dir: Path,
    seed_range: tuple[int, int],
    max_samples_per_file: int | None,
    max_files: int | None,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data from parquet files for given seed range."""

    log(f"\n=== Loading {label} Data (seeds {seed_range[0]}-{seed_range[1]}) ===")

    parquet_files = sorted(data_dir.glob("seed_*.parquet"))

    # Filter to relevant files
    relevant_files = []
    for f in parquet_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                seed = int(parts[1])
                if seed_range[0] <= seed <= seed_range[1]:
                    relevant_files.append(f)
            except ValueError:
                pass

    # Limit number of files if specified
    if max_files and len(relevant_files) > max_files:
        relevant_files = relevant_files[:max_files]
        log(f"Found {len(relevant_files)} files (limited to {max_files})")
    else:
        log(f"Found {len(relevant_files)} files")

    all_tokens = []
    all_masks = []
    all_players = []
    all_targets = []
    all_legal = []

    t0 = time.time()
    for i, f in enumerate(relevant_files):
        result = process_file_vectorized(f, max_samples_per_file, rng)
        if result is not None:
            tokens, masks, players, targets, legal = result
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_players.append(players)
            all_targets.append(targets)
            all_legal.append(legal)

            if (i + 1) % 10 == 0 or i == len(relevant_files) - 1:
                total_samples = sum(len(t) for t in all_targets)
                log(f"  [{i+1}/{len(relevant_files)}] {f.name}: {len(tokens):,} samples "
                    f"(total: {total_samples:,}, {time.time()-t0:.1f}s)")

    if not all_tokens:
        raise ValueError(f"No data loaded for {label}")

    return (
        np.concatenate(all_tokens, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_players, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_legal, axis=0),
    )


# =============================================================================
# Training
# =============================================================================

def compute_accuracy(
    model: DominoTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute accuracy and loss on a data loader."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for tokens, masks, players, targets, legal in loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            targets = targets.to(device)
            legal = legal.to(device)

            logits = model(tokens, masks, players)

            # Mask illegal moves
            logits = logits.masked_fill(legal == 0, float('-inf'))

            # Loss
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * len(targets)

            # Accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return acc, avg_loss


def train_epoch(
    model: DominoTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch, return (accuracy, loss)."""
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for tokens, masks, players, targets, legal in loader:
        tokens = tokens.to(device)
        masks = masks.to(device)
        players = players.to(device)
        targets = targets.to(device)
        legal = legal.to(device)

        optimizer.zero_grad()
        logits = model(tokens, masks, players)

        # Mask illegal moves before loss
        logits_masked = logits.masked_fill(legal == 0, float('-inf'))

        loss = F.cross_entropy(logits_masked, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

        # Accuracy
        with torch.no_grad():
            preds = logits_masked.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return acc, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Transformer move prediction diagnostic")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per file (default: None = all)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files to load per split (default: None = all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log("Using CPU")

    # Print configuration
    log("\n=== Configuration ===")
    log(f"Max samples per file: {args.max_samples or 'all'}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Epochs: {args.epochs}")
    log(f"Architecture: {args.n_layers} layers, {args.n_heads} heads, {args.embed_dim} dim")

    data_dir = Path(args.data_dir)

    # Load data
    # Train: seeds 0-89, Test: seeds 90-99
    train_tokens, train_masks, train_players, train_targets, train_legal = load_data(
        data_dir, (0, 89), args.max_samples, args.max_files, rng, "Train"
    )

    # Use proportional max_files for test (10% of train seeds)
    test_max_files = max(1, args.max_files // 9) if args.max_files else None
    test_tokens, test_masks, test_players, test_targets, test_legal = load_data(
        data_dir, (90, 99), args.max_samples, test_max_files, rng, "Test"
    )

    log(f"\n=== Data Summary ===")
    log(f"Train samples: {len(train_targets):,}")
    log(f"Test samples: {len(test_targets):,}")
    log(f"Token shape: {train_tokens.shape}")

    # Analyze target distribution
    train_target_counts = np.bincount(train_targets, minlength=7)
    log(f"Train target distribution: {train_target_counts}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_tokens),
        torch.tensor(train_masks),
        torch.tensor(train_players),
        torch.tensor(train_targets),
        torch.tensor(train_legal),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_tokens),
        torch.tensor(test_masks),
        torch.tensor(test_players),
        torch.tensor(test_targets),
        torch.tensor(test_legal),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=0,
    )

    # Create model
    log("\n=== Creating Model ===")
    model = DominoTransformer(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    log("\n=== Training ===")
    log(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>10} | {'Test Acc':>9} | {'Gap':>5}")
    log("-" * 65)

    best_test_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
        test_acc, test_loss = compute_accuracy(model, test_loader, device)

        scheduler.step()

        gap = train_acc / test_acc if test_acc > 0 else float('inf')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc

        log(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_loss:10.4f} | {test_acc:8.2%} | {gap:5.2f}x")

    # Final results
    log("\n=== Final Results ===")
    log(f"Best train accuracy: {best_train_acc:.2%}")
    log(f"Best test accuracy:  {best_test_acc:.2%}")

    if best_test_acc > 0:
        gap = best_train_acc / best_test_acc
        log(f"Generalization gap:  {gap:.2f}x")

        log("\n=== Success Criteria ===")
        log(f"MLP baseline gap: 1.8x")
        log(f"Target gap:       < 1.3x")
        log(f"Achieved gap:     {gap:.2f}x")

        if gap < 1.3:
            log("\n✓ SUCCESS: Transformer generalizes across seeds!")
        elif gap < 1.5:
            log("\n~ PARTIAL: Better than MLP but not at target")
        else:
            log("\n✗ FAIL: Gap still too large")

    # Spot check predictions
    log("\n=== Spot Check (20 random test samples) ===")
    model.eval()
    indices = rng.choice(len(test_targets), size=min(20, len(test_targets)), replace=False)

    log(f"{'True':>6} | {'Pred':>6} | {'Correct':>7} | {'Legal Moves':>12}")
    log("-" * 40)

    correct_count = 0
    with torch.no_grad():
        for idx in indices:
            tokens = torch.tensor(test_tokens[idx:idx+1]).to(device)
            masks = torch.tensor(test_masks[idx:idx+1]).to(device)
            players = torch.tensor(test_players[idx:idx+1]).to(device)
            legal = torch.tensor(test_legal[idx:idx+1]).to(device)

            logits = model(tokens, masks, players)
            logits = logits.masked_fill(legal == 0, float('-inf'))
            pred = logits.argmax(dim=-1).item()
            true = test_targets[idx]

            correct = "✓" if pred == true else "✗"
            if pred == true:
                correct_count += 1

            legal_str = "".join(str(i) for i in range(7) if test_legal[idx][i] > 0)
            log(f"{true:6d} | {pred:6d} | {correct:>7} | {legal_str:>12}")

    log(f"\nSpot check accuracy: {correct_count}/20 = {correct_count/20:.0%}")


if __name__ == "__main__":
    main()
